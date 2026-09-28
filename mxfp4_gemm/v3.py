from cuda.bindings.driver import CUstream

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.experimental import primitives as prims
from cutlass.utils import HardwareInfo


class MxFp4Gemm:
    def __init__(self):
        # One CTA owns 128 rows; a pair computes a 256x256 output tile.
        self.BM = 128
        self.BN = 256
        self.BK = 256
        self.group_m = 2 * self.BM
        self.b_rows = self.BN // 2
        self.num_stages = 6
        self.acc_stages = 2

        self.row_bytes = self.BK // 2  # Two FP4 values per byte.
        self.sf_blocks = self.BK // 128  # Each scale block covers 128 rows x 128 K.
        self.sf_bytes = self.sf_blocks * 512
        self.b_scale_groups = self.BN // 128
        self.sfb_bytes = self.b_scale_groups * self.sf_bytes
        # TMA and MMA encode the same swizzle with different enums.
        self.tma_swizzle = cuda.TensorMapSwizzle.s128b
        self.mma_swizzle = prims.Tcgen05SmemSwizzle.SWIZZLE_128B

        # Overlap 32 result columns to leave space for 24 scale columns in TMEM.
        self.overlap_cols = 32
        self.acc_stride = self.BN - self.overlap_cols
        self.tmem_cols = 512

    @cute.jit
    def __call__(self, A, B, sfa, sfb, C, stream: CUstream):
        M, N = C.shape
        tma_A = cuda.create_tensor_map_tiled_from_view(
            A, box_dims=(self.BM, self.row_bytes),
            stride_order=(1, 0), swizzle=self.tma_swizzle,
        )
        tma_B = cuda.create_tensor_map_tiled_from_view(
            B, box_dims=(self.row_bytes, self.b_rows),
            stride_order=(0, 1), swizzle=self.tma_swizzle,
        )

        # Scales are already packed by the benchmark. Int64 only groups eight bytes
        # into each TMA element; it does not change the E8M0 scale values.
        tma_SFA = cuda.create_tensor_map_tiled_from_view(
            cute.recast_tensor(sfa, cutlass.Int64), box_dims=(1, self.sf_bytes // 8),
            stride_order=(1, 0), swizzle=cuda.TensorMapSwizzle.none,
        )
        tma_SFB = cuda.create_tensor_map_tiled_from_view(
            cute.recast_tensor(sfb, cutlass.Int64), box_dims=(self.b_scale_groups, self.sf_bytes // 8),
            stride_order=(1, 0), swizzle=cuda.TensorMapSwizzle.none,
        )
        tiles = cute.ceil_div(M, self.group_m) * cute.ceil_div(N, self.BN)
        # Persistent pairs reuse their buffers as they advance through output tiles.
        clusters = min(tiles, HardwareInfo().get_device_multiprocessor_count() // 2)
        self.mxfp4_kernel(A, C, tma_A, tma_B, tma_SFA, tma_SFB).launch(
            grid=(2 * clusters, 1, 1), block=(6 * 32, 1, 1),
            cluster=(2, 1, 1), stream=stream,
        )

    @cute.kernel
    def mxfp4_kernel(
        self, A, C,
        tma_A: cutlass.GridConstant[cuda.TensorMap],
        tma_B: cutlass.GridConstant[cuda.TensorMap],
        tma_SFA: cutlass.GridConstant[cuda.TensorMap],
        tma_SFB: cutlass.GridConstant[cuda.TensorMap],
    ):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(tid // 32)
        rank = cute.arch.block_idx_in_cluster()
        leader = rank == 0
        cluster_id = cta // 2
        clusters = ctas // 2
        M, N = C.shape
        K = A.shape[1] * 2
        grid_n = cute.ceil_div(N, self.BN)
        tiles = cute.ceil_div(M, self.group_m) * grid_n
        k_tiles = cute.ceil_div(K, self.BK)
        a_bytes = self.BM * self.row_bytes
        b_bytes = self.b_rows * self.row_bytes

        # TMA and MMA walk the same ring independently. A buffer becomes reusable
        # only when Tensor Core hardware signals that its reads have finished.
        sA = cutlass.Array(
            cutlass.Uint8, self.num_stages * a_bytes,
            space=cutlass.AddressSpace.smem, alignment=8 * self.row_bytes,
        )
        sB = cutlass.Array(
            cutlass.Uint8, self.num_stages * b_bytes,
            space=cutlass.AddressSpace.smem, alignment=8 * self.row_bytes,
        )
        sSFA = cutlass.Array(
            cutlass.Uint8, self.num_stages * self.sf_bytes,
            space=cutlass.AddressSpace.smem, alignment=128,
        )
        sSFB = cutlass.Array(
            cutlass.Uint8, self.num_stages * self.sfb_bytes,
            space=cutlass.AddressSpace.smem, alignment=128,
        )

        input_full = cutlass.Array(cutlass.Int64, self.num_stages, space=cutlass.AddressSpace.smem, alignment=8)
        input_empty = cutlass.Array(cutlass.Int64, self.num_stages, space=cutlass.AddressSpace.smem, alignment=8)
        acc_full = cutlass.Array(cutlass.Int64, self.acc_stages, space=cutlass.AddressSpace.smem, alignment=8)
        acc_empty = cutlass.Array(cutlass.Int64, self.acc_stages, space=cutlass.AddressSpace.smem, alignment=8)
        overlap_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        tmem_base = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)

        if tid == 0:
            for stage in cutlass.range_constexpr(self.num_stages):
                prims.mbarrier_init(input_full.data_ptr(stage), 1)
                prims.mbarrier_init(input_empty.data_ptr(stage), 1)
            for stage in cutlass.range_constexpr(self.acc_stages):
                prims.mbarrier_init(acc_full.data_ptr(stage), 1)
                # Both CTAs must finish reading: 128 output threads per CTA.
                prims.mbarrier_init(acc_empty.data_ptr(stage), 256)
            prims.mbarrier_init(overlap_done, 256)
            prims.prefetch_tensormap(tma_A.get_ptr())
            prims.prefetch_tensormap(tma_B.get_ptr())
            prims.prefetch_tensormap(tma_SFA.get_ptr())
            prims.prefetch_tensormap(tma_SFB.get_ptr())

        prims.fence_mbarrier_init()
        prims.barrier_cluster_arrive_relaxed()
        if warp == 5:
            # Allocation is warp-collective; all 32 lanes participate.
            # In EACH CTA: result 0 is 0..255, result 1 is 224..479.
            # Their overlap is 224..255; A scales use 480..487, B scales 488..503.
            # The epilogue reads the overlapping columns FIRST, then releases them.
            prims.tcgen05_alloc(tmem_base, self.tmem_cols, group=prims.CTAGroup.CTA_2)
            prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_2)

        # Neither CTA touches its peer's barriers before both have initialized them.
        prims.barrier_cluster_wait()
        prims.barrier_cta_sync(0)
        base = tmem_base[0]
        sf_base = base + (self.acc_stages - 1) * self.acc_stride + self.BN

        if warp == 4:  # Producer: global memory -> shared memory.
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(0)
            for tile in cutlass.range(cluster_id, tiles, clusters):
                tile_row = (tile // grid_n) * self.group_m + rank * self.BM
                tile_col = (tile % grid_n) * self.BN
                for k_tile in cutlass.range(k_tiles):
                    # Initial barriers are phase 0, so waiting on parity 1 passes
                    # immediately on the first use. Later it waits for the last MMA.
                    while not prims.mbarrier_try_wait_parity(input_empty.data_ptr(stage), phase ^ 1):
                        pass
                    if prims.elect_sync():
                        k_start = k_tile * self.BK
                        sf_start = (k_start // 128) * 64  # 64 Int64 packages per scale block.
                        # The leader counts both CTAs' loads. CTA_2 routes the
                        # peer's completion signals to the leader's barrier.
                        if leader:
                            prims.mbarrier_arrive_expect_tx(
                                input_full.data_ptr(stage), 2 * (a_bytes + b_bytes + self.sf_bytes + self.sfb_bytes),
                            )
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            sA.data_ptr(stage * a_bytes), tma_A.get_ptr(), (k_start // 2, tile_row),
                            input_full.subview(stage), [], group=prims.CTAGroup.CTA_2,
                        )
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            sB.data_ptr(stage * b_bytes), tma_B.get_ptr(), (k_start // 2, tile_col + rank * self.b_rows),
                            input_full.subview(stage), [], group=prims.CTAGroup.CTA_2,
                        )
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            sSFA.data_ptr(stage * self.sf_bytes), tma_SFA.get_ptr(), (sf_start, tile_row // 128),
                            input_full.subview(stage), [], group=prims.CTAGroup.CTA_2,
                        )
                        # Both CTAs need ALL B scales: each outputs all 256 columns.
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            sSFB.data_ptr(stage * self.sfb_bytes), tma_SFB.get_ptr(), (sf_start, tile_col // 128),
                            input_full.subview(stage), [], group=prims.CTAGroup.CTA_2,
                        )
                    stage += 1
                    if stage == self.num_stages:
                        stage = 0
                        phase ^= 1
                    # Do not reset stage/phase between output tiles: the ring keeps running.

        elif warp == 5:  # Consumer: shared A/B + TMEM scales -> TMEM FP32 sums.
            if leader:
                stage = cutlass.Int32(0)
                phase = cutlass.Int32(0)
                acc_stage = cutlass.Int32(0)
                acc_phase = cutlass.Int32(0)
                overlap_phase = cutlass.Int32(0)

                idesc = prims.Tcgen05MxOmmaInstrDesc.build(
                    a_dtype=cutlass.Float4E2M1FN, b_dtype=cutlass.Float4E2M1FN,
                    scale_format=1, m_dim=self.group_m, n_dim=self.BN, k_dim=0, sparsity_version=0,
                )
                for tile in cutlass.range(cluster_id, tiles, clusters):
                    while not prims.mbarrier_try_wait_parity(acc_empty.data_ptr(acc_stage), acc_phase ^ 1):
                        pass  # The output warps may still be reading this result buffer.

                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    d_tmem = prims.TmemAddr(base + acc_stage * self.acc_stride).as_ptr(cutlass.Float32)

                    for k_tile in cutlass.range(k_tiles):
                        while not prims.mbarrier_try_wait_parity(input_full.data_ptr(stage), phase):
                            pass
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

                        if prims.elect_sync():
                            a_desc = prims.Tcgen05SmemDesc.build(
                                sA.data_ptr(stage * a_bytes), leading_byte_offset=16,
                                stride_byte_offset=8 * self.row_bytes, layout=self.mma_swizzle,
                            )
                            b_desc = prims.Tcgen05SmemDesc.build(
                                sB.data_ptr(stage * b_bytes), leading_byte_offset=16,
                                stride_byte_offset=8 * self.row_bytes, layout=self.mma_swizzle,
                            )
                            sfa_desc = prims.Tcgen05SmemDesc.build(
                                sSFA.data_ptr(stage * self.sf_bytes), stride_byte_offset=8 * 16,
                            )
                            sfb_desc = prims.Tcgen05SmemDesc.build(
                                sSFB.data_ptr(stage * self.sfb_bytes), stride_byte_offset=8 * 16,
                            )
                            # CTA_2 copies fetch from BOTH CTAs at matching shared offsets.
                            # A scales differ by row-half; B scales are identical in both CTAs.
                            for sf_block in cutlass.range_constexpr(self.sf_blocks):
                                sfa_tmem = prims.TmemAddr(sf_base + sf_block * 4).as_ptr(cutlass.Int32)
                                prims.tcgen05_cp(
                                    prims.Tcgen05CpShape.SHAPE_32X128B, sfa_tmem,
                                    sfa_desc.advance_start_address(sf_block * 512),
                                    multicast=prims.Tcgen05CpMulticast.WARPX4, group=prims.CTAGroup.CTA_2,
                                )
                                for n_group in cutlass.range_constexpr(self.b_scale_groups):
                                    # SMEM groups by N first. MMA wants both N halves beside
                                    # each other in TMEM for the same K scale block.
                                    sfb_tmem = prims.TmemAddr(
                                        sf_base + self.sf_blocks * 4 + (sf_block * self.b_scale_groups + n_group) * 4,
                                    ).as_ptr(cutlass.Int32)
                                    prims.tcgen05_cp(
                                        prims.Tcgen05CpShape.SHAPE_32X128B, sfb_tmem,
                                        sfb_desc.advance_start_address(n_group * self.sf_bytes + sf_block * 512),
                                        multicast=prims.Tcgen05CpMulticast.WARPX4, group=prims.CTAGroup.CTA_2,
                                    )
                            # BK=256 issues four K=64 MMAs but pays the buffer handoff once.
                            for k_mma in cutlass.range_constexpr(self.BK // 64):
                                sf_id = (k_mma % 2) * 2  # Select byte pair 0,1 or 2,3.
                                mma_desc = idesc | (sf_id << 4) | (sf_id << 29)
                                sfa_tmem = prims.TmemAddr(sf_base + (k_mma // 2) * 4).as_ptr(cutlass.Int32)
                                sfb_tmem = prims.TmemAddr(
                                    sf_base + self.sf_blocks * 4 + (k_mma // 2) * self.b_scale_groups * 4,
                                ).as_ptr(cutlass.Int32)
                                prims.tcgen05_mma_block_scale(
                                    prims.Tcgen05MMAKind.MXF4, prims.CTAGroup.CTA_2,
                                    d_tmem, a_desc.advance_start_address(k_mma * 32),
                                    b_desc.advance_start_address(k_mma * 32), mma_desc,
                                    k_tile > 0 or k_mma > 0, sfa_tmem, sfb_tmem,
                                    scale_vec_size=prims.Tcgen05MMAScaleVecSize.X2,
                                )
                            prims.tcgen05_commit(
                                input_empty.data_ptr(stage),
                                multicast_mask=3, group=prims.CTAGroup.CTA_2,
                            )
                        stage += 1
                        if stage == self.num_stages:
                            stage = 0
                            phase ^= 1
                    if prims.elect_sync():
                        prims.tcgen05_commit(
                            acc_full.data_ptr(acc_stage),
                            multicast_mask=3, group=prims.CTAGroup.CTA_2,
                        )
                    while not prims.mbarrier_try_wait_parity(overlap_done, overlap_phase):
                        pass  # Both CTAs must read the overlap before the next MMA overwrites it.
                    overlap_phase ^= 1
                    acc_stage += 1
                    if acc_stage == self.acc_stages:
                        acc_stage = 0
                        acc_phase ^= 1

        else:  # Warps 0..3: store the previous result while MMA builds the next.
            C_vectors = cute.zipped_divide(C, (1, 16))[(0, None), None]
            store_bf16x16 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=256,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )
            # Both CTAs report their completed reads to the MMA leader.
            leader_overlap_done = prims.mapa(overlap_done.data_ptr(), 0)
            acc_stage = cutlass.Int32(0)
            acc_phase = cutlass.Int32(0)
            for tile in cutlass.range(cluster_id, tiles, clusters):
                row = (tile // grid_n) * self.group_m + rank * self.BM + tid
                tile_col = (tile % grid_n) * self.BN
                while not prims.mbarrier_try_wait_parity(acc_full.data_ptr(acc_stage), acc_phase):
                    pass

                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                for chunk in cutlass.range_constexpr(self.BN // 16):
                    # Read the 32 overlapping columns first, then the remaining 224.
                    if cutlass.const_expr(chunk < self.overlap_cols // 16):
                        col = (1 - acc_stage) * self.acc_stride + chunk * 16
                    else:
                        col = (chunk * 16 - self.overlap_cols) + acc_stage * self.overlap_cols
                    addr = prims.TmemAddr.from_row_col(
                        warp * 32, base + acc_stage * self.acc_stride + col,
                    )
                    values = prims.tcgen05_ld('32x32b', addr.as_ptr(cutlass.Float32), num=16)
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    if cutlass.const_expr(chunk == self.overlap_cols // 16 - 1):
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.mbarrier_arrive(leader_overlap_done)

                    tmp = cute.make_rmem_tensor(16, cutlass.BFloat16)
                    tmp.store(cute.TensorSSA(
                        values.to(cutlass.BFloat16).ir_value(), (16,), cutlass.BFloat16,
                    ))
                    if row < M and tile_col + col + 16 <= N:
                        cute.copy(store_bf16x16, tmp, C_vectors[None, (row, (tile_col + col) // 16)])

                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                prims.mbarrier_arrive(prims.mapa(acc_empty.data_ptr(acc_stage), 0))
                acc_stage += 1
                if acc_stage == self.acc_stages:
                    acc_stage = 0
                    acc_phase ^= 1

        # Only synchronize the entire block after all three warp roles finish.
        prims.barrier_cta_sync(0)
        prims.barrier_cluster_arrive_relaxed()
        prims.barrier_cluster_wait()  # Keep both CTAs alive until peer accesses finish.
        if warp == 5:
            prims.tcgen05_dealloc(
                prims.TmemAddr(base).as_ptr(cutlass.Float32), self.tmem_cols,
                group=prims.CTAGroup.CTA_2,
            )


mxfp4 = MxFp4Gemm()