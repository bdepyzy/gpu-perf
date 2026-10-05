from cuda.bindings.driver import CUstream

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.experimental import primitives as prims
from cutlass.utils import HardwareInfo


class MxFp4Gemm:
    def __init__(self, bk=256, stages=3, persistent=True, overlap=True, bn=256):
        self.BM = 128
        self.BN = bn
        self.BK = bk
        self.num_stages = stages
        self.persistent = persistent
        self.acc_stages = 2 if overlap else 1
        assert bk in (128, 256)
        assert bn in (128, 256)
        assert stages in (2, 3, 4)
        self.row_bytes = bk // 2
        self.sf_blocks = bk // 128
        self.sf_bytes = self.sf_blocks * 512
        self.b_scale_groups = bn // 128
        self.sfb_bytes = self.b_scale_groups * self.sf_bytes
        self.tma_swizzle = cuda.TensorMapSwizzle.s64b if bk == 128 else cuda.TensorMapSwizzle.s128b
        self.mma_swizzle = prims.Tcgen05SmemSwizzle.SWIZZLE_64B if bk == 128 else prims.Tcgen05SmemSwizzle.SWIZZLE_128B

        scale_cols = self.sf_blocks * 4 * (1 + self.b_scale_groups)
        self.overlap_cols = ((scale_cols + 15) // 16) * 16 if overlap else 0
        self.acc_stride = self.BN - self.overlap_cols

        self.tmem_cols = 2 * bn

    @cute.jit
    def __call__(self, A, B, sfa, sfb, C, stream: CUstream):
        M, N = C.shape
        tma_A = cuda.create_tensor_map_tiled_from_view(A, box_dims=(self.BM, self.row_bytes), stride_order=(1, 0), swizzle=self.tma_swizzle,)
        tma_B = cuda.create_tensor_map_tiled_from_view(B, box_dims=(self.row_bytes, self.BN), stride_order=(0, 1), swizzle=self.tma_swizzle,)

        tma_SFA = cuda.create_tensor_map_tiled_from_view(
            cute.recast_tensor(sfa, cutlass.Int64), box_dims=(1, self.sf_bytes // 8),
            stride_order=(1, 0), swizzle=cuda.TensorMapSwizzle.none,
        )
        tma_SFB = cuda.create_tensor_map_tiled_from_view(

            cute.recast_tensor(sfb, cutlass.Int64), box_dims=(self.b_scale_groups, self.sf_bytes // 8),
            stride_order=(1, 0), swizzle=cuda.TensorMapSwizzle.none,
        )
        tiles = cute.ceil_div(M, self.BM) * cute.ceil_div(N, self.BN)
        ctas = tiles
        if cutlass.const_expr(self.persistent):
            ctas = min(tiles, (512 // self.tmem_cols) * HardwareInfo().get_device_multiprocessor_count())
        self.mxfp4_kernel(A, C, tma_A, tma_B, tma_SFA, tma_SFB).launch(
            grid=(ctas, 1, 1), block=(6 * 32, 1, 1), stream=stream,
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
        M, N = C.shape
        K = A.shape[1] * 2
        grid_n = cute.ceil_div(N, self.BN)
        tiles = cute.ceil_div(M, self.BM) * grid_n
        k_tiles = cute.ceil_div(K, self.BK)
        a_bytes = self.BM * self.row_bytes
        b_bytes = self.BN * self.row_bytes

        sA = cutlass.Array(cutlass.Uint8, self.num_stages * a_bytes, space=cutlass.AddressSpace.smem, alignment=8 * self.row_bytes)
        sB = cutlass.Array(cutlass.Uint8, self.num_stages * b_bytes, space=cutlass.AddressSpace.smem, alignment=8 * self.row_bytes)
        sSFA = cutlass.Array(cutlass.Uint8, self.num_stages * self.sf_bytes, space=cutlass.AddressSpace.smem, alignment=128)
        sSFB = cutlass.Array(cutlass.Uint8, self.num_stages * self.sfb_bytes, space=cutlass.AddressSpace.smem, alignment=128)

        full = cutlass.Array(cutlass.Int64, self.num_stages, space=cutlass.AddressSpace.smem, alignment=8)
        empty = cutlass.Array(cutlass.Int64, self.num_stages, space=cutlass.AddressSpace.smem, alignment=8)
        acc_full = cutlass.Array(cutlass.Int64, self.acc_stages, space=cutlass.AddressSpace.smem, alignment=8)
        acc_empty = cutlass.Array(cutlass.Int64, self.acc_stages, space=cutlass.AddressSpace.smem, alignment=8)
        partial = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        tmem_base = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)

        if tid == 0:
            for stage in cutlass.range_constexpr(self.num_stages):
                prims.mbarrier_init(full.data_ptr(stage), 1)
                prims.mbarrier_init(empty.data_ptr(stage), 1)
            for stage in cutlass.range_constexpr(self.acc_stages):
                prims.mbarrier_init(acc_full.data_ptr(stage), 1)

                prims.mbarrier_init(acc_empty.data_ptr(stage), 128)
            prims.mbarrier_init(partial, 128)
            prims.prefetch_tensormap(tma_A.get_ptr())
            prims.prefetch_tensormap(tma_B.get_ptr())
            prims.prefetch_tensormap(tma_SFA.get_ptr())
            prims.prefetch_tensormap(tma_SFB.get_ptr())

        if warp == 5:
            prims.tcgen05_alloc(tmem_base, self.tmem_cols)
            prims.tcgen05_relinquish_alloc_permit()

        prims.fence_mbarrier_init()
        prims.barrier_cta_sync(0)
        base = tmem_base[0]
        sf_base = base + (self.acc_stages - 1) * self.acc_stride + self.BN

        if warp == 4:
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(0)
            for tile in cutlass.range(cta, tiles, ctas):
                tile_row = (tile // grid_n) * self.BM
                tile_col = (tile % grid_n) * self.BN
                for k_tile in cutlass.range(k_tiles):
                    while not prims.mbarrier_try_wait_parity(empty.data_ptr(stage), phase ^ 1):
                        pass
                    if prims.elect_sync():
                        k_start = k_tile * self.BK
                        sf_start = (k_start // 128) * 64
                        prims.mbarrier_arrive_expect_tx(full.data_ptr(stage), a_bytes + b_bytes + self.sf_bytes + self.sfb_bytes)
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sA.data_ptr(stage * a_bytes), tma_A.get_ptr(), (k_start // 2, tile_row), full.data_ptr(stage),
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sB.data_ptr(stage * b_bytes), tma_B.get_ptr(), (k_start // 2, tile_col), full.data_ptr(stage),
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sSFA.data_ptr(stage * self.sf_bytes), tma_SFA.get_ptr(), (sf_start, tile_row // 128), full.data_ptr(stage),
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sSFB.data_ptr(stage * self.sfb_bytes), tma_SFB.get_ptr(), (sf_start, tile_col // 128), full.data_ptr(stage),
                        )
                    stage += 1
                    if stage == self.num_stages:
                        stage = 0
                        phase ^= 1

        elif warp == 5:
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(0)
            acc_stage = cutlass.Int32(0)
            acc_phase = cutlass.Int32(0)
            partial_phase = cutlass.Int32(0)

            idesc = prims.Tcgen05MxOmmaInstrDesc.build(
                a_dtype=cutlass.Float4E2M1FN, b_dtype=cutlass.Float4E2M1FN,
                scale_format=1, m_dim=self.BM, n_dim=self.BN, k_dim=0, sparsity_version=0,
            )
            for tile in cutlass.range(cta, tiles, ctas):
                while not prims.mbarrier_try_wait_parity(acc_empty.data_ptr(acc_stage), acc_phase ^ 1):
                    pass

                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                d_tmem = prims.TmemAddr(base + acc_stage * self.acc_stride).as_ptr(cutlass.Float32)

                for k_tile in cutlass.range(k_tiles):
                    while not prims.mbarrier_try_wait_parity(full.data_ptr(stage), phase):
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
                        sfa_desc = prims.Tcgen05SmemDesc.build(sSFA.data_ptr(stage * self.sf_bytes), stride_byte_offset=8 * 16,)
                        sfb_desc = prims.Tcgen05SmemDesc.build(sSFB.data_ptr(stage * self.sfb_bytes), stride_byte_offset=8 * 16,)
                        for sf_block in cutlass.range_constexpr(self.sf_blocks):
                            sfa_tmem = prims.TmemAddr(sf_base + sf_block * 4).as_ptr(cutlass.Int32)
                            prims.tcgen05_cp(
                                prims.Tcgen05CpShape.SHAPE_32X128B, sfa_tmem,
                                sfa_desc.advance_start_address(sf_block * 512), multicast=prims.Tcgen05CpMulticast.WARPX4,
                            )
                            for n_group in cutlass.range_constexpr(self.b_scale_groups):
                                sfb_tmem = prims.TmemAddr(sf_base + self.sf_blocks * 4 + (sf_block * self.b_scale_groups + n_group) * 4).as_ptr(cutlass.Int32)
                                prims.tcgen05_cp(
                                    prims.Tcgen05CpShape.SHAPE_32X128B, sfb_tmem,
                                    sfb_desc.advance_start_address(n_group * self.sf_bytes + sf_block * 512),
                                    multicast=prims.Tcgen05CpMulticast.WARPX4,
                                )

                        for k_mma in cutlass.range_constexpr(self.BK // 64):
                            sf_id = (k_mma % 2) * 2
                            mma_desc = idesc | (sf_id << 4) | (sf_id << 29)
                            sfa_tmem = prims.TmemAddr(sf_base + (k_mma // 2) * 4).as_ptr(cutlass.Int32)
                            sfb_tmem = prims.TmemAddr(sf_base + self.sf_blocks * 4 + (k_mma // 2) * self.b_scale_groups * 4).as_ptr(cutlass.Int32)
                            prims.tcgen05_mma_block_scale(
                                prims.Tcgen05MMAKind.MXF4, prims.CTAGroup.CTA_1,
                                d_tmem, a_desc.advance_start_address(k_mma * 32),
                                b_desc.advance_start_address(k_mma * 32), mma_desc,
                                k_tile > 0 or k_mma > 0, sfa_tmem, sfb_tmem,
                                scale_vec_size=prims.Tcgen05MMAScaleVecSize.X2,
                            )
                        prims.tcgen05_commit(empty.data_ptr(stage))
                    stage += 1
                    if stage == self.num_stages:
                        stage = 0
                        phase ^= 1
                if prims.elect_sync():
                    prims.tcgen05_commit(acc_full.data_ptr(acc_stage))
                if cutlass.const_expr(self.acc_stages == 2):
                    while not prims.mbarrier_try_wait_parity(partial, partial_phase):
                        pass
                    partial_phase ^= 1
                acc_stage += 1
                if acc_stage == self.acc_stages:
                    acc_stage = 0
                    acc_phase ^= 1

        else:
            C_vectors = cute.zipped_divide(C, (1, 16))[(0, None), None]
            store_bf16x16 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=256,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )
            acc_stage = cutlass.Int32(0)
            acc_phase = cutlass.Int32(0)
            for tile in cutlass.range(cta, tiles, ctas):
                row = (tile // grid_n) * self.BM + tid
                tile_col = (tile % grid_n) * self.BN
                while not prims.mbarrier_try_wait_parity(acc_full.data_ptr(acc_stage), acc_phase):
                    pass

                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if cutlass.const_expr(self.acc_stages == 2):
                    for shared_col in cutlass.range_constexpr(0, self.overlap_cols, 16):
                        addr = prims.TmemAddr.from_row_col(warp * 32, base + self.acc_stride + shared_col)
                        values = prims.tcgen05_ld('32x32b', addr.as_ptr(cutlass.Float32), num=16)
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        if cutlass.const_expr(shared_col + 16 == self.overlap_cols):
                            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                            prims.mbarrier_arrive(partial)
                        tmp = cute.make_rmem_tensor(16, cutlass.BFloat16)
                        tmp.store(cute.TensorSSA(values.to(cutlass.BFloat16).ir_value(), (16,), cutlass.BFloat16))

                        col = tile_col + (1 - acc_stage) * self.acc_stride + shared_col
                        if row < M and col + 16 <= N:
                            cute.copy(store_bf16x16, tmp, C_vectors[None, (row, col // 16)])

                for col_start in cutlass.range_constexpr(0, self.acc_stride, 16):
                    addr = prims.TmemAddr.from_row_col(warp * 32, base + acc_stage * self.BN + col_start)
                    values = prims.tcgen05_ld('32x32b', addr.as_ptr(cutlass.Float32), num=16)
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    tmp = cute.make_rmem_tensor(16, cutlass.BFloat16)
                    tmp.store(cute.TensorSSA(values.to(cutlass.BFloat16).ir_value(), (16,), cutlass.BFloat16))
                    col = tile_col + col_start + acc_stage * self.overlap_cols
                    if row < M and col + 16 <= N:
                        cute.copy(store_bf16x16, tmp, C_vectors[None, (row, col // 16)])

                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                prims.mbarrier_arrive(acc_empty.data_ptr(acc_stage))
                acc_stage += 1
                if acc_stage == self.acc_stages:
                    acc_stage = 0
                    acc_phase ^= 1

        prims.barrier_cta_sync(0)
        if warp == 5:
            prims.tcgen05_dealloc(prims.TmemAddr(base).as_ptr(cutlass.Float32), self.tmem_cols)


mxfp4 = MxFp4Gemm()
