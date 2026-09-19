from cuda.bindings.driver import CUstream

import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.cute as cute

from cutlass.experimental import primitives as prims

SCALE_GROUP_SIZE = 32  # MXFP4 uses one scale for every 32 K values.

class MxFp4Gemm:
    def __init__(self):
        self.BM = self.BN = 128
        self.BK = 128
        # Four output warps, one TMA warp, one MMA warp.
        self.threads_per_cta = 6 * 32

    @cute.jit
    def __call__(self, A, B, sfa, sfb, C, stream: CUstream):
        M, N = C.shape

        tma_desc_A = cuda.create_tensor_map_tiled_from_view(
            A,
            box_dims=(self.BM, self.BK // 2),  # Two FP4 values fit in each byte.
            stride_order=(1, 0),  # TMA coordinates are (packed K position, row).
            swizzle=cuda.TensorMapSwizzle.s64b,  # BK=128 occupies 64 packed bytes per row.
        )
        
        tma_desc_B = cuda.create_tensor_map_tiled_from_view(
            B,
            box_dims=(self.BK // 2, self.BN),  # Two FP4 values fit in each byte.
            stride_order=(0, 1),  # B's TMA coordinates are (packed K byte, column).
            swizzle=cuda.TensorMapSwizzle.s64b,  # Match the Tensor Core shared-memory descriptor.
        )

        # The benchmark has already rearranged scales into 128x4 blocks.
        # Recast groups eight scale bytes into one Int64 WITHOUT changing their bits.
        # One block is now 64 Int64 elements, below TMA's 256-element box limit.
        tma_desc_SFA = cuda.create_tensor_map_tiled_from_view(
            cute.recast_tensor(sfa, cutlass.Int64),
            box_dims=(1, 64),  # One group of 128 rows, one complete 512-byte block.
            stride_order=(1, 0),  # Coordinates: (eight-byte package, row group).
            swizzle=cuda.TensorMapSwizzle.none,
        )
        tma_desc_SFB = cuda.create_tensor_map_tiled_from_view(
            cute.recast_tensor(sfb, cutlass.Int64),
            box_dims=(1, 64),  # Same layout, but the 128 rows represent B columns.
            stride_order=(1, 0),
            swizzle=cuda.TensorMapSwizzle.none,
        )

        self.mxfp4_kernel(A, B, C,
                          tma_desc_A, tma_desc_B, 
                          tma_desc_SFA, tma_desc_SFB
        ).launch(
            grid=(cute.ceil_div(N, self.BN), cute.ceil_div(M, self.BM), 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
        )
        
    @cute.kernel
    def mxfp4_kernel(
        self, A, B, C,
        tma_desc_A: cutlass.GridConstant[cuda.TensorMap],
        tma_desc_B: cutlass.GridConstant[cuda.TensorMap],
        tma_desc_SFA: cutlass.GridConstant[cuda.TensorMap],
        tma_desc_SFB: cutlass.GridConstant[cuda.TensorMap],
    ):
        tid, _, _ = cute.arch.thread_idx()
        block_x, block_y, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        M, N = C.shape
        K = A.shape[1] * 2
        tile_row = block_y * self.BM
        tile_col = block_x * self.BN

        # Two buffers: TMA can fill one while MMA consumes the other.
        # Each K=128 row contains 64 packed bytes. TMA applies a 64-byte swizzle;
        # the MMA descriptor below describes that same layout to the Tensor Core.
        a_bytes = self.BM * (self.BK // 2)
        b_bytes = self.BN * (self.BK // 2)
        # A 64-byte swizzle repeats every 512 bytes, giving these buffers base offset zero.
        sA = cutlass.Array(cutlass.Uint8, 2 * a_bytes, space=cutlass.AddressSpace.smem, alignment=512)
        sB = cutlass.Array(cutlass.Uint8, 2 * b_bytes, space=cutlass.AddressSpace.smem, alignment=512)
        sSFA = cutlass.Array(cutlass.Uint8, 2 * 512, space=cutlass.AddressSpace.smem, alignment=128)
        sSFB = cutlass.Array(cutlass.Uint8, 2 * 512, space=cutlass.AddressSpace.smem, alignment=128)

        # full: TMA finished writing this buffer. empty: MMA finished reading it.
        mbar_full = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
        mbar_empty = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
        mbar_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        tmem_base = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)

        if tid == 0:
            for stage in cutlass.range_constexpr(2):
                prims.mbarrier_init(mbar_full.data_ptr(stage), 1)
                prims.mbarrier_init(mbar_empty.data_ptr(stage), 1)
            prims.mbarrier_init(mbar_done, 1)
            prims.prefetch_tensormap(tma_desc_A.get_ptr())
            prims.prefetch_tensormap(tma_desc_B.get_ptr())
            prims.prefetch_tensormap(tma_desc_SFA.get_ptr())
            prims.prefetch_tensormap(tma_desc_SFB.get_ptr())

        is_epi = warp_id < 4
        is_tma = warp_id == 4
        is_tc = warp_id == 5

        if is_tc:
            # ALL 32 lanes must allocate together; do not put alloc inside elect_sync.
            # Columns 0..127 hold FP32 sums; 128..131 and 132..135 hold scales.
            # Allocation sizes are powers of two, so 136 used columns require 256.
            prims.tcgen05_alloc(tmem_base, 256)
            prims.tcgen05_relinquish_alloc_permit()

        prims.fence_mbarrier_init()
        prims.barrier_cta_sync(0)  # Publish initialized barriers and the TMEM address.
        base = tmem_base[0]

        if is_tma:
            # Before: the TMA branch used packed_k_start before its K loop existed.
            # This warp now owns its own loop and never executes the old scalar GEMM.
            for iteration in cutlass.range(0, K // self.BK):
                stage = iteration % 2
                phase = (iteration // 2) % 2
                # Skip the empty wait on the first use: nobody has consumed it yet.
                if iteration >= 2:
                    while not prims.mbarrier_try_wait_parity(mbar_empty.data_ptr(stage), phase ^ 1):
                        pass  # MMA must finish before TMA overwrites this buffer.
                if prims.elect_sync():
                    k_start = iteration * self.BK
                    packed_k_start = k_start // 2
                    # Four scales cover K=128. Both K=64 halves use the same block.
                    sf_package_start = (k_start // 128) * 64
                    # Each buffer gets its own scale copy, so it can progress independently.
                    sz = a_bytes + b_bytes + 512 + 512
                    prims.mbarrier_arrive_expect_tx(mbar_full.data_ptr(stage), sz)
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sA.data_ptr(stage * a_bytes), tma_desc_A.get_ptr(),
                        (packed_k_start, tile_row), mbar_full.data_ptr(stage),
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sB.data_ptr(stage * b_bytes), tma_desc_B.get_ptr(),
                        (packed_k_start, tile_col), mbar_full.data_ptr(stage),
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sSFA.data_ptr(stage * 512), tma_desc_SFA.get_ptr(),
                        (sf_package_start, tile_row // 128), mbar_full.data_ptr(stage),
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sSFB.data_ptr(stage * 512), tma_desc_SFB.get_ptr(),
                        (sf_package_start, tile_col // 128), mbar_full.data_ptr(stage),
                    )

        elif is_tc:
            d_tmem = prims.TmemAddr(base).as_ptr(cutlass.Float32)
            sfa_tmem = prims.TmemAddr(base + 128).as_ptr(cutlass.Int32)
            sfb_tmem = prims.TmemAddr(base + 132).as_ptr(cutlass.Int32)
            # MXFP4: E2M1 operands, E8M0 scales, and one scale per 32 K values.
            # k_dim=0 means a dense K=64 MMA for this FP4 instruction family.
            idesc = prims.Tcgen05MxOmmaInstrDesc.build(
                a_dtype=cutlass.Float4E2M1FN, b_dtype=cutlass.Float4E2M1FN,
                scale_format=1, m_dim=self.BM, n_dim=self.BN,
                k_dim=0, sparsity_version=0,
            )
            for iteration in cutlass.range(0, K // self.BK):
                stage = iteration % 2
                phase = (iteration // 2) % 2
                while not prims.mbarrier_try_wait_parity(mbar_full.data_ptr(stage), phase):
                    pass  # This buffer is not readable until all four TMA copies finish.
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if prims.elect_sync():
                    # Descriptors describe memory; they do not copy or rearrange it.
                    # MMA and TMA have different swizzle enums: use the MMA enum here.
                    a_desc = prims.Tcgen05SmemDesc.build(
                        sA.data_ptr(stage * a_bytes), leading_byte_offset=16,
                        stride_byte_offset=8 * 64, layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
                    )
                    b_desc = prims.Tcgen05SmemDesc.build(
                        sB.data_ptr(stage * b_bytes), leading_byte_offset=16,
                        stride_byte_offset=8 * 64, layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
                    )
                    # A packed scale block has 32 rows of 16 bytes, without a swizzle.
                    sfa_desc = prims.Tcgen05SmemDesc.build(
                        sSFA.data_ptr(stage * 512), stride_byte_offset=8 * 16,
                    )
                    sfb_desc = prims.Tcgen05SmemDesc.build(
                        sSFB.data_ptr(stage * 512), stride_byte_offset=8 * 16,
                    )
                    # Replicate each 512-byte scale block into all four TMEM lane groups.
                    # MMA reads scales from TMEM, while A/B remain in shared memory.
                    prims.tcgen05_cp(
                        prims.Tcgen05CpShape.SHAPE_32X128B, sfa_tmem, sfa_desc,
                        multicast=prims.Tcgen05CpMulticast.WARPX4,
                    )
                    prims.tcgen05_cp(
                        prims.Tcgen05CpShape.SHAPE_32X128B, sfb_tmem, sfb_desc,
                        multicast=prims.Tcgen05CpMulticast.WARPX4,
                    )
                    # SF IDs are byte offsets: 0 selects scales 0,1; 2 selects 2,3.
                    # Bits 4..5 select B's pair; bits 29..30 select A's pair.
                    # Before: one K=64 MMA per iteration skipped half of your new BK=128 tile.
                    for k_mma in cutlass.range_constexpr(self.BK // 64):
                        sf_id = k_mma * 2
                        mma_desc = idesc | (sf_id << 4) | (sf_id << 29)
                        # Advance 64 FP4 values = 32 bytes for the second MMA.
                        prims.tcgen05_mma_block_scale(
                            prims.Tcgen05MMAKind.MXF4, prims.CTAGroup.CTA_1,
                            d_tmem, a_desc.advance_start_address(k_mma * 32),
                            b_desc.advance_start_address(k_mma * 32), mma_desc,
                            iteration > 0 or k_mma > 0, sfa_tmem, sfb_tmem,
                            scale_vec_size=prims.Tcgen05MMAScaleVecSize.X2,
                        )
                    # Hardware signals empty only after the queued scale copies/MMA finish.
                    prims.tcgen05_commit(mbar_empty.data_ptr(stage))
            if prims.elect_sync():
                prims.tcgen05_commit(mbar_done)  # Wake the epilogue after the last MMA.

        elif is_epi:
            while not prims.mbarrier_try_wait_parity(mbar_done, 0):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            # Warps 0..3 read TMEM rows 0..31, 32..63, 64..95, 96..127.
            # Each lane stores one output row, sixteen columns at a time.
            # This view groups C into 16-column chunks; it does not move any data.
            C_vectors = cute.zipped_divide(C, (1, 16))[(0, None), None]
            store_bf16x16 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16,
                num_bits_per_copy=256,  # 16 BF16 values x 16 bits = 32 bytes per copy.
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )
            row = tile_row + tid
            for col_start in cutlass.range_constexpr(0, self.BN, 16):
                tmem_addr = prims.TmemAddr.from_row_col(warp_id * 32, base + col_start)
                values = prims.tcgen05_ld("32x32b", tmem_addr.as_ptr(cutlass.Float32), num=16)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                # Convert all 16 FP32 accumulators to BF16 in registers.
                bf16_values = values.to(cutlass.BFloat16)
                tmp = cute.make_rmem_tensor(16, cutlass.BFloat16)
                tmp.store(cute.TensorSSA(bf16_values.ir_value(), (16,), cutlass.BFloat16))
                # Before: C[row, col] = cutlass.BFloat16(values[j]) stored one value at a time.
                # Benchmark sizes are multiples of 128, so each 16-value chunk fits fully.
                if row < M and tile_col + col_start + 16 <= N:
                    coord = (row, (tile_col + col_start) // 16)
                    cute.copy(store_bf16x16, tmp, C_vectors[None, coord])
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)

        # No block-wide barrier inside the separate warp loops: that would deadlock.
        prims.barrier_cta_sync(0)  # All epilogue reads must finish before freeing TMEM.
        if is_tc:
            prims.tcgen05_dealloc(prims.TmemAddr(base).as_ptr(cutlass.Float32), 256)


mxfp4 = MxFp4Gemm()
