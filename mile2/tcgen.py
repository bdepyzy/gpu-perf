from benchmark.bench import app

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.experimental import primitives as prims


class Kernel:
    def __init__(self):
        self.tile_shape = (128, 128)
        self.tile_m, self.tile_n = self.tile_shape
        # SWIZZLE_128B requires 128 contiguous operand bytes: 64 fp16 values.
        self.tile_k = 64
        self.threads_per_cta = 256

    @cute.jit
    def __call__(self, srcA, srcB, dstC):
        tma_desc_A = cuda.create_tensor_map_tiled_from_view(srcA, box_dims=(self.tile_m, self.tile_k), swizzle=cuda.TensorMapSwizzle.s128b)
        tma_desc_B = cuda.create_tensor_map_tiled_from_view(srcB, box_dims=(self.tile_k, self.tile_n), swizzle=cuda.TensorMapSwizzle.s128b)

        blocks_m = dstC.shape[0] // self.tile_m
        blocks_n = dstC.shape[1] // self.tile_n
        self.kernel(srcA, srcB, dstC, tma_desc_A, tma_desc_B).launch(grid=(blocks_n, blocks_m, 1), block=(self.threads_per_cta, 1, 1))

    @cute.kernel
    def kernel(self, gA, gB, gC, tma_desc_A: cutlass.GridConstant[cuda.TensorMap], tma_desc_B: cutlass.GridConstant[cuda.TensorMap]):
        tx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        Kdim = gA.shape[1]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        addr_smem = cutlass.AddressSpace.smem
        tmem_num_col = self.tile_n

        # tcgen05 consumes K-major tiles. B is therefore stored as (N, K),
        # matching the working NVIDIA single-tile example.
        sA = cutlass.Array(gA.element_type, (self.tile_m, self.tile_k), space=addr_smem)
        sB = cutlass.Array(gB.element_type, (self.tile_n, self.tile_k), space=addr_smem)
        # One barrier pair per possible K tile avoids phase reuse while this
        # first tcgen05 implementation is intentionally single-buffered.
        max_k_tiles = 4
        mbar_tma = cutlass.Array(cutlass.Int64, max_k_tiles, space=addr_smem)
        mbar_mma = cutlass.Array(cutlass.Int64, max_k_tiles, space=addr_smem)
        tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=addr_smem)

        if prims.elect_sync():
            prims.prefetch_tensormap(tma_desc_A.get_ptr())
            prims.prefetch_tensormap(tma_desc_B.get_ptr())
            for stage in cutlass.range_constexpr(max_k_tiles):
                prims.mbarrier_init(mbar_tma.data_ptr(stage), 1)
                prims.mbarrier_init(mbar_mma.data_ptr(stage), 1)

        prims.fence_mbarrier_init()
        prims.barrier_cta_sync(0)

        is_tma = warp_idx == 0
        is_tc = warp_idx == 1
        is_epi = warp_idx >= 2

        # Warps 2 and 3 are neither producers nor part of the epilogue
        # warpgroup. Exited threads are excluded from the later CTA barrier.
        if warp_idx == 2 or warp_idx == 3:
            prims.exit()

        if is_tc:
            prims.tcgen05_alloc(tmem_ptr_i32, tmem_num_col)
            prims.tcgen05_relinquish_alloc_permit()

        prims.barrier_cta_sync(0)

        tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int32)
        desc_A_root = prims.Tcgen05SmemDesc.build(sA, leading_byte_offset=16, stride_byte_offset=1024, layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B)
        desc_B_root = prims.Tcgen05SmemDesc.build(sB, leading_byte_offset=16, stride_byte_offset=1024, layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B)
        idesc = prims.Tcgen05InstrDesc.build(c_dtype=cutlass.Float32, n_dim=self.tile_n, m_dim=self.tile_m)

        for k0 in range(0, Kdim, self.tile_k):
            stage = k0 // self.tile_k
            if is_tma:
                if prims.elect_sync():
                    transaction_bytes = tma_desc_A.global_tx_bytes() + tma_desc_B.global_tx_bytes()
                    prims.mbarrier_arrive_expect_tx(mbar_tma.data_ptr(stage), transaction_bytes)
                    prims.cp_async_bulk_tensor_shared_cta_global(sA, tma_desc_A.get_ptr(), (k0, bidy * self.tile_m), mbar_tma.data_ptr(stage))
                    prims.cp_async_bulk_tensor_shared_cta_global(sB, tma_desc_B.get_ptr(), (bidx * self.tile_n, k0), mbar_tma.data_ptr(stage))

            while not prims.mbarrier_try_wait_parity(mbar_tma.data_ptr(stage), 0, time_limit=10000000):
                pass

            if is_tc:
                if prims.elect_sync():
                    for i in cutlass.range_constexpr(4):
                        desc_offset_bytes: cutlass.Constexpr[int] = 16 * 2 * i
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_1,
                            tmem_ptr,
                            desc_A_root.advance_start_address(desc_offset_bytes),
                            desc_B_root.advance_start_address(desc_offset_bytes),
                            idesc,
                            k0 != 0 or i != 0,
                        )
                    # Waiting for this completion before the next iteration
                    # makes the single SMEM stage safe to overwrite.
                    prims.tcgen05_commit(mbar_mma.data_ptr(stage))

            while not prims.mbarrier_try_wait_parity(mbar_mma.data_ptr(stage), 0, time_limit=10000000):
                pass

        if is_epi:
            epilogue_warp = warp_idx % 4
            tmem_base = prims.TmemAddr(tmem_ptr_i32.load())
            row_id = tmem_base.row_id + epilogue_warp * 32

            for n in range(0, self.tile_n, 2):
                tmem_addr = prims.TmemAddr.from_row_col(row_id, tmem_base.col_id + n).as_ptr(cutlass.Float32)
                c_rmem = prims.tcgen05_ld("32x32b", tmem_addr, num=2)
                m = bidy * self.tile_m + tx % self.tile_m
                for i in cutlass.range_constexpr(2):
                    gC[m, bidx * self.tile_n + n + i] = cutlass.Float16(c_rmem[i])

        prims.barrier_cta_sync(0)
        if is_tc:
            prims.tcgen05_dealloc(tmem_ptr, tmem_num_col)
