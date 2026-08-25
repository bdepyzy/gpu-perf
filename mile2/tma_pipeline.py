from typing import Union

from benchmark.bench import app

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.experimental.cuda as cuda
from cutlass.experimental import primitives as prims


class Kernel:
    def __init__(self):
        self.tile_shape = (128, 128)
        self.tile_m, self.tile_n = self.tile_shape
        self.tile_k = 32
        self.thread_tile_shape = (8, 8)
        self.thread_tile_m, self.thread_tile_n = self.thread_tile_shape
        self.threads_x, self.threads_y = 16, 16
        self.cluster_shape_mn = (1, 1)
        self.threads_per_cta = self.threads_x * self.threads_y
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(self, srcA, srcB, dstC):
        # These are the two trays used by one K iteration.
        sA_layout = cute.make_layout((self.tile_m, self.tile_k), stride=(self.tile_k, 1))
        sB_layout = cute.make_layout((self.tile_k, self.tile_n), stride=(self.tile_n, 1))

        tma_desc_A = cuda.create_tensor_map_tiled_from_view(srcA, box_dims=(self.tile_m, self.tile_k))
        tma_desc_B = cuda.create_tensor_map_tiled_from_view(srcB, box_dims=(self.tile_k, self.tile_n))

        blocks_m = dstC.shape[0] // self.tile_m
        blocks_n = dstC.shape[1] // self.tile_n
        self.kernel(srcA, srcB, dstC, sA_layout, sB_layout, tma_desc_A, tma_desc_B).launch(grid=(blocks_n, blocks_m, 1), block=(self.threads_x, self.threads_y, 1), cluster=(*self.cluster_shape_mn, 1))

    @cute.kernel
    def kernel(
        self,
        gA,
        gB,
        gC,
        sA_layout: Union[cute.Layout, cute.ComposedLayout],
        sB_layout: Union[cute.Layout, cute.ComposedLayout],
        tma_desc_A: cutlass.GridConstant[cuda.TensorMap],
        tma_desc_B: cutlass.GridConstant[cuda.TensorMap],
    ):
        # A[M,K] B[K,N]
        tx, ty, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        # Top-left C coordinate owned by this thread
        m0 = bidy * self.tile_m + ty * self.thread_tile_m
        n0 = bidx * self.tile_n + tx * self.thread_tile_n
        Kdim = gA.shape[1]

        # This thread's output tile uses FP32 accumulators.
        acc = cute.make_rmem_tensor(self.thread_tile_shape, cutlass.Float32)

        for mi in cutlass.range_constexpr(self.thread_tile_m):
            for ni in cutlass.range_constexpr(self.thread_tile_n):
                acc[(mi, ni)] = cutlass.Float32(0.0)

        smem = utils.SmemAllocator()
        sA = smem.allocate_tensor(gA.element_type, sA_layout, byte_alignment=self.buffer_align_bytes)
        sB = smem.allocate_tensor(gB.element_type, sB_layout, byte_alignment=self.buffer_align_bytes)

        mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            if prims.elect_sync():
                prims.prefetch_tensormap(tma_desc_A.get_ptr())
                prims.prefetch_tensormap(tma_desc_B.get_ptr())
                prims.mbarrier_init(mbar, 1)

        prims.fence_mbarrier_init()
        prims.barrier_cta_sync(0)

        # Per-k register fragments of the original micro-tile.
        a_frag = cute.make_rmem_tensor((self.thread_tile_m, 1), gA.element_type)
        b_frag = cute.make_rmem_tensor((1, self.thread_tile_n), gB.element_type)

        phase = cutlass.Int32(0)
        for k0 in range(0, Kdim, self.tile_k):
            if warp_idx == 0:
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(mbar, tma_desc_A.global_tx_bytes() + tma_desc_B.global_tx_bytes())
                if prims.elect_sync():
                    prims.cp_async_bulk_tensor_shared_cta_global(sA.iterator, tma_desc_A.get_ptr(), (k0, bidy * self.tile_m), mbar)
                if prims.elect_sync():
                    prims.cp_async_bulk_tensor_shared_cta_global(sB.iterator, tma_desc_B.get_ptr(), (bidx * self.tile_n, k0), mbar)

            while not prims.mbarrier_try_wait_parity(mbar, phase):
                pass

            for k in cutlass.range_constexpr(self.tile_k):
                # cute.local_tile(tensor, (tile_height, tile_width), (tile_row, tile_col)) ==
                # tensor[tile_row * tile_height : (tile_row + 1) * tile_height,
                #       tile_col * tile_width  : (tile_col + 1) * tile_width,
                # ]
                cute.autovec_copy(cute.local_tile(sA, (self.thread_tile_m, 1), (ty, k)), a_frag)
                cute.autovec_copy(cute.local_tile(sB, (1, self.thread_tile_n), (k, tx)), b_frag)

                for mi in cutlass.range_constexpr(self.thread_tile_m):
                    a = cutlass.Float32(a_frag[(mi, 0)])
                    for ni in cutlass.range_constexpr(self.thread_tile_n):
                        acc[(mi, ni)] = acc[(mi, ni)] + a * cutlass.Float32(b_frag[(0, ni)])
            prims.barrier_cta_sync(0)
            phase = phase ^ 1

        for mi in cutlass.range_constexpr(self.thread_tile_m):
            for ni in cutlass.range_constexpr(self.thread_tile_n):
                gC[(m0 + mi, n0 + ni)] = gC.element_type(acc[(mi, ni)])
