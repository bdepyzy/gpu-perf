from benchmark.bench import app

import cutlass
from cutlass import cute
from cutlass.utils import SmemAllocator


TILE = 16


@cute.kernel
def gemm_kernel(gA, gB, gC):
    tx, ty, _ = cute.arch.thread_idx()
    block_n, block_m, _ = cute.arch.block_idx()
    row = block_m * TILE + ty
    column = block_n * TILE + tx
    k_extent = gA.shape[1]

    smem_layout = cute.make_layout((TILE, TILE), stride=(TILE, 1))
    smem = SmemAllocator()
    smem_a = smem.allocate_tensor(gA.element_type, smem_layout, byte_alignment=16)
    smem_b = smem.allocate_tensor(gB.element_type, smem_layout, byte_alignment=16)

    accumulator = cutlass.Float32(0.0)
    for k0 in range(0, k_extent, TILE):
        smem_a[(ty, tx)] = gA[(row, k0 + tx)]
        smem_b[(ty, tx)] = gB[(k0 + ty, column)]
        cute.arch.sync_threads()
        for k in cutlass.range_constexpr(TILE):
            accumulator += cutlass.Float32(smem_a[(ty, k)]) * cutlass.Float32(smem_b[(k, tx)])
        cute.arch.sync_threads()
    gC[(row, column)] = gC.element_type(accumulator)


@cute.jit
def gemm(mA, mB, mC):
    m, n = mC.shape
    gemm_kernel(mA, mB, mC).launch(grid=(n // TILE, m // TILE, 1), block=(TILE, TILE, 1))
