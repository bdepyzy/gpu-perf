from benchmark.bench import app

import cutlass
from cutlass import cute
from cutlass.utils import SmemAllocator


BM, BN, BK = 128, 128, 32
TM, TN = 8, 8
TX, TY = 16, 16


@cute.kernel
def gemm_kernel(gA, gB, gC):
    tx, ty, _ = cute.arch.thread_idx()
    block_n, block_m, _ = cute.arch.block_idx()
    m0 = block_m * BM + ty * TM
    n0 = block_n * BN + tx * TN
    k_extent = gA.shape[1]

    accumulator = cute.make_rmem_tensor((TM, TN), cutlass.Float32)
    for mi in cutlass.range_constexpr(TM):
        for ni in cutlass.range_constexpr(TN):
            accumulator[(mi, ni)] = cutlass.Float32(0.0)

    smem_a_layout = cute.make_composed_layout(cute.make_swizzle(2, 3, 5), 0, cute.make_layout((BM, BK), stride=(BK, 1)))
    smem_b_layout = cute.make_layout((BK, BN), stride=(BN, 1))
    smem = SmemAllocator()
    smem_a = smem.allocate_tensor(gA.element_type, smem_a_layout, byte_alignment=16)
    smem_b = smem.allocate_tensor(gB.element_type, smem_b_layout, byte_alignment=16)

    thread = ty * TX + tx
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type, num_bits_per_copy=128)
    copy_a = cute.make_tiled_copy_tv(atom, cute.make_layout((64, 4), stride=(4, 1)), cute.make_layout((1, 8)))
    copy_b = cute.make_tiled_copy_tv(atom, cute.make_layout((16, 16), stride=(16, 1)), cute.make_layout((1, 8)))
    thread_a = copy_a.get_slice(thread)
    thread_b = copy_b.get_slice(thread)
    thread_smem_a = thread_a.partition_D(smem_a)
    thread_smem_b = thread_b.partition_D(smem_b)

    fragment_a = cute.make_rmem_tensor((TM, 1), gA.element_type)
    fragment_b = cute.make_rmem_tensor((1, TN), gB.element_type)

    for k0 in range(0, k_extent, BK):
        tile_a = cute.local_tile(gA, (BM, BK), (block_m, k0 // BK))
        tile_b = cute.local_tile(gB, (BK, BN), (k0 // BK, block_n))
        cute.copy(atom, thread_a.partition_S(tile_a), thread_smem_a)
        cute.copy(atom, thread_b.partition_S(tile_b), thread_smem_b)
        cute.arch.sync_threads()

        for k in cutlass.range_constexpr(BK):
            cute.autovec_copy(cute.local_tile(smem_a, (TM, 1), (ty, k)), fragment_a)
            cute.autovec_copy(cute.local_tile(smem_b, (1, TN), (k, tx)), fragment_b)
            for mi in cutlass.range_constexpr(TM):
                a = cutlass.Float32(fragment_a[(mi, 0)])
                for ni in cutlass.range_constexpr(TN):
                    accumulator[(mi, ni)] += a * cutlass.Float32(fragment_b[(0, ni)])
        cute.arch.sync_threads()

    for mi in cutlass.range_constexpr(TM):
        for ni in cutlass.range_constexpr(TN):
            gC[(m0 + mi, n0 + ni)] = gC.element_type(accumulator[(mi, ni)])


@cute.jit
def gemm(mA, mB, mC):
    m, n = mC.shape
    gemm_kernel(mA, mB, mC).launch(grid=(n // BN, m // BM, 1), block=(TX, TY, 1))
