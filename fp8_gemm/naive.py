import cutlass
from cutlass import cute


@cute.kernel
def _gemm_kernel(gA, gB, gC):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx
    m_dim, n_dim = gC.shape
    k_dim = gA.shape[1]
    if idx < m_dim * n_dim:
        mi = idx // n_dim
        ni = idx % n_dim
        acc = cutlass.Float32(0.0)
        for ki in range(k_dim):
            acc += cutlass.Float32(gA[mi, ki]) * cutlass.Float32(gB[ni, ki])
        gC[mi, ni] = gC.element_type(acc)


@cute.jit
def gemm(mA, mB, mC):
    mA = cute.make_tensor(cute.recast_ptr(mA.iterator, dtype=cutlass.Float8E4M3FN), mA.layout)
    mB = cute.make_tensor(cute.recast_ptr(mB.iterator, dtype=cutlass.Float8E4M3FN), mB.layout)
    m_dim, n_dim = mC.shape
    _gemm_kernel(mA, mB, mC).launch(grid=(cute.ceil_div(m_dim * n_dim, 256), 1, 1), block=(256, 1, 1))
