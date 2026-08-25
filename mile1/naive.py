from benchmark.bench import app

from cutlass import cute


@cute.kernel
def add_kernel(gA, gB, gC):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    index = bidx * bdim + tidx
    m, n = gA.shape
    if index < m * n:
        row = index // n
        column = index % n
        gC[row, column] = gA[row, column] + gB[row, column]


@cute.jit
def add(mA, mB, mC):
    threads = 256
    elements = cute.size(mA)
    add_kernel(mA, mB, mC).launch(grid=(cute.ceil_div(elements, threads), 1, 1), block=(threads, 1, 1))
