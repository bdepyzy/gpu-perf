from benchmark.bench import app

from cutlass import cute


@cute.kernel
def add_kernel(gA, gB, gC):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_index = bidx * bdim + tidx
    tile_rows, tile_columns = gA.shape[1]
    if thread_index < tile_rows * tile_columns:
        column = thread_index % tile_columns
        row = thread_index // tile_columns
        gC[(None, (row, column))] = gA[(None, (row, column))].load() + gB[(None, (row, column))].load()


@cute.jit
def add(mA, mB, mC):
    threads = 256
    gA = cute.zipped_divide(mA, (1, 8))
    gB = cute.zipped_divide(mB, (1, 8))
    gC = cute.zipped_divide(mC, (1, 8))
    tiles = cute.size(gC, mode=[1])
    add_kernel(gA, gB, gC).launch(grid=(cute.ceil_div(tiles, threads), 1, 1), block=(threads, 1, 1))
