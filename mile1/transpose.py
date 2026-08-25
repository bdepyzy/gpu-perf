from benchmark.bench import app

from cutlass import cute


@cute.kernel
def add_kernel(gA, gB, gC, tv_layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    block_coord = ((None, None), bidx)
    thread_coord = (tidx, None)
    thread_a = cute.composition(gA[block_coord], tv_layout)[thread_coord]
    thread_b = cute.composition(gB[block_coord], tv_layout)[thread_coord]
    thread_c = cute.composition(gC[block_coord], tv_layout)[thread_coord]
    thread_c[None] = thread_a.load() + thread_b.load()


@cute.jit
def add(mA, mB, mC):
    load_bytes = 16
    thread_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    value_layout = cute.make_ordered_layout((16, load_bytes), order=(1, 0))
    value_layout = cute.recast_layout(mA.element_type.width, 8, value_layout)
    tile_shape, tv_layout = cute.make_layout_tv(thread_layout, value_layout)
    tiled_a = cute.zipped_divide(mA, tile_shape)
    tiled_b = cute.zipped_divide(mB, tile_shape)
    tiled_c = cute.zipped_divide(mC, tile_shape)
    block_layout = cute.make_ordered_layout(cute.select(tiled_a.shape[1], mode=[1, 0]), order=(1, 0))
    tiled_a = cute.composition(tiled_a, (None, block_layout))
    tiled_b = cute.composition(tiled_b, (None, block_layout))
    tiled_c = cute.composition(tiled_c, (None, block_layout))
    add_kernel(tiled_a, tiled_b, tiled_c, tv_layout).launch(grid=(cute.size(tiled_c, mode=[1]), 1, 1), block=(cute.size(tv_layout, mode=[0]), 1, 1))
