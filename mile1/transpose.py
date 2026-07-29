import modal

from bench import benchmark


image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.11.0",
        "cutlass",
        "nvidia-cutlass",
        "nvidia-cutlass-dsl",
        "git+https://github.com/NTT123/cute-viz.git",
    )
    .add_local_python_source("bench")
)
app = modal.App("milestone1", image=image)


@app.function(gpu="RTX-PRO-6000")
def run():
    import torch
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack

    @cute.kernel
    def elementwise_add_kernel(gA, gB, gC, tv_layout):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # Keep the complete in-tile coordinate and select one block tile.
        block_coord = ((None, None), bidx)
        block_a = gA[block_coord]
        block_b = gB[block_coord]
        block_c = gC[block_coord]

        # Convert each block from (row, column) coordinates into
        # (thread, value) coordinates.
        thread_values_a = cute.composition(block_a, tv_layout)
        thread_values_b = cute.composition(block_b, tv_layout)
        thread_values_c = cute.composition(block_c, tv_layout)

        # Select this thread while keeping all values assigned to it.
        thread_coord = (tidx, None)
        thread_a = thread_values_a[thread_coord]
        thread_b = thread_values_b[thread_coord]
        thread_c = thread_values_c[thread_coord]

        # The data operation is still elementwise addition.
        thread_c[None] = thread_a.load() + thread_b.load()

    @cute.jit
    def launch_elementwise_add(mA, mB, mC):
        load_bytes = 16
        dtype = mA.element_type

        thread_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
        value_layout = cute.make_ordered_layout(
            (16, load_bytes),
            order=(1, 0),
        )
        value_layout = cute.recast_layout(dtype.width, 8, value_layout)
        tile_shape, tv_layout = cute.make_layout_tv(
            thread_layout,
            value_layout,
        )

        print(f"[INFO] Tile shape: {tile_shape}")
        print(f"[INFO] Thread/value layout: {tv_layout}")

        # ((coordinate inside tile), (which tile))
        tiled_a = cute.zipped_divide(mA, tile_shape)
        tiled_b = cute.zipped_divide(mB, tile_shape)
        tiled_c = cute.zipped_divide(mC, tile_shape)

        # Swap the two block-grid modes. This transposes the order in which
        # tiles are enumerated; it does not transpose the matrix contents.
        transposed_block_layout = cute.make_ordered_layout(
            cute.select(tiled_a.shape[1], mode=[1, 0]),
            order=(1, 0),
        )

        tiled_a = cute.composition(
            tiled_a,
            (None, transposed_block_layout),
        )
        tiled_b = cute.composition(
            tiled_b,
            (None, transposed_block_layout),
        )
        tiled_c = cute.composition(
            tiled_c,
            (None, transposed_block_layout),
        )

        print("[INFO] Tiled tensors after block-layout transpose:")
        print(f"[INFO]   A: {tiled_a.type}")
        print(f"[INFO]   B: {tiled_b.type}")
        print(f"[INFO]   C: {tiled_c.type}")

        elementwise_add_kernel(
            tiled_a,
            tiled_b,
            tiled_c,
            tv_layout,
        ).launch(
            grid=(cute.size(tiled_c, mode=[1]), 1, 1),
            block=(cute.size(tv_layout, mode=[0]), 1, 1),
        )

    m, n = 16384, 8192
    a = torch.randn(m, n, device="cuda", dtype=torch.float16)
    b = torch.randn(m, n, device="cuda", dtype=torch.float16)
    c = torch.empty(m, n, device="cuda", dtype=torch.float16)

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    elementwise_add = cute.compile(
        launch_elementwise_add,
        a_,
        b_,
        c_,
    )
    elementwise_add(a_, b_, c_)

    torch.testing.assert_close(c, a + b)
    benchmark(elementwise_add, a_, b_, c_)


# ============================================================================
# Literal matrix transpose (kept only for comparison)
# ============================================================================
# This is a different operation: it produces C = A.T with shape (N, M).
#
# @cute.kernel
# def transpose_kernel(gA, gC):
#     tidx, _, _ = cute.arch.thread_idx()
#     bidx, _, _ = cute.arch.block_idx()
#     bdim, _, _ = cute.arch.block_dim()
#
#     linear_idx = bidx * bdim + tidx
#     _, n = gA.shape
#
#     row = linear_idx // n
#     col = linear_idx % n
#
#     gC[(col, row)] = gA[(row, col)]
#
#
# @cute.jit
# def launch_transpose(mA, mC):
#     threads_per_block = 256
#     num_elements = cute.size(mA)
#     num_blocks = (
#         num_elements + threads_per_block - 1
#     ) // threads_per_block
#
#     transpose_kernel(mA, mC).launch(
#         grid=(num_blocks, 1, 1),
#         block=(threads_per_block, 1, 1),
#     )
#
#
# m, n = 16384, 8192
# a = torch.randn(m, n, device="cuda", dtype=torch.float16)
# c = torch.empty(n, m, device="cuda", dtype=torch.float16)
#
# a_ = from_dlpack(a, assumed_align=16)
# c_ = from_dlpack(c, assumed_align=16)
#
# transpose = cute.compile(launch_transpose, a_, c_)
# transpose(a_, c_)
#
# torch.testing.assert_close(c, a.T)
# benchmark(transpose, a_, c_)
