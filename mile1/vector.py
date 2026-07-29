from pathlib import Path

import modal

from bench import benchmark

REMOTE_SVG_PATH = "/tmp/tv_layout.svg"

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
    from cute_viz import render_tv_layout_svg
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack

    @cute.jit
    def visualize():
        # Create and render a layout to file
        # layout = cute.make_layout( ((16,16),(256,2)), stride=((512,8192),(1,256)))
        # display_layout(layout)

        tv_layout = cute.make_layout(((32, 4), (8, 4)), stride=((128, 4), (16, 1)))
        render_tv_layout_svg(tv_layout, (16, 256), REMOTE_SVG_PATH)

        thr_block_layout = cute.make_layout((16, 256), stride=(512, 1))
        print(cute.composition(thr_block_layout, tv_layout))

    @cute.kernel
    def K(gA, gB, gC):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()

        thread_idx = bidx * bdim + tidx

        _, n = gA.shape[1]
        ni = thread_idx % n
        mi = thread_idx // n

        a_val = gA[(None, (mi, ni))].load()
        b_val = gB[(None, (mi, ni))].load()
        #print(f"sliced gA {a_val}")
        #print(f"sliced gB {b_val}")

        gC[(None, (mi, ni))] = a_val + b_val


    @cute.jit
    def L(mA, mB, mC):
        threads_per_block = 256

        gA = cute.zipped_divide(mA, (1, 8))
        gB = cute.zipped_divide(mB, (1, 8))
        gC = cute.zipped_divide(mC, (1, 8))

        print("[DSL INFO] Tiled Tensors:")
        print(f"[DSL INFO]   gA = {gA}")
        print(f"[DSL INFO]   gB = {gB}")
        print(f"[DSL INFO]   gC = {gC}")

        K(gA, gB, gC).launch(
            grid=(cute.size(gC, mode=[1]) // threads_per_block, 1, 1),
            block=(threads_per_block, 1, 1),
        )

    M, N = 16384, 8192  # Using large matrices to measure performance
    a = torch.randn(M, N, device="cuda", dtype=torch.float16)  # Random input A
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)  # Random input B
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)  # Output buffer

    a_ = from_dlpack(a, assumed_align=16)  # CuTe tensor A
    b_ = from_dlpack(b, assumed_align=16)  # CuTe tensor B
    c_ = from_dlpack(c, assumed_align=16)  # CuTe tensor C

    out_ = cute.compile(L, a_, b_, c_)
    out_(a_, b_, c_)

    torch.testing.assert_close(c, a + b)
    #visualize()

    benchmark(out_, a_, b_, c_)
