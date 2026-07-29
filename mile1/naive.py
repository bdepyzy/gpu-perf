import modal

from bench import benchmark

image = (modal.Image.debian_slim(python_version="3.13").uv_pip_install("torch==2.11.0", "cutlass", "nvidia-cutlass", "nvidia-cutlass-dsl").add_local_python_source("bench"))
app = modal.App("milestone1", image=image)


@app.function(gpu="RTX-PRO-6000")
def run():
    import torch
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack

    @cute.kernel
    def naive_kernel(gA, gB, gC):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()

        thread_idx = bidx * bdim + tidx
        _, n = gA.shape

        ni = thread_idx % n
        mi = thread_idx // n

        a_val = gA[mi, ni]
        b_val = gB[mi, ni]

        gC[mi, ni] = a_val + b_val


    @cute.jit
    def naive(mA, mB, mC):
        threads = 256

        m, n = mA.shape

        kernel = naive_kernel(mA, mB, mC)
        kernel.launch(
            grid=((m * n) // threads, 1, 1),
            block=(threads, 1, 1),
        )

    M, N = 16384, 8192  # Using large matrices to measure performance
    a = torch.randn(M, N, device="cuda", dtype=torch.float16)  # Random input A
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)  # Random input B
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)  # Output buffer

    a_ = from_dlpack(a, assumed_align=16)  # CuTe tensor A
    b_ = from_dlpack(b, assumed_align=16)  # CuTe tensor B
    c_ = from_dlpack(c, assumed_align=16)  # CuTe tensor C

    out_ = cute.compile(naive, a_, b_, c_)
    out_(a_, b_, c_)
    torch.testing.assert_close(c, a + b)

    benchmark(out_, a_, b_, c_)
