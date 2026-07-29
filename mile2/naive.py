import modal

from bench import benchmark_gemm

image = modal.Image.debian_slim(python_version="3.13").uv_pip_install("torch==2.11.0", "cutlass", "nvidia-cutlass", "nvidia-cutlass-dsl").add_local_python_source("bench")
app = modal.App("milestone2", image=image)


@app.function(gpu="RTX-PRO-6000", timeout=30 * 60)
def run(suite="all"):
    import cutlass
    import torch
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.utils import SmemAllocator

    @cute.kernel
    def K(gA, gB, gC):
        # A[M,K] B[K,N]
        tx, ty, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        T = 16
        m = bidy * T + ty
        n = bidx * T + tx
        Kdim = gA.shape[1]

        smem_layout = cute.make_layout((T, T), stride=(T, 1))
        smem = SmemAllocator()

        smem_A = smem.allocate_tensor(gA.element_type, smem_layout, byte_alignment=16)
        smem_B = smem.allocate_tensor(gB.element_type, smem_layout, byte_alignment=16)

        acc = cutlass.Float32(0.0)
        for k0 in range(0, Kdim, T):
            a_col = k0 + tx
            b_row = k0 + ty
            smem_A[(ty, tx)] = gA[(m, a_col)]
            smem_B[(ty, tx)] = gB[(b_row, n)]
            cute.arch.sync_threads()
            for k in cutlass.range_constexpr(T):
                a = cutlass.Float32(smem_A[(ty, k)])
                b = cutlass.Float32(smem_B[(k, tx)])
                acc += a * b
            cute.arch.sync_threads()
        gC[(m, n)] = gC.element_type(acc)

    @cute.jit
    def L(mA, mB, mC):
        T = 16
        M, N = mC.shape
        blocks_n = N // T
        blocks_m = M // T
        K(mA, mB, mC).launch(grid=(blocks_n, blocks_m, 1), block=(T, T, 1))

    square_shapes = [(4096, 4096, 4096)]
    warmup=5
    iterations=100

    if suite == "smoke": shapes = [(128, 128, 128)]
    elif suite == "all": shapes = square_shapes
    else: raise ValueError("suite must be one of: smoke, all")

    print(f"Suite={suite} | shapes={len(shapes)} | warmup={warmup} | iterations={iterations} | C[M,N]=A[M,K]@B[K,N]")

    torch.manual_seed(0)
    results = []

    for problem_m, problem_n, problem_k in shapes:
        if problem_m != problem_n or problem_n != problem_k or problem_m % 16 != 0: raise ValueError("This kernel requires square dimensions divisible by 16")
        a = torch.randn(problem_m, problem_k, device="cuda", dtype=torch.float16)
        b = torch.randn(problem_k, problem_n, device="cuda", dtype=torch.float16)
        c = torch.empty(problem_m, problem_n, device="cuda", dtype=torch.float16)
        a_cute = from_dlpack(a, assumed_align=16)
        b_cute = from_dlpack(b, assumed_align=16)
        c_cute = from_dlpack(c, assumed_align=16)
        gemm = cute.compile(L, a_cute, b_cute, c_cute)
        gemm(a_cute, b_cute, c_cute)

        reference = torch.matmul(a, b)
        abs_error = (c - reference).abs()
        max_abs_error = abs_error.max().item()
        mean_abs_error = abs_error.float().mean().item()
        try:
            torch.testing.assert_close(c, reference, rtol=1e-2, atol=5e-1)
        except AssertionError:
            print(f"CORRECTNESS FAILED {problem_m}x{problem_n}x{problem_k} | max_abs={max_abs_error:.6f} | mean_abs={mean_abs_error:.6f}")
            raise

        result = benchmark_gemm(gemm, a_cute, b_cute, c_cute, a, b, problem_m, problem_n, problem_k, warmup=warmup, iterations=iterations)
        result["max_abs_error"] = max_abs_error
        result["mean_abs_error"] = mean_abs_error
        results.append(result)
        del a, b, c, reference
        torch.cuda.empty_cache()

    return results
