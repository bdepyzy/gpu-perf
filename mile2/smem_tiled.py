import modal

from bench import benchmark_gemm

image = modal.Image.debian_slim(python_version="3.13").uv_pip_install("torch==2.11.0", "cutlass", "nvidia-cutlass", "nvidia-cutlass-dsl").add_local_python_source("bench")
app = modal.App("milestone2", image=image)


@app.function(gpu="B200", timeout=1800)
def run():
    import cutlass
    import torch
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.utils import SmemAllocator

    # C rows and columns computed by one block
    BM = 128
    BN = 128
    # K values staged per shared-memory tile
    BK = 16
    # C rows and columns computed by one thread
    TM = 8
    TN = 8
    # CUDA threads along x and y in one block
    TX = 16
    TY = 16

    @cute.kernel
    def K(gA, gB, gC):
        # A[M,K] B[K,N]
        tx, ty, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        # Top-left C coordinate owned by this thread
        m0 = bidy * BM + ty * TM
        n0 = bidx * BN + tx * TN
        Kdim = gA.shape[1]

        # This thread's TM x TN FP32 accumulators
        acc = cute.make_rmem_tensor((TM, TN), cutlass.Float32)

        for mi in cutlass.range_constexpr(TM):
            for ni in cutlass.range_constexpr(TN):
                acc[(mi, ni)] = cutlass.Float32(0.0)


        # Current shared-memory tile layout
        sA_layout = cute.make_layout((BM, BK), stride=(BK, 1))
        sB_layout = cute.make_layout((BK, BN), stride=(BN, 1))
        smem = SmemAllocator()

        sA = smem.allocate_tensor(gA.element_type, sA_layout, byte_alignment=16)
        sB = smem.allocate_tensor(gB.element_type, sB_layout, byte_alignment=16)

        for k0 in range(0, Kdim, BK):
            tid = TX * ty + tx
            threads = TY * TX

            for load in cutlass.range_constexpr((BM * BK) // threads):
                s_idx = tid + load * threads
                s_row = s_idx // BK
                s_col = s_idx % BK
                sA[(s_row, s_col)] = gA[(bidy * BM + s_row, k0 + s_col)]

            for load in cutlass.range_constexpr((BK * BN) // threads):
                s_idx = tid + load * threads
                s_row = s_idx // BN
                s_col = s_idx % BN
                sB[(s_row, s_col)] = gB[(k0 + s_row, bidx * BN + s_col)]
            cute.arch.sync_threads()

            for k in cutlass.range_constexpr(BK):
                for mi in cutlass.range_constexpr(TM):

                    a = cutlass.Float32(sA[(ty * TM + mi, k)])

                    for ni in cutlass.range_constexpr(TN):

                        b = cutlass.Float32(sB[(k, tx * TN + ni)])

                        acc[(mi, ni)] = acc[(mi, ni)] + a * b
            cute.arch.sync_threads()

        for mi in cutlass.range_constexpr(TM):
            for ni in cutlass.range_constexpr(TN):
                gC[(m0 + mi, n0 + ni)] = gC.element_type(acc[(mi, ni)])

    @cute.jit
    def L(mA, mB, mC):
        M, N = mC.shape
        blocks_n = N // BN
        blocks_m = M // BM
        K(mA, mB, mC).launch(grid=(blocks_n, blocks_m, 1), block=(TX, TY, 1))

    shapes = [256, 512, 768, 1024, 2048, 4096, 8192]
    warmup=5
    iterations=100

    print(f"Shapes={len(shapes)} | warmup={warmup} | iterations={iterations} | C[M,N]=A[M,K]@B[K,N]")

    torch.manual_seed(0)
    results = []

    for size in shapes:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        c = torch.empty(size, size, device="cuda", dtype=torch.float16)
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
            print(f"CORRECTNESS FAILED {size}x{size}x{size} | max_abs={max_abs_error:.6f} | mean_abs={mean_abs_error:.6f}")
            raise

        result = benchmark_gemm(gemm, a_cute, b_cute, c_cute, a, b, size, size, size, warmup=warmup, iterations=iterations)
        result["max_abs_error"] = max_abs_error
        result["mean_abs_error"] = mean_abs_error
        results.append(result)
        del a, b, c, reference
        torch.cuda.empty_cache()

    return results
