def _tflops(m, n, k, time_us): return 2 * m * n * k / (time_us * 1_000_000)


def _benchmark_torch_matmul(a, b, out, warmup, iterations):
    import torch

    for _ in range(warmup):
        torch.matmul(a, b, out=out)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        torch.matmul(a, b, out=out)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1_000 / iterations


def benchmark_gemm(fn, a_cute, b_cute, c_cute, a_torch, b_torch, m, n, k, warmup=5, iterations=100):
    import torch
    from cutlass import cute

    cute_time_us = cute.testing.benchmark(fn, kernel_arguments=cute.testing.JitArguments(a_cute, b_cute, c_cute), warmup_iterations=warmup, iterations=iterations)
    torch_out = torch.empty((m, n), device=a_torch.device, dtype=a_torch.dtype)
    torch_time_us = _benchmark_torch_matmul(a_torch, b_torch, torch_out, warmup, iterations)
    cute_tflops = _tflops(m, n, k, cute_time_us)
    torch_tflops = _tflops(m, n, k, torch_time_us)
    percent_of_torch = 100 * torch_time_us / cute_time_us
    result = {"m": m, "n": n, "k": k, "cute_time_us": cute_time_us, "cute_tflops": cute_tflops, "torch_time_us": torch_time_us, "torch_tflops": torch_tflops, "percent_of_torch": percent_of_torch}

    print(f"{f'{m}x{n}x{k}':<15} | CuTe {cute_time_us:>10.2f} us {cute_tflops:>9.2f} TF/s | torch {torch_time_us:>10.2f} us {torch_tflops:>9.2f} TF/s | {percent_of_torch:>6.2f}%")
    return result
