import math


def _peak_bandwidth(gpu_name):
    if "RTX PRO 6000 Blackwell Server Edition" in gpu_name:
        return 1_597
    if "RTX PRO 6000" in gpu_name:
        return 1_792
    if "B200" in gpu_name:
        return 8_000
    return None


def _tensor_bytes(tensor):
    return math.prod(tensor.shape) * tensor.element_type.width // 8


def benchmark(fn, *args, bytes_moved=None, flops=0):
    import torch
    from cutlass import cute

    avg_time_us = cute.testing.benchmark(
        fn,
        kernel_arguments=cute.testing.JitArguments(*args),
        warmup_iterations=5,
        iterations=100,
    )

    if bytes_moved is None:
        bytes_moved = sum(_tensor_bytes(tensor) for tensor in args)

    result = {
        "time_us": avg_time_us,
        "gbps": bytes_moved / (avg_time_us * 1_000),
    }
    if flops:
        result["tflops"] = flops / (avg_time_us * 1_000_000)

    gpu_name = torch.cuda.get_device_name()
    peak_gbps = _peak_bandwidth(gpu_name)

    result["gpu"] = gpu_name
    if peak_gbps:
        result["peak_gbps"] = peak_gbps
        result["peak_bw_pct"] = 100 * result["gbps"] / peak_gbps

    print(f"Performance Metrics for {gpu_name}:")
    print("-------------------")
    print(f"Kernel execution time: {avg_time_us:.4f} us")
    print(f"Memory throughput: {result['gbps']:.2f} GB/s")
    if peak_gbps:
        print(f"Theoretical peak bandwidth: {peak_gbps:.0f} GB/s")
        print(f"Percent of theoretical peak: {result['peak_bw_pct']:.2f}%")
    return result
