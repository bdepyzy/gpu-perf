import math
import statistics

import torch
from cutlass.cute.runtime import from_dlpack

from benchmark.common import (
    B200_HBM_GBPS,
    B200_PEAK_TFLOPS,
    compare,
    format_shape,
    make_models,
    print_table,
    problem_modules,
    to_cuda,
)

DEPS = ()
PROBLEM = "fp8_gemm"
TOLERANCE = 0.15

SHAPES = [
    {"M": 1, "N": 4096, "K": 4096},
    {"M": 8, "N": 4096, "K": 4096},
    {"M": 64, "N": 4096, "K": 4096},
    {"M": 256, "N": 4096, "K": 4096},
    {"M": 1024, "N": 4096, "K": 4096},
    {"M": 4096, "N": 4096, "K": 4096},
    {"M": 4096, "N": 11008, "K": 4096},
    {"M": 4096, "N": 4096, "K": 11008},
]

L2_FLUSH_FLOATS = 256 * 1024 * 1024 // 4
WARMUP_ITERS = 10
TIMED_ITERS = 50
PAD = 16


def _solution_fn(model, inputs):
    model(*inputs)
    m = inputs[0].shape[0]
    c = torch.empty(m, model.N, device="cuda", dtype=torch.bfloat16)
    a_ = from_dlpack(inputs[0].view(torch.uint8), assumed_align=16)
    b_ = from_dlpack(model.weight.detach(), assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)
    compiled = model._compiled
    return lambda: compiled(a_, b_, c_)


def _cublaslt_fn(x, weight):
    k = x.shape[1]
    amax = weight.float().abs().amax().clamp(min=1e-12)
    scale = amax / 448.0
    w8 = (weight.float() / scale).to(torch.float8_e4m3fn)
    padded = (k + PAD - 1) // PAD * PAD
    if padded != k:
        xp = torch.zeros(x.shape[0], padded, device=x.device, dtype=x.dtype)
        xp[:, :k] = x
        wp = torch.zeros(w8.shape[0], padded, device=w8.device, dtype=w8.dtype)
        wp[:, :k] = w8
        x, w8 = xp, wp
    scale_a = torch.tensor(1.0, device=x.device)
    scale_b = scale.to(x.device)
    return lambda: torch._scaled_mm(x, w8.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)


def _rel_error(ref, actual):
    return ((ref.float() - actual.float()).norm() / ref.float().norm().clamp(min=1e-6)).item()


def _bench(fn, l2_flush, warmup=WARMUP_ITERS, iters=TIMED_ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        l2_flush.zero_()
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def evaluate(workload=None):
    source_dir, reference, solution = problem_modules(PROBLEM)
    peak_tflops = B200_PEAK_TFLOPS["fp8"]
    l2_flush = torch.empty(L2_FLUSH_FLOATS, dtype=torch.float32, device="cuda")
    rows = []
    speedups = []

    for shape in SHAPES:
        reference_model, solution_model = make_models(PROBLEM, reference, solution, shape)
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)
        inputs = to_cuda(reference.get_inputs())
        with torch.no_grad():
            ref_out = reference_model(*inputs)
            sol_out = solution_model(*inputs)
        ok, message = compare(ref_out, sol_out, TOLERANCE)
        if not ok:
            raise RuntimeError(f"solution correctness failed | {format_shape(shape)} | {message}")

        opponent_fn = None
        opp_us = None
        try:
            candidate = _cublaslt_fn(inputs[0], solution_model.weight.detach())
            with torch.no_grad():
                opp_out = candidate()
            rel = _rel_error(ref_out, opp_out)
            if math.isfinite(rel) and rel < 0.05:
                opponent_fn = candidate
            else:
                print(f"cuBLASLt FP8 skipped | {format_shape(shape)} | rel_fro={rel:.4f}")
        except RuntimeError as error:
            print(f"cuBLASLt FP8 skipped | {format_shape(shape)} | {error}")

        m, n, k = shape["M"], shape["N"], shape["K"]
        flops = 2 * m * n * k
        moved_bytes = m * k + k * n + m * n * 2
        compute_us = flops / (peak_tflops * 1_000_000)
        memory_us = moved_bytes / (B200_HBM_GBPS * 1_000)

        sol_us = _bench(_solution_fn(solution_model, inputs), l2_flush)
        if opponent_fn is not None:
            opp_us = _bench(opponent_fn, l2_flush)

        tflops = flops / sol_us / 1e6
        pct_peak = 100 * tflops / peak_tflops
        gbps = moved_bytes / sol_us / 1e3
        bound = "tc" if compute_us > memory_us else "mem"
        speedup = sol_us / opp_us if opp_us else float("nan")
        speedups.append(speedup)
        rows.append(
            (
                format_shape(shape),
                f"{sol_us:.2f} us",
                f"{tflops:,.1f}",
                f"{pct_peak:.2f}%",
                f"{gbps:,.0f}",
                bound,
                f"{opp_us:.2f} us" if opp_us else "n/a",
                f"{speedup:.2f}x" if opp_us else "-",
            )
        )

    geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    title = f"{torch.cuda.get_device_name()}  |  {PROBLEM}  |  fp8e4m3 x fp8e4m3 -> bf16"
    subtitle = f"L2 flushed per iter, median of {TIMED_ITERS}, warmup {WARMUP_ITERS}"
    headers = ("Shape", "Kernel", "TFLOPS", "%peak", "GB/s", "Bound", "cuBLASLt FP8", "Speedup")
    footer = ("Geomean", "", "", "", "", "", "", f"{geo:.2f}x")
    print_table(title, subtitle, headers, rows, footer)
