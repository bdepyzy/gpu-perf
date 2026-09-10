import math
import statistics

import torch
import torch.nn as nn
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from benchmark import common
from benchmark.common import (
    B200_HBM_GBPS,
    B200_PEAK_TFLOPS,
    GEMM_SHAPES,
    compare,
    format_ratio,
    format_shape,
    print_table,
    problem_modules,
    to_cuda,
)

DEPS = ()
PROBLEM = "fp8_gemm"
TOLERANCE = 0.05

SHAPES = GEMM_SHAPES

L2_FLUSH_FLOATS = 256 * 1024 * 1024 // 4
WARMUP_ITERS = 10
TIMED_ITERS = 50
PAD = 16


class SolutionAdapter(nn.Module):
    """Benchmark transport for a solution exporting gemm(mA, mB, mC)."""

    def __init__(self, solution, M: int, N: int, K: int):
        super().__init__()
        self.M, self.N, self.K = M, N, K
        self.weight = nn.Parameter(torch.empty(N, K, dtype=torch.float8_e4m3fn))
        self.entrypoint = solution.gemm
        self._compiled = None

    def _callables(self, x):
        output = torch.empty(self.M, self.N, device=x.device, dtype=torch.bfloat16)
        args = (
            from_dlpack(x.view(torch.uint8), assumed_align=16),
            from_dlpack(self.weight.detach().view(torch.uint8), assumed_align=16),
            from_dlpack(output, assumed_align=16),
        )
        if self._compiled is None:
            self._compiled = cute.compile(self.entrypoint, *args)
        return self._compiled, args, output

    def forward(self, x):
        compiled, args, output = self._callables(x)
        compiled(*args)
        return output

    def prepare_for_bench(self, inputs):
        compiled, args, _ = self._callables(inputs[0])
        return lambda: compiled(*args)


def _make_models(reference, solution, shape):
    common.apply_shape(PROBLEM, reference, shape)
    init_args = reference.get_init_inputs()
    reference_model = reference.Model(*init_args).to("cuda").eval()
    solution_model = SolutionAdapter(solution, *init_args).to("cuda").eval()
    solution_model.load_state_dict(reference_model.state_dict(), strict=True)
    return reference_model, solution_model


def _solution_fn(model, inputs):
    return model.prepare_for_bench(inputs)


def _cublaslt_fn(x, weight):
    k = x.shape[1]
    w8 = weight
    padded = (k + PAD - 1) // PAD * PAD
    if padded != k:
        xp = torch.zeros(x.shape[0], padded, device=x.device, dtype=x.dtype)
        xp[:, :k] = x
        wp = torch.zeros(w8.shape[0], padded, device=w8.device, dtype=w8.dtype)
        wp[:, :k] = w8
        x, w8 = xp, wp
    scale_a = torch.tensor(1.0, device=x.device)
    scale_b = torch.tensor(1.0, device=x.device)
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


def evaluate(workload=None, shape=None):
    source_dir, reference, solution = problem_modules(PROBLEM)
    peak_tflops = B200_PEAK_TFLOPS["fp8"]
    l2_flush = torch.empty(L2_FLUSH_FLOATS, dtype=torch.float32, device="cuda")
    rows = []
    speedups = []
    shapes = SHAPES
    if shape is not None:
        shapes = [candidate for candidate in SHAPES if candidate == {"M": shape, "N": shape, "K": shape}]
        if not shapes:
            available = ", ".join(str(candidate["M"]) for candidate in SHAPES)
            raise ValueError(f"unknown square shape {shape}; available sizes: {available}")

    for problem_shape in shapes:
        reference_model, solution_model = _make_models(reference, solution, problem_shape)
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)
        inputs = to_cuda(reference.get_inputs())
        with torch.no_grad():
            ref_out = reference_model(*inputs)
            sol_out = solution_model(*inputs)
        ok, message = compare(ref_out, sol_out, TOLERANCE)
        if not ok:
            raise RuntimeError(f"solution correctness failed | {format_shape(problem_shape)} | {message}")

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
                print(f"cuBLASLt FP8 skipped | {format_shape(problem_shape)} | rel_fro={rel:.4f}")
        except RuntimeError as error:
            print(f"cuBLASLt FP8 skipped | {format_shape(problem_shape)} | {error}")

        m, n, k = problem_shape["M"], problem_shape["N"], problem_shape["K"]
        flops = 2 * m * n * k
        moved_bytes = m * k + k * n + m * n * 2

        if common.CHECK_ONLY:
            print(f"ok | {format_shape(problem_shape)}", flush=True)
            continue

        sol_us = _bench(_solution_fn(solution_model, inputs), l2_flush)
        if opponent_fn is not None:
            opp_us = _bench(opponent_fn, l2_flush)

        tflops = flops / sol_us / 1e6
        pct_peak = 100 * tflops / peak_tflops
        gbps = moved_bytes / sol_us / 1e3
        speedup = opp_us / sol_us if opp_us else float("nan")
        speedups.append(speedup)
        rows.append(
            (
                format_shape(problem_shape),
                f"{sol_us:.2f} us",
                f"{tflops:,.1f}",
                f"{pct_peak:.2f}%",
                f"{gbps:,.0f}",
                f"{opp_us:.2f} us" if opp_us else "n/a",
                format_ratio(speedup) if opp_us else "-",
            )
        )

    if common.CHECK_ONLY:
        print("all shapes correct", flush=True)
        return

    geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    title = f"{torch.cuda.get_device_name()}  |  {PROBLEM}  |  fp8e4m3 x fp8e4m3 -> bf16"
    subtitle = f"L2 flushed per iter, median of {TIMED_ITERS}, warmup {WARMUP_ITERS}"
    headers = ("Shape", "Kernel", "TFLOPS", "%peak", "GB/s", "cuBLASLt FP8", "Speedup")
    footer = ("Geomean", "", "", "", "", "", format_ratio(geo))
    print_table(title, subtitle, headers, rows, footer)
