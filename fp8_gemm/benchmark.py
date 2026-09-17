import math

import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from fp8_gemm.reference import Model, get_inputs
from utils import benchmark_utils as bench

SHAPES = bench.GEMM_SHAPES

app = bench.create_app(__file__)


@app.function(gpu=bench.B200_GPU, timeout=bench.B200_TIMEOUT)
@torch.no_grad()
def run(solution_file: str, check: bool = False, shape: int | None = None):
    solution = bench.load("fp8_gemm", solution_file)
    rows, speedups = [], []
    for dims in bench.select_shapes(SHAPES, shape):
        M, N, K = dims["M"], dims["N"], dims["K"]
        model = Model(M, N, K).cuda().eval()
        torch.manual_seed(2026)
        x = get_inputs(M, K)[0].cuda()
        weight = model.weight.detach()
        output = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        args = tuple(from_dlpack(t, assumed_align=16) for t in (x.view(torch.uint8), weight.view(torch.uint8), output))
        compiled = cute.compile(solution.gemm, *args)
        compiled(*args)
        expected = model(x)
        ok, message = bench.compare(expected, output, 0.05)
        if not ok:
            raise RuntimeError(f"{bench.format_shape(dims)} | {message}")

        scale_a = torch.tensor(1.0, device="cuda")
        scale_b = torch.tensor(1.0, device="cuda")
        baseline = lambda: torch._scaled_mm(x, weight.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
        baseline_ok = False
        try:
            actual = baseline()
            error = ((expected.float() - actual.float()).norm() / expected.float().norm().clamp(min=1e-6)).item()
            baseline_ok = math.isfinite(error) and error < 0.05
            if not baseline_ok:
                print(f"cuBLASLt FP8 skipped | {bench.format_shape(dims)} | rel_fro={error:.4f}")
        except RuntimeError as error:
            print(f"cuBLASLt FP8 skipped | {bench.format_shape(dims)} | {error}")
        if check:
            print(f"ok | {bench.format_shape(dims)}", flush=True)
            continue

        kernel_us = bench.bench_median(lambda: compiled(*args), iters=50)
        baseline_us = bench.bench_median(baseline, iters=50) if baseline_ok else None
        flops = 2*M*N*K
        speedup = baseline_us / kernel_us if baseline_us else float("nan")
        speedups.append(speedup)
        rows.append((bench.format_shape(dims), bench.format_time(kernel_us), f"{flops/kernel_us/1e6:,.1f}",
                     bench.format_time(baseline_us) if baseline_us else "n/a",
                     f"{flops/baseline_us/1e6:,.1f}" if baseline_us else "n/a",
                     bench.format_ratio(speedup) if baseline_us else "-"))

    if check:
        print("all shapes correct", flush=True)
        return
    footer = ("Geomean", "", "", "", "", bench.format_ratio(bench.geomean(speedups)))
    bench.print_table(f"{torch.cuda.get_device_name()} | fp8_gemm | fp8e4m3 x fp8e4m3 -> bf16",
                      "L2 flushed per iter, median of 50, warmup 10",
                      ("Shape", "Kernel", "Kernel TFLOPS", "cuBLASLt FP8", "cuBLASLt TFLOPS", "Speedup"), rows, footer)


@app.local_entrypoint()
def main(*args):
    bench.launch(run, __file__, args, supports_shape=True)
