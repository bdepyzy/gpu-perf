import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from fp6_gemm.reference import Model, get_inputs
from utils import benchmark_utils as bench

SHAPES = bench.GEMM_SHAPES

app = bench.create_app(__file__)


@app.function(gpu=bench.B200_GPU, timeout=bench.B200_TIMEOUT)
@torch.no_grad()
def run(solution_file: str, check: bool = False, shape: int | None = None):
    torch.backends.cuda.preferred_blas_library("cublas")
    solution = bench.load("fp6_gemm", solution_file)
    rows, efficiencies, speedups = [], [], []
    for dims in bench.select_shapes(SHAPES, shape):
        M, N, K = dims["M"], dims["N"], dims["K"]
        model = Model(M, N, K).cuda().eval()
        torch.manual_seed(2026)
        x = get_inputs(M, K)[0].cuda()
        output = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        args = tuple(from_dlpack(t, assumed_align=16) for t in (x, model.w_q, output))
        compiled = cute.compile(solution.fp6, *args)
        compiled(*args)
        expected = model(x)
        ok, message = bench.compare(expected, output, 0.05)
        if not ok:
            raise RuntimeError(f"{bench.format_shape(dims)} | {message}")
        weight = model.dequantize()
        baseline_output = torch.empty_like(output)
        baseline = lambda: torch.mm(x, weight, out=baseline_output)
        ok, message = bench.compare(expected, baseline(), 0.05)
        if not ok:
            raise RuntimeError(f"cuBLAS BF16 | {bench.format_shape(dims)} | {message}")
        if check:
            print(f"ok | {bench.format_shape(dims)}", flush=True)
            continue

        flops = 2*M*N*K
        moved = 2*M*K + K*N*3//4 + 2*M*N
        kernel_us = bench.bench_median(lambda: compiled(*args))
        baseline_us = bench.bench_median(baseline)
        sol_us = bench.roofline_us(flops, moved, "bf16")
        efficiencies.append(sol_us / kernel_us)
        speedups.append(baseline_us / kernel_us)
        rows.append((bench.format_shape(dims), bench.format_time(kernel_us), bench.format_time(sol_us),
                     bench.format_percent(100 * efficiencies[-1]), bench.format_time(baseline_us),
                     bench.format_ratio(speedups[-1])))

    if check:
        print("all shapes correct", flush=True)
        return
    footer = ("Geomean", "", "", bench.format_percent(100 * bench.geomean(efficiencies)), "", bench.format_ratio(bench.geomean(speedups)))
    bench.print_table(f"{torch.cuda.get_device_name()} | fp6_gemm", "cuBLAS BF16; weights dequantized before timing. Roofline is an estimate, not a library baseline.",
                      ("Shape", "Kernel", "Roofline", "Roof eff.", "cuBLAS BF16", "Speedup"), rows, footer)


@app.local_entrypoint()
def main(*args):
    bench.launch(run, __file__, args, supports_shape=True)
