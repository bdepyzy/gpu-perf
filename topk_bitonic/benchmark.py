import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from topk_bitonic.reference import get_inputs
from utils import benchmark_utils as bench

SHAPES = [
    {"batch": 1, "n": 32768, "k": 32},
    {"batch": 1, "n": 131072, "k": 64},
    {"batch": 8, "n": 131072, "k": 64},
    {"batch": 32, "n": 131072, "k": 64},
    {"batch": 8, "n": 262144, "k": 64},
]

app = bench.create_app(__file__, ("flashinfer-python[cu13]",), nvcc=True)


@app.function(gpu=bench.B200_GPU, timeout=bench.B200_TIMEOUT)
@torch.no_grad()
def run(solution_file: str, check: bool = False, shape: int | None = None):
    from flashinfer import top_k

    solution = bench.load("topk_bitonic", solution_file)
    rows, efficiencies, speedups = [], [], []
    for dims in SHAPES:
        batch, n, k = dims["batch"], dims["n"], dims["k"]
        torch.manual_seed(2026)
        x = get_inputs(**dims)[0].cuda()
        mutates_input = getattr(solution, "MUTATES_INPUT", True)
        work = torch.empty_like(x) if mutates_input else x
        values = torch.empty(batch, k, dtype=x.dtype, device="cuda")
        indices = torch.empty(batch, k, dtype=torch.int64, device="cuda")
        tensors = (work, values, indices)
        if hasattr(solution, "workspace_shape"):
            workspace_shape = solution.workspace_shape(batch, n, k)
            tensors += (torch.empty(workspace_shape, dtype=torch.float32, device="cuda"),
                        torch.empty(workspace_shape, dtype=torch.int32, device="cuda"))
        args = tuple(from_dlpack(t, assumed_align=16) for t in tensors)
        compiled = cute.compile(solution.topk, *args)

        def kernel():
            if mutates_input:
                work.copy_(x)
            compiled(*args)

        kernel()
        expected = torch.topk(x, k)
        baselines = {"PyTorch": lambda: torch.topk(x, k), "FlashInfer": lambda: top_k(x, k, sorted=True)}
        ok, message = bench.compare_topk([x], expected, (values, indices), dims, 1e-4)
        if not ok:
            raise RuntimeError(f"{bench.format_shape(dims)} | {message}")
        for name, baseline in baselines.items():
            ok, message = bench.compare_topk([x], expected, baseline(), dims, 1e-4)
            if not ok:
                raise RuntimeError(f"{name} | {bench.format_shape(dims)} | {message}")
        if check:
            print(f"ok | {bench.format_shape(dims)}", flush=True)
            continue

        moved = batch*n*4 + batch*k*12
        kernel_us = bench.bench_median(kernel)
        times = {name: bench.bench_median(fn) for name, fn in baselines.items()}
        best = min(times, key=times.get)
        baseline_us = times[best]
        sol_us = moved / (bench.B200_HBM_GBPS * 1000)
        efficiencies.append(sol_us / kernel_us)
        speedups.append(baseline_us / kernel_us)
        rows.append((bench.format_shape(dims), bench.format_time(kernel_us), bench.format_time(sol_us),
                     bench.format_percent(100 * efficiencies[-1]), bench.format_time(times["PyTorch"]),
                     bench.format_time(times["FlashInfer"]), best, bench.format_ratio(speedups[-1])))

    if check:
        print("all shapes correct", flush=True)
        return
    footer = ("Geomean", "", "", bench.format_percent(100 * bench.geomean(efficiencies)), "", "", "", bench.format_ratio(bench.geomean(speedups)))
    bench.print_table(f"{torch.cuda.get_device_name()} | topk_bitonic", "Sorted top-k; speedup against the fastest measured library per shape",
                      ("Shape", "Kernel", "HBM bound", "HBM eff.", "PyTorch", "FlashInfer", "Best", "Speedup"), rows, footer)


@app.local_entrypoint()
def main(*args):
    bench.launch(run, __file__, args, supports_shape=False)
