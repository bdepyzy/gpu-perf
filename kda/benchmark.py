import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from kda.reference import Model, get_inputs
from utils import benchmark_utils as bench

SHAPES = [
    {"B": 1, "T": 512, "H": 16, "K": 128, "V": 128, "CHUNK_SIZE": 64},
    {"B": 4, "T": 1024, "H": 32, "K": 128, "V": 128, "CHUNK_SIZE": 64},
    {"B": 1, "T": 2048, "H": 32, "K": 128, "V": 128, "CHUNK_SIZE": 64},
    {"B": 1, "T": 4096, "H": 32, "K": 128, "V": 128, "CHUNK_SIZE": 64},
    {"B": 1, "T": 8192, "H": 64, "K": 128, "V": 128, "CHUNK_SIZE": 64},
]

app = bench.create_app(__file__, ("flash-linear-attention",))


@app.function(gpu=bench.B200_GPU, timeout=bench.B200_TIMEOUT)
@torch.no_grad()
def run(solution_file: str, check: bool = False, shape: int | None = None):
    from fla.ops.kda import chunk_kda

    solution = bench.load("kda", solution_file)
    rows, efficiencies, speedups = [], [], []
    for dims in SHAPES:
        B, T, H, K, V, C = (dims[key] for key in ("B", "T", "H", "K", "V", "CHUNK_SIZE"))
        model = Model(B, T, H, K, V, C).cuda().eval()
        torch.manual_seed(2026)
        inputs = [x.cuda() for x in get_inputs(**dims)]
        output = torch.empty_like(inputs[2])
        args = tuple(from_dlpack(t, assumed_align=16) for t in (*inputs, output))
        compiled = cute.compile(solution.kda, *args)
        compiled(*args)
        expected = model(*inputs)
        ok, message = bench.compare(expected, output, 0.05)
        if not ok:
            raise RuntimeError(f"{bench.format_shape(dims)} | {message}")
        def baseline():
            return chunk_kda(
                *inputs, scale=model.scale, initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
            )[0]
        ok, message = bench.compare(expected, baseline(), 0.05)
        if not ok:
            raise RuntimeError(f"FLA chunk_kda | {bench.format_shape(dims)} | {message}")
        if check:
            print(f"ok | {bench.format_shape(dims)}", flush=True)
            continue

        flops = 4*B*T*H*(K*V + C*K + C*V)
        moved = B*T*H*(8*K + 4*V + 2)
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
    bench.print_table(f"{torch.cuda.get_device_name()} | kda", "Baseline: FLA chunk_kda",
                      ("Shape", "Kernel", "Roofline", "Roof eff.", "Baseline", "Speedup"), rows, footer)


@app.local_entrypoint()
def main(*args):
    bench.launch(run, __file__, args, supports_shape=False)
