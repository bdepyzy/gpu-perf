import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from benchmark import common


DEPS = ()
SHAPES = [
    (1024, 1024),
    (4096, 1024),
    (1024, 4096),
    (4096, 4096),
]
VARIANTS = {
    "mile1_naive": "naive.py",
    "mile1_vector": "vector.py",
    "mile1_transpose": "transpose.py",
}


def evaluate(workload):
    solution = common.load(f"{workload}_solution", common.ROOT / "mile1" / VARIANTS[workload])
    solution_ratios = []
    opponent_ratios = []
    table_rows = []
    variant = workload.removeprefix("mile1_")

    for rows, columns in SHAPES:
        torch.manual_seed(2026)
        a = torch.randn(rows, columns, device="cuda", dtype=torch.float16)
        b = torch.randn(rows, columns, device="cuda", dtype=torch.float16)
        output = torch.empty_like(a)
        opponent_output = torch.empty_like(a)
        a_cute = from_dlpack(a, assumed_align=16)
        b_cute = from_dlpack(b, assumed_align=16)
        output_cute = from_dlpack(output, assumed_align=16)
        add = cute.compile(solution.add, a_cute, b_cute, output_cute)

        add(a_cute, b_cute, output_cute)
        torch.add(a, b, out=opponent_output)
        torch.testing.assert_close(output, opponent_output, rtol=1e-3, atol=1e-3)

        shape = f"{rows}x{columns}"
        if common.CHECK_ONLY:
            print(f"ok | {shape}", flush=True)
            continue

        solution_us = common.bench_median(lambda: add(a_cute, b_cute, output_cute))
        opponent_us = common.bench_median(lambda: torch.add(a, b, out=opponent_output))
        bytes_moved = 3 * rows * columns * 2
        sol_us = common.roofline_us(rows * columns, bytes_moved, "fp32")
        solution_ratios.append(sol_us / solution_us)
        opponent_ratios.append(opponent_us / solution_us)
        table_rows.append((shape, common.format_time(solution_us), common.format_time(sol_us), common.format_percent(100 * sol_us / solution_us), common.format_time(opponent_us), common.format_ratio(opponent_ratios[-1])))

    if common.CHECK_ONLY:
        print("all shapes correct", flush=True)
        return

    title = f"{torch.cuda.get_device_name()}  |  mile1/{variant}  |  FP16 add"
    footer = ("Geomean", "", "", common.format_percent(100 * common.geomean(solution_ratios)), "", common.format_ratio(common.geomean(opponent_ratios)))
    common.print_table(title, "Opponent: PyTorch CUDA add", ("Shape", "Kernel", "SOL", "SOL eff.", "Opponent", "Perf. vs opp."), table_rows, footer)
