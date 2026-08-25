import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from benchmark import common


DEPS = ()
VARIANTS = {
    "mile2_naive": "naive.py",
    "mile2_smem_tiled": "smem_tiled.py",
    "mile2_tma_pipeline": "tma_pipeline.py",
    "mile2_tcgen": "tcgen.py",
}
SHAPES = {
    "mile2_naive": [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ],
    "mile2_smem_tiled": [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ],
    "mile2_tma_pipeline": [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ],
    # The current tcgen05 lesson supports one K tile and one N tile. Larger K
    # and N require the next descriptor/epilogue pipeline milestone.
    "mile2_tcgen": [
        (128, 128, 64),
        (256, 128, 64),
        (512, 128, 64),
        (1024, 128, 64),
    ],
}


def evaluate(workload):
    solution = common.load(f"{workload}_solution", common.ROOT / "mile2" / VARIANTS[workload])
    entrypoint = solution.Kernel() if hasattr(solution, "Kernel") else solution.gemm
    solution_ratios = []
    opponent_ratios = []
    rows = []
    variant = workload.removeprefix("mile2_")

    for m, n, k in SHAPES[workload]:
        torch.manual_seed(2026)
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        if workload == "mile2_tcgen":
            # Logical B is KxN, backed by the K-major storage tcgen05 expects.
            b = torch.randn(n, k, device="cuda", dtype=torch.float16).T
        else:
            b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        output = torch.empty(m, n, device="cuda", dtype=torch.float16)
        opponent_output = torch.empty_like(output)
        a_cute = from_dlpack(a, assumed_align=16)
        b_cute = from_dlpack(b, assumed_align=16)
        output_cute = from_dlpack(output, assumed_align=16)
        gemm = cute.compile(entrypoint, a_cute, b_cute, output_cute)

        gemm(a_cute, b_cute, output_cute)
        torch.mm(a, b, out=opponent_output)
        torch.testing.assert_close(output, opponent_output, rtol=1e-2, atol=5e-1)

        solution_us = common.time_cuda(lambda: gemm(a_cute, b_cute, output_cute))
        opponent_us = common.time_cuda(lambda: torch.mm(a, b, out=opponent_output))
        flops = 2 * m * n * k
        bytes_moved = 2 * (m * k + k * n + m * n)
        sol_us = common.roofline_us(flops, bytes_moved, "bf16")
        solution_ratios.append(sol_us / solution_us)
        opponent_ratios.append(opponent_us / solution_us)
        shape = f"{m}x{n}x{k}"
        rows.append((shape, common.format_time(solution_us), common.format_time(sol_us), common.format_percent(100 * sol_us / solution_us), common.format_time(opponent_us), common.format_percent(100 * opponent_us / solution_us)))

    title = f"{torch.cuda.get_device_name()}  |  mile2/{variant}  |  FP16 GEMM"
    footer = ("Geomean", "", "", common.format_percent(100 * common.geomean(solution_ratios)), "", common.format_percent(100 * common.geomean(opponent_ratios)))
    common.print_table(title, "Opponent: PyTorch/cuBLASLt", ("Shape", "Kernel", "SOL", "SOL eff.", "Opponent", "Perf. vs opp."), rows, footer)
