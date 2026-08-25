import importlib.util
import math
import sys
from pathlib import Path


B200_PEAK_TFLOPS = {"fp8": 4500.0, "bf16": 2250.0, "fp32": 90.0}
B200_HBM_GBPS = 8000.0
WARMUP = 10
ITERATIONS = 30

ROOT = Path(__file__).resolve().parent.parent
REFERENCES = Path(__file__).resolve().parent / "references"

PROBLEMS = {
    "fp8_gemm": {
        "peak": "fp8",
        "flops": "2*M*N*K",
        "bytes": "M*K + K*N + M*N*2",
        "tol": 0.15,
    },
    "kda": {
        "peak": "bf16",
        "flops": "4*B*T*H*(K*V + CHUNK_SIZE*K + CHUNK_SIZE*V)",
        "bytes": "B*T*H*K*2 + B*T*H*K*2 + B*T*H*V*2 + B*T*H*K*4 + B*T*H*2 + B*T*H*V*2",
        "tol": 0.05,
        "forbidden": [
            "fla.ops.kda",
            "fla.ops.chunk_kda",
            "chunk_kda",
            "fused_recurrent_kda",
            "naive_chunk_kda",
            "naive_recurrent_kda",
        ],
    },
    "paged_attention": {
        "peak": "bf16",
        "flops": "4*batch*num_heads*seq_len*head_dim",
        "bytes": "2*batch*seq_len*num_kv_heads*head_dim*2 + batch*num_heads*head_dim*2*2",
        "tol": 0.02,
        "forbidden": [
            "vllm.attention",
            "flashinfer.batch_decode_with_paged_kv_cache",
            "flashinfer.decode",
            "torch.nn.functional.scaled_dot_product_attention",
            "F.scaled_dot_product_attention",
        ],
    },
    "topk_bitonic": {
        "peak": "fp32",
        "flops": "batch*n*4",
        "bytes": "batch*n*4 + batch*k*(4+8)",
        "tol": 1e-4,
        "forbidden": [
            "torch.topk",
            "torch.kthvalue",
            "torch.sort",
            "torch.argsort",
            "Tensor.topk",
            "Tensor.kthvalue",
            "Tensor.sort",
            "Tensor.argsort",
            "torch.ops.aten.topk",
            "torch.ops.aten.sort",
            "torch.ops.aten.kthvalue",
        ],
    },
    "w4a16_gemm": {
        "peak": "bf16",
        "flops": "2*M*N*K",
        "bytes": "M*K*2 + (K/2)*N + (K/128)*N*2 + (K/128)*N*2 + M*N*2",
        "tol": 0.10,
    },
}


def configure_root(root):
    global ROOT
    ROOT = Path(root)
    sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def problem_modules(problem):
    meta = PROBLEMS[problem]
    task_dir = ROOT / problem
    source_dir = REFERENCES / problem
    kernel_path = task_dir / "kernel.py"
    sys.path[:0] = [str(task_dir), str(source_dir)]
    reference = load(f"{problem}_reference", source_dir / "reference.py")
    solution = load(f"{problem}_solution", kernel_path)
    return source_dir, reference, solution


def apply_shape(problem, reference, shape):
    if problem == "paged_attention":
        names = (
            ("BATCH", "batch"),
            ("NUM_HEADS", "num_heads"),
            ("NUM_KV_HEADS", "num_kv_heads"),
            ("HEAD_DIM", "head_dim"),
            ("SEQ_LEN", "seq_len"),
            ("PAGE_SIZE", "page_size"),
        )
        for name, key in names:
            setattr(reference, name, shape[key])
    else:
        for name, value in shape.items():
            setattr(reference, name, value)


def make_models(problem, reference, solution, shape):
    apply_shape(problem, reference, shape)
    init_args = reference.get_init_inputs()
    reference_model = reference.Model(*init_args).to("cuda").eval()
    solution_model = solution.Model(*init_args).to("cuda").eval()
    solution_model.load_state_dict(reference_model.state_dict(), strict=True)
    return reference_model, solution_model


def to_cuda(values):
    return [value.to("cuda") if hasattr(value, "to") else value for value in values]


def compare(reference, actual, tolerance):
    import torch

    if not isinstance(reference, torch.Tensor) or not isinstance(actual, torch.Tensor):
        return False, f"expected tensor, got {type(actual).__name__}"
    if reference.shape != actual.shape:
        return False, f"shape {tuple(actual.shape)} != {tuple(reference.shape)}"
    if actual.dtype != reference.dtype:
        return False, f"dtype {actual.dtype} != {reference.dtype}"
    if not torch.isfinite(actual).all():
        return False, "output contains NaN or inf"
    if torch.allclose(reference, actual, atol=tolerance, rtol=tolerance):
        return True, ""
    error = (reference.float() - actual.float()).abs()
    return False, f"max_abs={error.max().item():.6g} mean_abs={error.mean().item():.6g}"


def compare_topk(inputs, reference, actual, shape, tolerance):
    import torch

    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        return False, "solution must return (values, indices)"
    reference_values, _ = reference
    values, indices = actual
    expected = (shape["batch"], shape["k"])
    if tuple(values.shape) != expected or tuple(indices.shape) != expected:
        return False, f"expected value/index shape {expected}"
    ok, message = compare(reference_values.float(), values.float(), tolerance)
    if not ok:
        return False, f"values: {message}"
    indices = indices.to(torch.int64)
    if indices.min().item() < 0 or indices.max().item() >= shape["n"]:
        return False, "indices out of range"
    return compare(reference_values.float(), torch.gather(inputs[0], -1, indices).float(), tolerance)


def time_cuda(fn, warmup=WARMUP, iterations=ITERATIONS):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def evaluate_formula(formula, shape):
    return float(eval(formula, {"__builtins__": {}}, shape))


def roofline_us(flops, bytes_moved, peak):
    compute_us = flops / (B200_PEAK_TFLOPS[peak] * 1_000_000)
    memory_us = bytes_moved / (B200_HBM_GBPS * 1_000)
    return max(compute_us, memory_us)


def format_time(time_us):
    return f"{time_us * 1000:.1f} ns" if time_us < 1 else f"{time_us:.2f} us"


def format_percent(value):
    return f"{value:.3f}%" if value < 0.1 else f"{value:.1f}%"


def format_shape(shape):
    aliases = {"batch": "B", "num_heads": "H", "num_kv_heads": "KV", "head_dim": "D", "seq_len": "L", "page_size": "P", "CHUNK_SIZE": "C"}
    return " ".join(f"{aliases.get(name, name)}{value}" for name, value in shape.items())


def geomean(values):
    return math.exp(sum(math.log(max(value, 1e-9)) for value in values) / len(values))


def print_table(title, subtitle, headers, rows, footer):
    widths = [max(len(str(row[index])) for row in [headers, *rows, footer]) for index in range(len(headers))]

    def render(row):
        return "  ".join(str(value).ljust(widths[index]) if index == 0 else str(value).rjust(widths[index]) for index, value in enumerate(row))

    rule = "  ".join("-" * width for width in widths)
    prelude = [title, subtitle, ""] if subtitle else [title, ""]
    lines = [*prelude, render(headers), rule, *(render(row) for row in rows), rule, render(footer)]
    print("\n".join(lines), flush=True)


def generic_evaluate(problem, shapes, opponent_factory):
    import torch

    source_dir, reference, solution = problem_modules(problem)
    meta = PROBLEMS[problem]
    sol_ratios = []
    opponent_ratios = []
    rows = []
    opponent_name = None

    for index, shape in enumerate(shapes):
        reference_model, solution_model = make_models(problem, reference, solution, shape)
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)
        inputs = to_cuda(reference.get_inputs())
        with torch.no_grad():
            reference_out = reference_model(*inputs)
            solution_out = solution_model(*inputs)
        checker = compare_topk if problem == "topk_bitonic" else None
        ok, message = (
            checker(inputs, reference_out, solution_out, shape, meta["tol"])
            if checker
            else compare(reference_out, solution_out, meta["tol"])
        )
        if not ok:
            raise RuntimeError(f"solution correctness failed | shape={index} | {message}")

        current_opponent, opponent_fn, comparable = opponent_factory(reference_model, inputs, shape, source_dir)
        opponent_name = current_opponent
        opponent_out = opponent_fn()
        torch.cuda.synchronize()
        if comparable:
            ok, message = (
                checker(inputs, reference_out, opponent_out, shape, meta["tol"])
                if checker
                else compare(reference_out, opponent_out, meta["tol"])
            )
            if not ok:
                raise RuntimeError(f"{current_opponent} correctness failed | shape={index} | {message}")

        solution_us = time_cuda(lambda: solution_model(*inputs))
        opponent_us = time_cuda(opponent_fn)
        flops = evaluate_formula(meta["flops"], shape)
        moved = evaluate_formula(meta["bytes"], shape)
        sol_us = roofline_us(flops, moved, meta["peak"])
        sol_ratios.append(sol_us / solution_us)
        opponent_ratios.append(opponent_us / solution_us)
        rows.append((format_shape(shape), format_time(solution_us), format_time(sol_us), format_percent(100 * sol_us / solution_us), format_time(opponent_us), format_percent(100 * opponent_us / solution_us)))

    sol_gmean = geomean(sol_ratios)
    opponent_gmean = geomean(opponent_ratios)
    title = f"{torch.cuda.get_device_name()}  |  {problem}"
    footer = ("Geomean", "", "", format_percent(100 * sol_gmean), "", format_percent(100 * opponent_gmean))
    print_table(title, f"Opponent: {opponent_name}", ("Shape", "Kernel", "SOL", "SOL eff.", "Opponent", "Perf. vs opp."), rows, footer)
