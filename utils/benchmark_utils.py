import argparse
import importlib.util
import math
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = Path("/workspace")
B200_GPU = "B200"
B200_TIMEOUT = 30 * 60

B200_PEAK_TFLOPS = {"fp8": 4500.0, "bf16": 2250.0}
B200_HBM_GBPS = 8000.0
WARMUP = 10
ITERATIONS = 30
GEMM_SHAPES = [{"M": size, "N": size, "K": size} for size in (128, 256, 512, 1024, 2048, 4096, 8192)]
_L2_FLUSH = None


def create_app(benchmark_file, deps=(), nvcc=False):
    if not modal.is_local():
        return modal.App()
    task_dir = Path(benchmark_file).resolve().parent
    image = modal.Image.debian_slim(python_version="3.13").env(
        {"PYTHONPATH": str(REMOTE_ROOT)}
    ).uv_pip_install(
        "torch==2.11.0",
        "nvidia-cutlass-dsl[cu13]==4.7.0",
        "numpy==2.5.2",
        "einops==0.8.2",
        *deps,
    )
    if nvcc:
        image = image.apt_install("g++").uv_pip_install(
            "cuda-toolkit[nvcc,cccl,crt,nvvm,cudart]==13.0.2", "ninja"
        ).env({"CUDA_HOME": "/usr/local/lib/python3.13/site-packages/nvidia/cu13"}).run_commands(
            'mkdir -p "$CUDA_HOME/lib64"',
            'ln -s ../lib/libcudart.so.13 "$CUDA_HOME/lib64/libcudart.so"',
        )

    image = image.env({"CUTE_DSL_SHOW_STACKTRACE": "1"})
    image = image.add_local_dir(
        Path(__file__).resolve().parent, str(REMOTE_ROOT / "utils")
    ).add_local_dir(task_dir, str(REMOTE_ROOT / task_dir.name))
    return modal.App(f"bench-{task_dir.name}", image=image)


def launch(run, benchmark_file, args, supports_shape=False):
    task_dir = Path(benchmark_file).resolve().parent
    parser = argparse.ArgumentParser(
        prog=f"uv run modal run {task_dir.name}/benchmark.py",
        description=f"Check and benchmark a {task_dir.name} kernel on B200.",
    )
    parser.add_argument("solution_file", help="kernel filename or path, e.g. v2.py or fp8_gemm/v2.py")
    parser.add_argument("--check", action="store_true", help="check correctness without timing")
    if supports_shape:
        parser.add_argument("--shape", type=int, help="run one square GEMM size, e.g. 128")
    options = parser.parse_args(args)
    solution = Path(options.solution_file)
    solution = (task_dir / solution if solution.parent == Path(".") else solution).resolve()
    if solution.parent != task_dir or solution.suffix != ".py":
        parser.error(f"select a Python kernel in {task_dir.name}/")
    if solution.name in {"benchmark.py", "reference.py", "sota.py"}:
        parser.error("select a kernel file, e.g. naive.py or v1.py")
    if not solution.is_file():
        parser.error(f"kernel file does not exist: {solution}")
    run.remote(solution.name, options.check, getattr(options, "shape", None))


def load(workload, filename):
    path = ROOT / workload / filename
    name = f"{workload}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def select_shapes(shapes, size):
    if size is None:
        return shapes
    selected = [shape for shape in shapes if shape == {"M": size, "N": size, "K": size}]
    if not selected:
        available = ", ".join(str(shape["M"]) for shape in shapes if shape["M"] == shape["N"] == shape["K"])
        raise ValueError(f"unknown square shape {size}; available sizes: {available}")
    return selected


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

    relative = (reference.float() - actual.float()).norm() / reference.float().norm().clamp_min(1e-6)
    if torch.allclose(reference, actual, atol=tolerance, rtol=tolerance) and relative <= tolerance:
        return True, ""
    error = (reference.float() - actual.float()).abs()
    return False, f"max_abs={error.max().item():.6g} mean_abs={error.mean().item():.6g} rel_l2={relative.item():.6g}"


def compare_topk(inputs, reference, actual, shape, tolerance):
    import torch

    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        return False, "solution must return (values, indices)"
    reference_values, _ = reference
    values, indices = actual
    expected = (shape["batch"], shape["k"])
    if tuple(values.shape) != expected or tuple(indices.shape) != expected:
        return False, f"expected value/index shape {expected}"
    ok, message = compare(reference_values, values, tolerance)
    if not ok:
        return False, f"values: {message}"
    if indices.dtype not in (torch.int32, torch.int64):
        return False, "indices must be integers"
    indices = indices.to(torch.int64)
    if indices.min().item() < 0 or indices.max().item() >= shape["n"]:
        return False, "indices out of range"
    ordered = indices.sort(dim=-1).values
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        return False, "indices must be unique within each row"
    return compare(reference_values.float(), torch.gather(inputs[0], -1, indices).float(), tolerance)


def _flush_l2():
    global _L2_FLUSH
    if _L2_FLUSH is None:
        import torch

        _L2_FLUSH = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    _L2_FLUSH.zero_()


def bench_median(fn, warmup=WARMUP, iters=ITERATIONS, cuda_graph=False):
    import statistics
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=cuda_graph)
    end = torch.cuda.Event(enable_timing=True, external=cuda_graph)
    graph = None
    if cuda_graph:
        _flush_l2()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _flush_l2()
            start.record()
            fn()
            end.record()
        graph.replay()
        end.synchronize()
    samples = []
    for _ in range(iters):
        if graph is not None:
            graph.replay()
        else:
            _flush_l2()
            start.record()
            fn()
            end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def roofline_us(flops, bytes_moved, peak):
    compute_us = flops / (B200_PEAK_TFLOPS[peak] * 1_000_000)
    memory_us = bytes_moved / (B200_HBM_GBPS * 1_000)
    return max(compute_us, memory_us)


def format_time(time_us):
    return f"{time_us * 1000:.1f} ns" if time_us < 1 else f"{time_us:.2f} us"


def format_percent(value):
    return f"{value:.3f}%" if value < 0.1 else f"{value:.1f}%"


def format_ratio(ratio):
    return f"{ratio:.3f}x" if ratio < 0.1 else f"{ratio:.2f}x"


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
