"""One checker and benchmark for the KernelBench-Hard exercises.

Run from the repo root:
    uv run python bench.py fp8_gemm check
    uv run python bench.py fp8_gemm bench
    uv run python bench.py fp8_gemm all

Each exercise lives in its own root folder and supplies solution.py with the
same Model API as its KernelBench-Hard reference.py.
"""
import argparse
import importlib.util
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HARD = ROOT / "KernelBench-Hard" / "problems"
B200_PEAK_TFLOPS = {"fp8": 4500.0, "bf16": 2250.0, "fp32": 90.0}
B200_PEAK_GBPS = 8000.0
PROBLEMS = {
    "fp8_gemm": {"source": "01_fp8_gemm", "regime": "compute", "peak": "fp8", "flops": "2*M*N*K", "bytes": "M*K + K*N + M*N*2", "tol": 0.15, "forbidden": ["torch._scaled_mm", "torch.ops.aten._scaled_mm"]},
    "kda": {"source": "02_kda_cutlass", "regime": "compute", "peak": "bf16", "flops": "4*B*T*H*(K*V + CHUNK_SIZE*K + CHUNK_SIZE*V)", "bytes": "B*T*H*K*2 + B*T*H*K*2 + B*T*H*V*2 + B*T*H*K*4 + B*T*H*2 + B*T*H*V*2", "tol": 0.05, "forbidden": ["fla.ops.kda", "fla.ops.chunk_kda", "chunk_kda", "fused_recurrent_kda", "naive_chunk_kda", "naive_recurrent_kda"]},
    "paged_attention": {"source": "03_paged_attention", "regime": "memory", "peak": "bf16", "flops": "4*batch*num_heads*seq_len*head_dim", "bytes": "2*batch*seq_len*num_kv_heads*head_dim*2 + batch*num_heads*head_dim*2*2", "tol": 0.02, "forbidden": ["vllm.attention", "flashinfer.batch_decode_with_paged_kv_cache", "flashinfer.decode", "torch.nn.functional.scaled_dot_product_attention", "F.scaled_dot_product_attention"]},
    "topk_bitonic": {"source": "05_topk_bitonic", "regime": "memory", "peak": "fp32", "flops": "batch*n*4", "bytes": "batch*n*4 + batch*k*(4+8)", "tol": 1e-4, "forbidden": ["torch.topk", "torch.kthvalue", "torch.sort", "torch.argsort", "Tensor.topk", "Tensor.kthvalue", "Tensor.sort", "Tensor.argsort", "torch.ops.aten.topk", "torch.ops.aten.sort", "torch.ops.aten.kthvalue"]},
    "sonic_moe_swiglu": {"source": "06_sonic_moe_swiglu", "regime": "compute", "peak": "bf16", "flops": "2*T_total*H*(2*I)", "bytes": "T_total*K*H*2 + E*H*(2*I)*2 + T_total*K*I*2", "tol": 0.02, "forbidden": ["torch.matmul", "torch.bmm", "torch.nn.functional.linear", "F.linear", "from sonic_moe", "import sonic_moe"]},
    "w4a16_gemm": {"source": "07_w4a16_gemm", "regime": "memory", "peak": "bf16", "flops": "2*M*N*K", "bytes": "M*K*2 + (K/2)*N + (K/128)*N*2 + (K/128)*N*2 + M*N*2", "tol": 0.10, "forbidden": ["bitsandbytes.functional.dequantize_4bit", "bitsandbytes.functional.gemv_4bit", "marlin_kernel.gemm", "torch.nn.functional.linear"]},
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _paths(problem):
    task_dir = ROOT / problem
    source_dir = HARD / PROBLEMS[problem]["source"]
    solution = task_dir / "solution.py"
    if not source_dir.is_dir(): raise RuntimeError(f"missing KernelBench-Hard source: {source_dir}")
    if not solution.is_file(): raise RuntimeError(f"missing your solution: {solution}")
    return task_dir, source_dir, solution


def _modules(problem):
    task_dir, source_dir, solution_path = _paths(problem)
    sys.path[:0] = [str(task_dir), str(source_dir)]
    reference = _load(f"{problem}_reference", source_dir / "reference.py")
    shapes = _load(f"{problem}_shapes", source_dir / "shapes.py")
    solution = _load(f"{problem}_solution", solution_path)
    if not hasattr(solution, "Model"): raise RuntimeError(f"{solution_path} must define Model")
    return task_dir, reference, shapes.SHAPES, solution


def _apply_shape(problem, reference, shape):
    if problem == "paged_attention":
        for name, key in [("BATCH", "batch"), ("NUM_HEADS", "num_heads"), ("NUM_KV_HEADS", "num_kv_heads"), ("HEAD_DIM", "head_dim"), ("SEQ_LEN", "seq_len"), ("PAGE_SIZE", "page_size")]: setattr(reference, name, shape[key])
    else:
        for name, value in shape.items(): setattr(reference, name, value)


def _to_cuda(values):
    return [value.to("cuda") if hasattr(value, "to") else value for value in values]


def _tolerance(tensor, override):
    import torch
    if override is not None: return override, override
    if tensor.dtype == torch.float32: return 1e-4, 1e-4
    if tensor.dtype in (torch.float16, torch.bfloat16): return 1e-2, 1e-2
    return 0.1, 0.1


def _compare(reference, actual, override):
    import torch
    if not isinstance(reference, torch.Tensor) or not isinstance(actual, torch.Tensor): return False, f"expected tensor, got {type(actual).__name__}"
    if reference.shape != actual.shape: return False, f"shape {tuple(actual.shape)} != {tuple(reference.shape)}"
    if actual.dtype != reference.dtype: return False, f"dtype {actual.dtype} != {reference.dtype}"
    if not torch.isfinite(actual).all(): return False, "output contains NaN or inf"
    atol, rtol = _tolerance(reference, override)
    if torch.allclose(reference, actual, atol=atol, rtol=rtol): return True, ""
    error = (reference.float() - actual.float()).abs()
    return False, f"max_abs={error.max().item():.6g} mean_abs={error.mean().item():.6g} atol={atol} rtol={rtol}"


def _check_forbidden(problem, task_dir):
    code = (task_dir / "solution.py").read_text()
    for name in PROBLEMS[problem]["forbidden"]:
        if re.search(re.escape(name), code): raise RuntimeError(f"forbidden op used: {name}")


def _models(problem, reference, solution, shape):
    import torch
    _apply_shape(problem, reference, shape)
    init_args = reference.get_init_inputs()
    ref_model = reference.Model(*init_args).to("cuda").eval()
    sol_model = solution.Model(*init_args).to("cuda").eval()
    sol_model.load_state_dict(ref_model.state_dict(), strict=True)
    return ref_model, sol_model


def _check_topk(inputs, ref_out, sol_out, shape, tol):
    import torch
    if not isinstance(sol_out, (tuple, list)) or len(sol_out) != 2: return False, "solution must return (values, indices)"
    ref_values, _ = ref_out
    values, indices = sol_out
    expected = (shape["batch"], shape["k"])
    if tuple(values.shape) != expected or tuple(indices.shape) != expected: return False, f"expected value/index shape {expected}"
    ok, message = _compare(ref_values.float(), values.float(), tol)
    if not ok: return False, f"values: {message}"
    indices = indices.to(torch.int64)
    if indices.min().item() < 0 or indices.max().item() >= shape["n"]: return False, "indices out of range"
    return _compare(ref_values.float(), torch.gather(inputs[0], -1, indices).float(), tol)


def check(problem, seeds):
    import torch
    task_dir, reference, shapes, solution = _modules(problem)
    _check_forbidden(problem, task_dir)
    for shape_index, shape in enumerate(shapes):
        ref_model, sol_model = _models(problem, reference, solution, shape)
        for seed in seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            inputs = _to_cuda(reference.get_inputs())
            with torch.no_grad():
                ref_out = ref_model(*inputs)
                sol_out = sol_model(*inputs)
            if problem == "topk_bitonic": ok, message = _check_topk(inputs, ref_out, sol_out, shape, PROBLEMS[problem]["tol"])
            else: ok, message = _compare(ref_out, sol_out, PROBLEMS[problem]["tol"])
            if not ok: raise RuntimeError(f"correctness failed | shape={shape_index} {shape} | seed={seed} | {message}")
    print(f"PASS | {problem} | shapes={len(shapes)} | seeds={len(seeds)}")


def _time(fn, inputs, warmup, iterations):
    import torch
    for _ in range(warmup): fn(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations): fn(*inputs)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def _eval_formula(formula, shape):
    return float(eval(formula, {"__builtins__": {}}, shape))


def evaluate(problem, warmup, iterations):
    import torch
    _, reference, shapes, solution = _modules(problem)
    meta = PROBLEMS[problem]
    fractions = []
    print(f"B200 | {problem} | warmup={warmup} | iterations={iterations}")
    for shape_index, shape in enumerate(shapes):
        ref_model, sol_model = _models(problem, reference, solution, shape)
        torch.manual_seed(2026)
        inputs = _to_cuda(reference.get_inputs())
        ref_us = _time(ref_model, inputs, warmup, iterations)
        sol_us = _time(sol_model, inputs, warmup, iterations)
        flops = _eval_formula(meta["flops"], shape)
        bytes_moved = _eval_formula(meta["bytes"], shape)
        sol_tflops = flops / sol_us / 1_000_000
        ref_tflops = flops / ref_us / 1_000_000
        sol_gbps = bytes_moved / sol_us / 1000
        ref_gbps = bytes_moved / ref_us / 1000
        if meta["regime"] == "compute":
            fraction = 100 * sol_tflops / B200_PEAK_TFLOPS[meta["peak"]]
            metric = f"{sol_tflops:.2f} TF/s ({fraction:.2f}% peak)"
            baseline = f"{ref_tflops:.2f} TF/s"
        else:
            fraction = 100 * sol_gbps / B200_PEAK_GBPS
            metric = f"{sol_gbps:.2f} GB/s ({fraction:.2f}% peak)"
            baseline = f"{ref_gbps:.2f} GB/s"
        fractions.append(max(fraction / 100, 1e-9))
        print(f"{shape_index}: {shape} | you {sol_us:.2f} us {metric} | ref {ref_us:.2f} us {baseline} | {100*ref_us/sol_us:.2f}% of ref time")
    gmean = math.exp(sum(math.log(value) for value in fractions) / len(fractions)) * 100
    print(f"geomean peak fraction: {gmean:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="KernelBench-Hard checker and B200 evaluator")
    parser.add_argument("problem", choices=PROBLEMS)
    parser.add_argument("command", choices=["check", "bench", "all"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--one-seed", action="store_true")
    args = parser.parse_args()
    if args.command in ("check", "all"): check(args.problem, (42,) if args.one_seed else (42, 123, 456))
    if args.command in ("bench", "all"): evaluate(args.problem, args.warmup, args.iterations)


if __name__ == "__main__": main()
