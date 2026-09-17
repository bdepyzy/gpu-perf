import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from mxfp4_gemm.reference import Model, get_inputs
from utils import benchmark_utils as bench

SHAPES = bench.GEMM_SHAPES

app = bench.create_app(__file__, ("apache-tvm-ffi==0.1.14",))


@app.function(gpu=bench.B200_GPU, timeout=bench.B200_TIMEOUT)
@torch.no_grad()
def run(solution_file: str, check: bool = False, shape: int | None = None):
    torch.backends.cuda.preferred_blas_library("cublas")
    solution = bench.load("mxfp4_gemm", solution_file)
    rows, speedups = [], []
    for dims in bench.select_shapes(SHAPES, shape):
        M, N, K = dims["M"], dims["N"], dims["K"]
        model = Model(M, N, K).cuda().eval()
        torch.manual_seed(2026)
        x = get_inputs(M, K)[0].cuda()
        output = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        args = tuple(from_dlpack(t, assumed_align=16) for t in (x, model.w_q, model.w_scales, output))
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(solution.mxfp4, *args, stream, options="--enable-tvm-ffi")
        inputs = (x, model.w_q, model.w_scales, output)
        compiled(*inputs)
        torch.cuda.synchronize()
        kernel_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(kernel_graph):
            compiled(*inputs)
        # Check replay itself, so a kernel on the wrong stream cannot pass with stale output.
        output.fill_(float("nan"))
        kernel_graph.replay()
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
        kernel_us = bench.bench_median(lambda: compiled(*inputs), iters=50, cuda_graph=True)
        baseline_us = bench.bench_median(baseline, iters=50, cuda_graph=True)
        speedups.append(baseline_us / kernel_us)
        rows.append((bench.format_shape(dims), bench.format_time(kernel_us), f"{flops/kernel_us/1e6:,.1f}",
                     bench.format_time(baseline_us), f"{flops/baseline_us/1e6:,.1f}",
                     bench.format_ratio(speedups[-1])))

    if check:
        print("all shapes correct", flush=True)
        return
    footer = ("Geomean", "", "", "", "", bench.format_ratio(bench.geomean(speedups)))
    bench.print_table(f"{torch.cuda.get_device_name()} | mxfp4_gemm",
                      "cuBLAS BF16: weights dequantized before timing. CUDA graphs, L2 flushed, median of 50, warmup 10.",
                      ("Shape", "Kernel", "Kernel TFLOPS", "cuBLAS BF16", "cuBLAS TFLOPS", "Speedup"), rows, footer)


@app.local_entrypoint()
def main(*args):
    bench.launch(run, __file__, args, supports_shape=True)
