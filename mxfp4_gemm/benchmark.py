import modal
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from mxfp4_gemm.reference import get_inputs, reference
from utils import benchmark_utils as bench

SHAPES = bench.GEMM_SHAPES

app = bench.create_app(
    __file__, ("apache-tvm-ffi==0.1.14", "flashinfer-python[cu13]==0.6.18.post1"), nvcc=True
)
cache_volume = modal.Volume.from_name("learn-kernels-flashinfer-cache", create_if_missing=True)


@app.function(
    gpu=bench.B200_GPU, timeout=bench.B200_TIMEOUT,
    volumes={"/root/.cache/flashinfer": cache_volume},
)
@torch.no_grad()
def run(solution_file: str, check: bool = False, shape: int | None = None):
    cache_volume.reload()
    from flashinfer import autotune, mm_fp4
    from flashinfer.jit import env as jit_env
    from flashinfer.quantization import block_scale_interleave

    tuning_dir = jit_env.FLASHINFER_WORKSPACE_DIR / "autotune"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    solution = bench.load("mxfp4_gemm", solution_file).mxfp4
    rows, speedups = [], []
    for dims in bench.select_shapes(SHAPES, shape):
        M, N, K = dims["M"], dims["N"], dims["K"]
        A, B, sfa, sfb = get_inputs(M, N, K)
        C = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

        packed_sfa = block_scale_interleave(sfa).reshape(M // 128, -1)
        packed_sfb = block_scale_interleave(sfb.T.contiguous()).reshape(N // 128, -1)
        inputs = (A, B, packed_sfa, packed_sfb, C)

        args = tuple(from_dlpack(t, assumed_align=32 if i == 4 else 16) for i, t in enumerate(inputs))
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(solution, *args, stream, options="--enable-tvm-ffi")
        compiled(*inputs)
        torch.cuda.synchronize()
        kernel_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(kernel_graph):
            compiled(*inputs)

        C.fill_(float("nan"))
        kernel_graph.replay()
        expected = reference(A, B, sfa, sfb)
        ok, message = bench.compare(expected, C, 0.05)
        if not ok:
            raise RuntimeError(f"{bench.format_shape(dims)} | {message}")

        baseline_sfa = packed_sfa.reshape(M, K // 32)
        baseline_sfb = packed_sfb.reshape(N, K // 32).T
        baseline_output = torch.empty_like(C)

        def baseline():
            return mm_fp4(A, B, baseline_sfa, baseline_sfb, out=baseline_output,
                          block_size=32, use_nvfp4=False, backend="cute-dsl")

        tuning_file = tuning_dir / f"mxfp4-b200-{M}-{N}-{K}.json"
        with autotune(tuning_buckets=(M,), cache=str(tuning_file)):
            baseline()
        ok, message = bench.compare(expected, baseline(), 0.05)
        if not ok:
            raise RuntimeError(f"FlashInfer/CuTe MXFP4 | {bench.format_shape(dims)} | {message}")
        cache_volume.commit()
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
                      "MXFP4 x MXFP4, FP32 accumulation, BF16 output. FlashInfer/CuTe autotuned; preparation excluded. "
                      "CUDA graphs, L2 flushed, median of 50, warmup 10.",
                      ("Shape", "Kernel", "Kernel TFLOPS", "FlashInfer/CuTe MXFP4", "Baseline TFLOPS", "Speedup"), rows, footer)


@app.local_entrypoint()
def main(*args):
    bench.launch(run, __file__, args, supports_shape=True)
