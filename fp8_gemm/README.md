# FP8 GEMM

Implement `Y = X @ W.T` on B200. `X` and `W` are FP8 E4M3, accumulation is FP32, and `Y` is BF16. Keep the `Model` interface in `kernel.py`; the benchmark owns the shapes and the correctness oracle.

```bash
# Local CUDA smoke test
uv run python -m fp8_gemm.kernel

# Modal correctness and benchmark
uv run modal run fp8_gemm/kernel.py --check
uv run modal run fp8_gemm/kernel.py --shape 512
uv run modal run fp8_gemm/kernel.py
```
