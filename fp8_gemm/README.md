# FP8 GEMM

```bash
uv run modal run fp8_gemm/benchmark.py v1.py
uv run modal run fp8_gemm/benchmark.py naive.py --check --shape 128
uv run modal run fp8_gemm/benchmark.py fp8_gemm/v2.py --shape 128
```

Kernel files export `gemm(mA, mB, mC)`; `benchmark.py` prepares tensors, compiles,
checks correctness, and times the kernel against cuBLASLt FP8.

Geomean speedups from M,N,K being 128 -> ... -> 8192

v1.py | warp-specialized, multi-stage, tcgen05 | 0.63x

N/A
128-wide CTA_1 MMA
No TMA multicast
No rasterization
Single accumulator
