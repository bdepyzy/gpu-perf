# FP6 GEMM

Implement an FP6 weight-quantized GEMM on B200. Activations are BF16. Weights
are e3m2 (6-bit float: 1 sign, 3 exponent, 2 mantissa; magnitudes up to 28),
packed four codes per three bytes along K, with no block scales. Accumulate in
FP32, return BF16.

Never materialize the dequantized weight matrix: unpack 6-bit fields and feed
fragments straight into the math. The 6-bit packing makes loads intentionally
awkward — that is the exercise. Native FP6 MMA would also require lower-precision activations; this workload keeps BF16 activations.

```bash
uv run modal run fp6_gemm/benchmark.py naive.py --check
uv run modal run fp6_gemm/benchmark.py naive.py
```

Kernel files export `fp6(x, w_q, y)`. Allocation and compilation live in
`benchmark.py`. Add `--shape 128` to run only that square GEMM size.

## Baseline

The baseline is cuBLAS BF16 on the same weights, dequantized before timing. The roofline is a theoretical BF16 compute/HBM bound, not a library measurement.
