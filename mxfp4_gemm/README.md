# MXFP4 GEMM

Implement an MXFP4 weight-quantized GEMM on B200. Activations are BF16. Weights
are e2m1 (4-bit float: 1 sign, 2 exponent, 1 mantissa; magnitudes
{0, 0.5, 1, 1.5, 2, 3, 4, 6}), packed two per byte along K, with one E8M0
power-of-two scale byte per block of 32 K values. Accumulate in FP32, return BF16.

The kernel decodes nibbles and applies scales during loading. This is a
weight-only format; a native MXFP4 × MXFP4 operation would also need quantized
activations.

```bash
uv run modal run mxfp4_gemm/benchmark.py naive.py --check
uv run modal run mxfp4_gemm/benchmark.py naive.py
uv run modal run mxfp4_gemm/benchmark.py v1.py
```

Kernel files export `mxfp4(x, w_q, w_scales, y, stream)`. Allocation and compilation live
in `benchmark.py`. The stream argument lets CUDA graph capture include the kernel. Add `--shape 128` to run only that square GEMM size.

## Baseline

The single baseline is **cuBLAS BF16 GEMM** on the same weights, dequantized before timing. The kernel includes MXFP4 decoding; the baseline measures dense BF16 multiplication. Both outputs are checked against the reference.

Timing uses CUDA graphs with L2 flushed before each timed region: median of 50 iterations after 10 warmups. Only square sizes 128 through 8192 are benchmarked.

This workload uses BF16 activations. A native MXFP4 × MXFP4 baseline would require quantized activations too.
