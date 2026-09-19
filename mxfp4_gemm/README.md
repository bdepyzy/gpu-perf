# MXFP4 × MXFP4 GEMM

Compute `C = A @ B` with both operands in MXFP4: packed E2M1 values and
one E8M0 power-of-two scale per 32 values along K. Accumulation is FP32;
output is BF16.

Kernel files export `mxfp4(A, B, sfa, sfb, C, stream)`:

| Argument | Stored shape | Dtype | Layout |
| --- | --- | --- | --- |
| A | `(M, K/2)` | uint8, two FP4 values per byte | Row-major |
| B | `(K/2, N)` | uint8, two FP4 values per byte | Column-major |
| sfa | `(M, K/32)` | uint8, E8M0 bits | Row-major |
| sfb | `(K/32, N)` | uint8, E8M0 bits | Column-major |
| C | `(M, N)` | BF16 | Row-major |

Even K values occupy the low nibble; odd K values occupy the high nibble.
E2M1 magnitudes are `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` with a sign bit.
Each E8M0 byte represents `2^(bits - 127)` for the finite scales used here.

`v1.py` exports `mxfp4 = MxFp4Gemm()`. Each block computes a 16×16 output tile,
cooperatively loading packed operands and scales into shared memory in chunks
of 64 K values. Threads synchronize before reading each tile and before
overwriting it. `naive.py` reads directly from global memory. Both versions use
one thread per output element and FP32 accumulation; neither uses tensor cores.

```bash
uv run modal run mxfp4_gemm/benchmark.py v1.py
uv run modal run mxfp4_gemm/benchmark.py naive.py
```

Add `--shape 128` to select one square size, or `--check` for correctness only.
Allocation, compilation, and CUDA graph capture live in `benchmark.py`.

## Baseline

The single baseline is **FlashInfer/CuTe MXFP4**, using
[`mm_fp4`](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html)
with `backend="cute-dsl"`, `block_size=32`, and `use_nvfp4=False`. It uses native
block-scaled tensor-core GEMM, with FP32 accumulation and BF16 output.
FlashInfer's `cutlass` backend in the pinned version does not support MXFP4.

Both implementations receive the same packed A and B. The benchmark rearranges
the scale bytes into the baseline's 128x4 layout before timing. Quantization,
scale rearrangement, compilation, and baseline autotuning are excluded.
Both outputs are checked against an independent FP32 multiplication of the
decoded operands.

Timing uses CUDA graphs with L2 flushed before each timed region: median of
50 iterations after 10 warmups. Square sizes 128 through 8192 are benchmarked.
The table reports latency, TFLOPS for both implementations, and speedup.

## Persistent baseline cache

The Modal Volume `learn-kernels-flashinfer-cache` is mounted at
`/root/.cache/flashinfer`. It saves FlashInfer's compiled kernels and a tuning
JSON file for each square size. The first run compiles and tunes; subsequent
runs reuse compatible cached results, even in a new Modal container.

Each shape's cache is committed before timing. New shapes or incompatible
library/compiler versions can require compilation or tuning again. Your own
kernel still goes through `cute.compile`, so edits to `v1.py` take effect.

Run the same benchmark command as usual; no cache flag is needed.
