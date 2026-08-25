# Milestone 2: FP16 GEMM

The solution ladder is:

- `naive.py`: one output per thread with 16x16 shared-memory tiles
- `smem_tiled.py`: register tiling and vectorized shared-memory staging
- `tma_pipeline.py`: TMA operand staging
- `tcgen.py`: Blackwell `tcgen05` tensor-core instructions

Each solution is an import-only kernel module. Modal configuration, B200
selection, shapes, timing, correctness, SOL, and the PyTorch/cuBLASLt opponent
live in `benchmark/`.

Run any stage from the repository root:

```bash
uv run modal run mile2/naive.py
uv run modal run mile2/smem_tiled.py
uv run modal run mile2/tma_pipeline.py
uv run modal run mile2/tcgen.py
```

Every suite contains four representative shapes. The current `tcgen05` lesson
supports one 64-element K tile and one 128-column N tile, so its benchmark
varies M while preserving that valid input contract. Extending K/N belongs to
the next descriptor, accumulator, and epilogue pipeline milestone.

`test.py` is the NVIDIA-derived minimal `tcgen05` reference.
