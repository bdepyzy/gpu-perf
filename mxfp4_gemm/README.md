`uv run modal run mxfp4_gemm/benchmark.py v3.py`

- v1: ~0.60× baseline geomean.
- v2: warp specialization, single-CTA MMA; 0.88× FlashInfer/CuTe geomean.
- v3: 0.83× baseline geomean.
