`uv run modal run fp6_gemm/benchmark.py v2.py`

- v1: fused E3M2 unpacking, shared-memory tiles, BF16 tcgen05 MMA. 0.061× cuBLAS BF16 geomean (pre-dequantized baseline).
- v2: 128-column tiles, bitwise E3M2 decode, vector stores. 1.16× v1 geomean on B200 ([results](v2_results.json)).
