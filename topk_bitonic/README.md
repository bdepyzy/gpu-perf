`uv run modal run topk_bitonic/benchmark.py v2.py`

- v1: blockwise bitonic sort, candidate merge, input preserved. 0.27× fastest PyTorch/FlashInfer baseline geomean.
- v2: register/shuffle sorting, sorted-run merge. 2.94× v1 geomean on B200 ([results](v2_results.json)).
