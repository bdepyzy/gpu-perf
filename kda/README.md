`uv run modal run kda/benchmark.py v2.py`

- v1: register-held state, warp reductions, value-column tiling. 0.20× FLA chunk_kda geomean.
- v2: staged inputs, 8-lane reductions, one value column per thread. 2.31× v1 geomean on B200 ([results](v2_results.json)).
