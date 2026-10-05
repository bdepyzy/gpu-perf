`uv run modal run paged_attention/benchmark.py v2.py`

- v1: split-context decode, online softmax, warp reductions. 0.17× fastest cuDNN/FlashInfer baseline geomean.
- v2: shorter context splits, CTA-local reduction, parallel output merge. 1.83× v1 geomean on B200 ([results](v2_results.json)).
