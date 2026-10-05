`uv run perf_experiments/benchmark.py --suite all --quick --samples 3`

- Launch: empty/tiny kernels, host/device launches, CUDA graphs.
- Sync: barriers, shuffles, shared-memory exchange, clusters.
- Memory: dependent scalar/vector loads, cache sizes, strides.
- Results: CSV timings, SASS, GPU/compiler metadata in `results/`.
