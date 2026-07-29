# learn-kernels

fp8_gemm

kda_cutlass

paged_attention

topk_bitonic

sonic_moe_swiglu

w4a16_gemm

## Usage

Each problem folder has a Modal harness (`bench.py`) that runs the repo-root
`bench.py` checker/evaluator on a B200 against `KernelBench-Hard` references.
Write your kernel as `solution.py` in the problem folder (same `Model` API as
the problem's `reference.py`), then from the repo root:

```bash
modal run fp8_gemm/bench.py                      # check + bench
modal run fp8_gemm/bench.py --command check
modal run fp8_gemm/bench.py --command bench --iterations 200
```
