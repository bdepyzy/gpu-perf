import importlib
import inspect
import time
from pathlib import Path

import modal

from benchmark import common
from benchmark.image import B200_GPU, B200_TIMEOUT, b200_base_image


BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parent
REMOTE_ROOT = Path("/workspace")

MODULES = {
    "fp8_gemm": "benchmark.fp8_gemm",
    "kda": "benchmark.kda",
    "paged_attention": "benchmark.paged_attention",
    "topk_bitonic": "benchmark.topk_bitonic",
    "w4a16_gemm": "benchmark.w4a16_gemm",
    "mile1_naive": "benchmark.mile1",
    "mile1_vector": "benchmark.mile1",
    "mile1_transpose": "benchmark.mile1",
    "mile2_naive": "benchmark.mile2",
    "mile2_smem_tiled": "benchmark.mile2",
    "mile2_tma_pipeline": "benchmark.mile2",
    "mile2_tcgen": "benchmark.mile2",
}


def _importing_workload():
    for frame in inspect.stack():
        path = Path(frame.filename).resolve()
        if path.name == "kernel.py" and path.parent.name in MODULES:
            return path.parent.name
        candidate = f"{path.parent.name}_{path.stem}"
        if candidate in MODULES:
            return candidate
    return None


WORKLOAD = _importing_workload()
benchmark_module = importlib.import_module(MODULES[WORKLOAD]) if WORKLOAD else None

image = b200_base_image
if benchmark_module is not None and benchmark_module.DEPS:
    image = image.uv_pip_install(*benchmark_module.DEPS)
if WORKLOAD is not None:
    solution_dir = WORKLOAD.split("_", 1)[0] if WORKLOAD.startswith("mile") else WORKLOAD
    image = image.add_local_dir(ROOT / solution_dir, REMOTE_ROOT / solution_dir)

app = modal.App(f"bench-{WORKLOAD or 'workload'}", image=image)


@app.function(gpu=B200_GPU, timeout=B200_TIMEOUT)
def run(workload: str, check: bool = False):
    common.configure_root(REMOTE_ROOT)
    if check:
        common.set_check_only(True)
    module = importlib.import_module(MODULES[workload])
    module.evaluate(workload)


@app.local_entrypoint()
def main(check: bool = False):
    if WORKLOAD is None:
        raise RuntimeError("import benchmark.bench.app from a supported solution file")
    run.remote(WORKLOAD, check)
    time.sleep(0.25)
