"""Modal B200 harness for sonic_moe_swiglu: write solution.py here, then `modal run sonic_moe_swiglu/bench.py --command check|bench|all [--warmup N --iterations N --one-seed]`."""
from pathlib import Path

import modal

PROBLEM = "sonic_moe_swiglu"
REPO_ROOT = Path(__file__).resolve().parent.parent
image = modal.Image.debian_slim(python_version="3.13").uv_pip_install("torch==2.11.0", "cutlass", "nvidia-cutlass", "nvidia-cutlass-dsl", "einops").add_local_file(REPO_ROOT / "bench.py", "/root/harness.py").add_local_dir(REPO_ROOT / "KernelBench-Hard" / "problems", "/root/KernelBench-Hard/problems").add_local_dir(REPO_ROOT / PROBLEM, f"/root/{PROBLEM}")
app = modal.App(f"kbh-{PROBLEM}", image=image)


@app.function(gpu="B200", timeout=30 * 60)
def run(command: str = "all", warmup: int = 10, iterations: int = 100, one_seed: bool = False):
    import importlib.util; spec = importlib.util.spec_from_file_location("kbh_harness", "/root/harness.py"); bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench); command in ("check", "all") and bench.check(PROBLEM, (42,) if one_seed else (42, 123, 456)); command in ("bench", "all") and bench.evaluate(PROBLEM, warmup, iterations)


@app.local_entrypoint()
def main(command: str = "all", warmup: int = 10, iterations: int = 100, one_seed: bool = False): run.remote(command, warmup, iterations, one_seed)
