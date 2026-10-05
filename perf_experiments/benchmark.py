import csv
import hashlib
import io
import json
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.13")
    .add_local_file(HERE / "microbench.cu", "/src/microbench.cu", copy=True)
    .run_commands(
        "mkdir -p /opt/experiments",
        "nvcc -O3 -std=c++17 -arch=sm_100 -rdc=true -lineinfo "
        "-Xptxas=-v /src/microbench.cu -lcudadevrt "
        "-o /opt/experiments/microbench > /opt/experiments/build.log 2>&1 "
        "|| { cat /opt/experiments/build.log; exit 1; }",
        "cuobjdump --dump-sass /opt/experiments/microbench > /opt/experiments/sass.txt",
        "cuobjdump --dump-resource-usage /opt/experiments/microbench > /opt/experiments/resources.txt",
    )
)
app = modal.App("b200-performance-experiments", image=image)


def capture(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


@app.function(gpu="B200", timeout=3600)
def run(suite: str, quick: bool, samples: int):
    metadata = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "suite": suite, "quick": quick, "samples": samples,
        "source_sha256": hashlib.sha256(Path("/src/microbench.cu").read_bytes()).hexdigest(),
        "nvcc": capture(["nvcc", "--version"]),
        "gpu_before": capture(["nvidia-smi", "-q"]),
    }
    command = ["/opt/experiments/microbench", suite, str(samples), "1" if quick else "0"]
    print("Running:", " ".join(command), flush=True)

    with Path("/tmp/samples.csv").open("w") as stdout, Path("/tmp/run.log").open("w") as log:
        process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.PIPE, text=True)
        for line in process.stderr:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    metadata["returncode"] = code
    metadata["gpu_after"] = capture(["nvidia-smi", "-q"])
    files = {name: (Path("/opt/experiments") / name).read_text()
             for name in ("build.log", "sass.txt", "resources.txt")}
    files.update({"samples.csv": Path("/tmp/samples.csv").read_text(),
                  "run.log": Path("/tmp/run.log").read_text(),
                  "metadata.json": json.dumps(metadata, indent=2),
                  "microbench.cu": Path("/src/microbench.cu").read_text()})
    return files


def summarize(raw):
    groups = {}
    for row in csv.DictReader(io.StringIO(raw)):
        key = tuple(row[k] for k in ("suite", "case", "metric", "unit", "iterations",
                                    "bytes", "stride", "threads", "nodes"))
        groups.setdefault(key, []).append(float(row["value"]))
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["suite", "case", "metric", "unit", "iterations", "bytes", "stride",
                     "threads", "nodes", "samples", "min", "median", "p90", "max"])
    for key, values in groups.items():
        values.sort()
        writer.writerow([*key, len(values), values[0], statistics.median(values),
                         values[max(0, (9 * len(values) + 9) // 10 - 1)], values[-1]])
    return out.getvalue()


@app.local_entrypoint()
def main(suite: str = "all", quick: bool = False, samples: int = 15, output: str = ""):
    if suite not in {"all", "launch", "sync", "memory"}:
        raise ValueError("suite must be all, launch, sync, or memory")
    if not 3 <= samples <= 1000:
        raise ValueError("samples must be between 3 and 1000")
    destination = Path(output) if output else HERE / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    files = run.remote(suite, quick, samples)
    files["summary.csv"] = summarize(files["samples.csv"])
    for name, contents in files.items():
        (destination / name).write_text(contents)
    print(f"Saved results: {destination.resolve()}")
    print(files["summary.csv"])
    if json.loads(files["metadata.json"])["returncode"]:
        raise RuntimeError(f"Experiment failed; inspect {destination / 'run.log'}")


if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "modal", "run", __file__, *sys.argv[1:]]))
