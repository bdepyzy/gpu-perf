from benchmark.common import generic_evaluate, load


DEPS = ("bitsandbytes",)
SHAPES = [
    {"M": 1, "N": 2048, "K": 1024},
    {"M": 8, "N": 2048, "K": 1024},
    {"M": 32, "N": 2048, "K": 1024},
    {"M": 1, "N": 4096, "K": 2048},
    {"M": 16, "N": 3072, "K": 1024},
]


def _opponent(model, inputs, shape, source_dir):
    sota = load("w4a16_gemm_sota", source_dir / "sota.py")
    if not sota.is_available():
        raise RuntimeError("bitsandbytes NF4 is unavailable")
    return (
        "bitsandbytes NF4 (different quantization)",
        lambda: sota.sota_forward(inputs[0], model),
        False,
    )


def evaluate(workload=None):
    generic_evaluate("w4a16_gemm", SHAPES, _opponent)
