from benchmark.common import GEMM_SHAPES, generic_evaluate, load


DEPS = ("bitsandbytes",)
SHAPES = GEMM_SHAPES


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
