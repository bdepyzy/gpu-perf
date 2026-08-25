from benchmark.common import generic_evaluate, load


DEPS = ("flash-linear-attention",)
SHAPES = [
    {"B": 1, "T": 512, "H": 16, "K": 128, "V": 128, "CHUNK_SIZE": 64},
    {"B": 4, "T": 1024, "H": 32, "K": 128, "V": 128, "CHUNK_SIZE": 64},
    {"B": 1, "T": 2048, "H": 32, "K": 128, "V": 128, "CHUNK_SIZE": 64},
]


def _opponent(model, inputs, shape, source_dir):
    sota = load("kda_sota", source_dir / "sota.py")
    if not sota.is_available():
        raise RuntimeError("FLA FlashKDA is unavailable")
    return "FLA FlashKDA", lambda: sota.sota_forward(*inputs, scale=model.scale), True


def evaluate(workload=None):
    generic_evaluate("kda", SHAPES, _opponent)
