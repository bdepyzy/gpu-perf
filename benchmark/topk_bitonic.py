import torch

from benchmark.common import generic_evaluate


DEPS = ()
SHAPES = [
    {"batch": 1, "n": 32768, "k": 32},
    {"batch": 32, "n": 4096, "k": 8},
    {"batch": 16, "n": 8192, "k": 16},
    {"batch": 8, "n": 6000, "k": 16},
    {"batch": 64, "n": 2048, "k": 1},
]


def _opponent(model, inputs, shape, source_dir):
    return "torch.topk", lambda: torch.topk(inputs[0], shape["k"]), True


def evaluate(workload=None):
    generic_evaluate("topk_bitonic", SHAPES, _opponent)
