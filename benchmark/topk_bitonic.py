import torch

from benchmark.common import generic_evaluate


DEPS = ()
SHAPES = [
    {"batch": 1, "n": 32768, "k": 32},
    {"batch": 1, "n": 131072, "k": 64},
    {"batch": 8, "n": 131072, "k": 64},
    {"batch": 32, "n": 131072, "k": 64},
    {"batch": 8, "n": 262144, "k": 64},
]


def _opponent(model, inputs, shape, source_dir):
    return "torch.topk", lambda: torch.topk(inputs[0], shape["k"]), True


def evaluate(workload=None):
    generic_evaluate("topk_bitonic", SHAPES, _opponent)
