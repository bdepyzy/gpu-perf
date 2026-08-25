import torch

from benchmark.common import generic_evaluate


DEPS = ()
SHAPES = [
    {"M": 128, "N": 128, "K": 128},
    {"M": 256, "N": 256, "K": 256},
    {"M": 512, "N": 512, "K": 512},
    {"M": 32, "N": 1024, "K": 512},
    {"M": 256, "N": 768, "K": 255},
]


def _opponent(model, inputs, shape, source_dir):
    x = inputs[0].to(torch.bfloat16)
    weight = model.weight.detach()
    return "PyTorch/cuBLASLt BF16", lambda: x @ weight.T, True


def evaluate(workload=None):
    generic_evaluate("fp8_gemm", SHAPES, _opponent)
