import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, batch: int, n: int, k: int):
        super().__init__()
        self.batch, self.n, self.k = batch, n, k

    def forward(self, x: torch.Tensor):
        values, indices = torch.topk(x, k=self.k, dim=-1, largest=True, sorted=True)
        return values, indices


def get_inputs(batch, n, k):
    x = torch.randn(batch, n, dtype=torch.float32)
    return [x]
