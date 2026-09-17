import torch
import torch.nn as nn


class Model(nn.Module):
    """y = (x @ w.T).to(bf16), where x is fp8_e4m3 (M, K), w is fp8_e4m3 (N, K)."""

    def __init__(self, M: int, N: int, K: int):
        super().__init__()
        self.M, self.N, self.K = M, N, K
        w = torch.empty(N, K, dtype=torch.bfloat16)
        nn.init.normal_(w, std=0.02)
        self.register_buffer("weight", w.to(torch.float8_e4m3fn))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to bf16 for the naive reference; the kernel equivalent would
        # use mma.sync f8f6f4 kind directly.
        x_bf = x.to(torch.bfloat16)
        w_bf = self.weight.to(torch.bfloat16)
        return x_bf @ w_bf.T  # (M, N) bf16


def get_inputs(M, K):
    # fp8_e4m3 input; random uniform in [-4, 4] then cast.
    x = (torch.rand(M, K) * 8 - 4).to(torch.float8_e4m3fn)
    return [x]
