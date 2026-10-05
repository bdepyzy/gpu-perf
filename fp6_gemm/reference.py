import torch
import torch.nn as nn


LEVELS = (
    0.0, 0.0625, 0.125, 0.1875,
    0.25, 0.3125, 0.375, 0.4375,
    0.5, 0.625, 0.75, 0.875,
    1.0, 1.25, 1.5, 1.75,
    2.0, 2.5, 3.0, 3.5,
    4.0, 5.0, 6.0, 7.0,
    8.0, 10.0, 12.0, 14.0,
    16.0, 20.0, 24.0, 28.0,
)


def _pack_fp6(codes: torch.Tensor) -> torch.Tensor:
    K, N = codes.shape
    assert K % 4 == 0
    c = codes.view(K // 4, 4, N)
    b0 = c[:, 0] | ((c[:, 1] & 0x3) << 6)
    b1 = (c[:, 1] >> 2) | ((c[:, 2] & 0xF) << 4)
    b2 = (c[:, 2] >> 4) | (c[:, 3] << 2)
    return torch.stack([b0, b1, b2], dim=1).reshape(K // 4 * 3, N).contiguous()


def _unpack_fp6(w_packed: torch.Tensor, K: int) -> torch.Tensor:
    N = w_packed.shape[1]
    b = w_packed.view(K // 4, 3, N)
    c0 = b[:, 0] & 0x3F
    c1 = (b[:, 0] >> 6) | ((b[:, 1] & 0xF) << 2)
    c2 = (b[:, 1] >> 4) | ((b[:, 2] & 0x3) << 4)
    c3 = b[:, 2] >> 2
    return torch.stack([c0, c1, c2, c3], dim=1).reshape(K, N)


class Model(nn.Module):
    def __init__(self, M: int, N: int, K: int):
        super().__init__()
        assert K % 4 == 0
        self.M, self.N, self.K = M, N, K

        torch.manual_seed(0x6 ^ (M * 1315423911 + N * 2654435761 + K))
        w_full = torch.randn(K, N, dtype=torch.float32).clamp(-28.0, 28.0)

        levels = torch.tensor(LEVELS)
        idx = torch.bucketize(w_full.abs(), (levels[:-1] + levels[1:]) * 0.5)
        codes = ((w_full < 0).to(torch.uint8) << 5) | idx.to(torch.uint8)

        self.register_buffer("w_q", _pack_fp6(codes))

    def dequantize(self):
        codes = _unpack_fp6(self.w_q, self.K)
        levels = torch.tensor(LEVELS, device=codes.device)
        mag = levels[(codes & 31).long()]
        w = torch.where(codes & 32 != 0, -mag, mag)
        return w.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.bfloat16) @ self.dequantize()


def get_inputs(M, K):
    x = torch.randn(M, K, dtype=torch.bfloat16)
    return [x]
