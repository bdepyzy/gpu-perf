import torch
import torch.nn as nn


BLOCK = 32  # elements per E8M0 scale block
LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # non-negative e2m1 values; max 6


def _pack_fp4(w_q: torch.Tensor) -> torch.Tensor:
    """Pack (K, N) uint8 e2m1 codes in [0,15] into (K//2, N) uint8.

    Even rows go in the low nibble, odd rows in the high nibble.
    """
    K, N = w_q.shape
    assert K % 2 == 0
    lo = w_q[0::2].to(torch.uint8) & 0xF
    hi = w_q[1::2].to(torch.uint8) & 0xF
    return (lo | (hi << 4)).contiguous()


def _unpack_fp4(w_packed: torch.Tensor) -> torch.Tensor:
    """Unpack (K//2, N) uint8 -> (K, N) uint8 codes in [0,15]."""
    lo = w_packed & 0xF
    hi = w_packed >> 4
    return torch.stack([lo, hi], dim=1).reshape(-1, w_packed.shape[1])


def _e2m1_decode(codes: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    mag = levels[(codes & 7).long()]
    return torch.where(codes & 8 != 0, -mag, mag)


class Model(nn.Module):
    """MXFP4 GEMM: y = x @ dequant(w_fp4, scales). Activations BF16.

    Weights are e2m1 (1 sign + 2 exp + 1 mantissa), packed two per byte along K,
    with one E8M0 (power-of-two) scale per block of 32 K values.
    """

    def __init__(self, M: int, N: int, K: int):
        super().__init__()
        assert K % BLOCK == 0 and K % 2 == 0
        self.M, self.N, self.K = M, N, K
        n_blocks = K // BLOCK

        torch.manual_seed(0xF4 ^ (M * 1315423911 + N * 2654435761 + K))
        w_full = torch.randn(K, N, dtype=torch.float32) * 0.02

        w_b = w_full.view(n_blocks, BLOCK, N)
        amax = w_b.abs().amax(dim=1, keepdim=True)  # (n_blocks, 1, N)
        # Round scales down to powers of two; values above 6 * scale saturate.
        e = torch.floor(torch.log2(amax.clamp_min(1e-12) / 6.0)).clamp(-126, 127)
        scales_f = torch.exp2(e)

        levels = torch.tensor(LEVELS)
        idx = (w_b.abs() / scales_f).unsqueeze(-1).sub(levels).abs().argmin(-1)  # nearest level
        codes = ((w_b < 0).to(torch.uint8) << 3) | idx.to(torch.uint8)  # (n_blocks, 32, N)

        self.register_buffer("w_q", _pack_fp4(codes.reshape(K, N)))  # (K//2, N) uint8
        self.register_buffer("w_scales", (e + 127).to(torch.uint8).view(n_blocks, N))  # E8M0 bits

    def dequantize(self):
        codes = _unpack_fp4(self.w_q)  # (K, N)
        levels = torch.tensor(LEVELS, device=codes.device)
        w = _e2m1_decode(codes, levels)
        s = torch.exp2(self.w_scales.float() - 127).repeat_interleave(BLOCK, dim=0)
        return (w * s).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.bfloat16) @ self.dequantize()


def get_inputs(M, K):
    x = torch.randn(M, K, dtype=torch.bfloat16)
    return [x]
