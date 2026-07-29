"""Naive KDA forward (Kimi Delta Attention, chunked form): reference math in torch fp32, every plain contraction through one naive CuTeDSL batched GEMM."""
import torch
import torch.nn as nn
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _bmm_kernel(gA, gB, gC):
    tidx, _, _ = cute.arch.thread_idx(); bidx, _, _ = cute.arch.block_idx(); bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx
    z_dim, m_dim, n_dim = gC.shape
    k_dim = gA.shape[2]
    if idx < z_dim * m_dim * n_dim:
        ni = idx % n_dim; mi = (idx // n_dim) % m_dim; zi = idx // (n_dim * m_dim); acc = cutlass.Float32(0.0)
        for ki in range(k_dim): acc += gA[zi, mi, ki] * gB[zi, ki, ni]
        gC[zi, mi, ni] = acc


@cute.jit
def _bmm(mA, mB, mC):
    z_dim, m_dim, n_dim = mC.shape
    _bmm_kernel(mA, mB, mC).launch(grid=(cute.ceil_div(z_dim * m_dim * n_dim, 256), 1, 1), block=(256, 1, 1))


_compiled_bmm = {}


def _cute_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.contiguous(); b = b.contiguous()
    z, m, k = a.shape; n = b.shape[2]
    c = torch.empty(z, m, n, device=a.device, dtype=torch.float32)
    a_ = from_dlpack(a, assumed_align=16); b_ = from_dlpack(b, assumed_align=16); c_ = from_dlpack(c, assumed_align=16)
    if (z, m, n, k) not in _compiled_bmm: _compiled_bmm[(z, m, n, k)] = cute.compile(_bmm, a_, b_, c_)
    _compiled_bmm[(z, m, n, k)](a_, b_, c_)
    return c


def _kda_forward(q, k, v, g, beta, scale: float, chunk: int) -> torch.Tensor:
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]; BT = chunk; NT = T // BT
    q, k, v, g, beta = (x.to(torch.float32) for x in (q, k, v, g, beta))
    q = q * scale
    q = q.view(B, NT, BT, H, K).permute(0, 3, 1, 2, 4); k = k.view(B, NT, BT, H, K).permute(0, 3, 1, 2, 4); v = v.view(B, NT, BT, H, V).permute(0, 3, 1, 2, 4); g = g.view(B, NT, BT, H, K).permute(0, 3, 1, 2, 4); beta = beta.view(B, NT, BT, H).permute(0, 3, 1, 2)
    g = g.cumsum(-2)
    mask_diag_upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    A = torch.zeros(*q.shape[:-1], BT, dtype=torch.float32, device=q.device)
    for i in range(BT): A[..., i] = (k * (g - g[..., i:i + 1, :]).exp() * k[..., i, :].unsqueeze(-2)).sum(-1)
    A = A * beta[..., None]
    A = -A.masked_fill(mask_diag_upper, 0)
    for i in range(1, BT): A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float32, device=q.device)) * beta[..., None, :]
    Z = B * H * NT
    A3 = A.reshape(Z, BT, BT)
    w = _cute_bmm(A3, (g.exp() * k).contiguous().reshape(Z, BT, K)).view(B, H, NT, BT, K)
    u = _cute_bmm(A3, v.contiguous().reshape(Z, BT, V)).view(B, H, NT, BT, V)
    ZB = B * H
    S = q.new_zeros(B, H, K, V)
    o = torch.zeros_like(v)
    mask_strict_upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(NT):
        q_i = q[:, :, i]; k_i = k[:, :, i]; g_i = g[:, :, i]; w_i = w[:, :, i]; u_i = u[:, :, i]
        Aqk = torch.zeros(B, H, BT, BT, dtype=torch.float32, device=q.device)
        for j in range(BT): Aqk[..., j] = (q_i * (g_i - g_i[:, :, j:j + 1, :]).exp() * k_i[:, :, j, :].unsqueeze(-2)).sum(-1)
        Aqk = Aqk.masked_fill(mask_strict_upper, 0)
        v_i = u_i - _cute_bmm(w_i.reshape(ZB, BT, K), S.reshape(ZB, K, V)).view(B, H, BT, V)
        o[:, :, i] = _cute_bmm((q_i * g_i.exp()).reshape(ZB, BT, K), S.reshape(ZB, K, V)).view(B, H, BT, V) + _cute_bmm(Aqk.reshape(ZB, BT, BT), v_i.reshape(ZB, BT, V)).view(B, H, BT, V)
        S = S * g_i[:, :, -1].exp().unsqueeze(-1)
        S = S + _cute_bmm(((g_i[:, :, -1:] - g_i).exp() * k_i).transpose(-1, -2).reshape(ZB, K, BT), v_i.reshape(ZB, BT, V)).view(B, H, K, V)
    return o.permute(0, 2, 3, 1, 4).reshape(B, T, H, V).to(dtype)


class Model(nn.Module):
    """KDA forward (chunk form). No learned parameters; all inputs are activations."""

    def __init__(self, B: int, T: int, H: int, K: int, V: int, chunk_size: int = 64):
        super().__init__(); self.B, self.T, self.H, self.K, self.V = B, T, H, K, V
        self.chunk_size = chunk_size; self.scale = float(K) ** -0.5
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return _kda_forward(q, k, v, g, beta, scale=self.scale, chunk=self.chunk_size)
