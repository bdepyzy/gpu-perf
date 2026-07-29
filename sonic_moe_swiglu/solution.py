"""Naive CuTeDSL grouped-GEMM + SwiGLU: per expert two naive GEMMs (one thread per output, K-loop, fp32 accum), then torch silu * up."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _gemm_kernel(gA, gB, gC):
    tidx, _, _ = cute.arch.thread_idx(); bidx, _, _ = cute.arch.block_idx(); bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx
    m_dim, n_dim = gC.shape
    k_dim = gA.shape[1]
    if idx < m_dim * n_dim:
        mi = idx // n_dim; ni = idx % n_dim; acc = cutlass.Float32(0.0)
        for ki in range(k_dim): acc += cutlass.Float32(gA[mi, ki]) * cutlass.Float32(gB[ki, ni])
        gC[mi, ni] = gC.element_type(acc)


@cute.jit
def _gemm(mA, mB, mC):
    m_dim, n_dim = mC.shape
    _gemm_kernel(mA, mB, mC).launch(grid=(cute.ceil_div(m_dim * n_dim, 256), 1, 1), block=(256, 1, 1))


class Model(nn.Module):
    """h_e = silu(x_e @ W_gate[e]) * (x_e @ W_up[e]) per expert row-slice."""

    def __init__(self, T_total: int, H: int, I: int, E: int, K: int):  # noqa: E741
        super().__init__(); self.T_total, self.H, self.I, self.E, self.K = T_total, H, I, E, K
        self.W_gate = nn.Parameter(torch.empty(E, H, I, dtype=torch.bfloat16)); nn.init.normal_(self.W_gate, std=0.02)
        self.W_up = nn.Parameter(torch.empty(E, H, I, dtype=torch.bfloat16)); nn.init.normal_(self.W_up, std=0.02)
        self._compiled = {}

    def _gemm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.contiguous(); b = b.contiguous()
        m, k = a.shape; n = b.shape[1]
        c = torch.empty(m, n, device=a.device, dtype=torch.bfloat16)
        a_ = from_dlpack(a, assumed_align=16); b_ = from_dlpack(b, assumed_align=16); c_ = from_dlpack(c, assumed_align=16)
        if (m, n, k) not in self._compiled: self._compiled[(m, n, k)] = cute.compile(_gemm, a_, b_, c_)
        self._compiled[(m, n, k)](a_, b_, c_)
        return c

    def forward(self, hidden_states: torch.Tensor, expert_offsets: torch.Tensor) -> torch.Tensor:
        T_perm, _ = hidden_states.shape
        out = torch.empty(T_perm, self.I, dtype=torch.bfloat16, device=hidden_states.device)
        for e in range(self.E):
            start = int(expert_offsets[e].item()); end = int(expert_offsets[e + 1].item())
            if end == start: continue
            x_e = hidden_states[start:end]
            gate = self._gemm(x_e, self.W_gate[e].detach()); up = self._gemm(x_e, self.W_up[e].detach())
            out[start:end] = F.silu(gate) * up
        return out
