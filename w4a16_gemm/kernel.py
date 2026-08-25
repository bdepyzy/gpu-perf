from benchmark.bench import app
import torch
import torch.nn as nn
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

GROUP_SIZE = 128


@cute.kernel
def _gemm_kernel(gX, gW, gS, gZ, gC, gs: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx
    m_dim, n_dim = gC.shape
    kh_dim = gW.shape[0]
    if idx < m_dim * n_dim:
        mi = idx // n_dim
        ni = idx % n_dim
        acc = cutlass.Float32(0.0)
        for k2 in range(kh_dim):
            byte = cutlass.Int32(gW[k2, ni])
            ke = 2 * k2
            ko = 2 * k2 + 1
            ge = ke // gs
            go = ko // gs
            acc += cutlass.Float32(gX[mi, ke]) * ((cutlass.Float32(byte & 15) - cutlass.Float32(gZ[ge, ni])) * cutlass.Float32(gS[ge, ni]))
            acc += cutlass.Float32(gX[mi, ko]) * ((cutlass.Float32(byte >> 4) - cutlass.Float32(gZ[go, ni])) * cutlass.Float32(gS[go, ni]))
        gC[mi, ni] = gC.element_type(acc)


@cute.jit
def _gemm(mX, mW, mS, mZ, mC, gs: cutlass.Int32):
    m_dim, n_dim = mC.shape
    _gemm_kernel(mX, mW, mS, mZ, mC, gs).launch(grid=(cute.ceil_div(m_dim * n_dim, 256), 1, 1), block=(256, 1, 1))


class Model(nn.Module):
    """y = x @ dequant(w_q, scales, zeros); buffers carried over from the reference state_dict."""

    def __init__(self, M: int, N: int, K: int, group_size: int = GROUP_SIZE):
        super().__init__()
        assert K % group_size == 0 and K % 2 == 0
        self.M, self.N, self.K = M, N, K
        self.group_size = group_size
        self.register_buffer("w_q", torch.empty(K // 2, N, dtype=torch.uint8))
        self.register_buffer("scales", torch.empty(K // group_size, N, dtype=torch.bfloat16))
        self.register_buffer("zeros", torch.empty(K // group_size, N, dtype=torch.bfloat16))
        self._compiled = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.empty(self.M, self.N, device=x.device, dtype=torch.bfloat16)
        x_ = from_dlpack(x, assumed_align=16)
        w_ = from_dlpack(self.w_q, assumed_align=16)
        s_ = from_dlpack(self.scales, assumed_align=16)
        z_ = from_dlpack(self.zeros, assumed_align=16)
        c_ = from_dlpack(c, assumed_align=16)
        gs = cutlass.Int32(self.group_size)
        if self._compiled is None:
            self._compiled = cute.compile(_gemm, x_, w_, s_, z_, c_, gs)
        self._compiled(x_, w_, s_, z_, c_, gs)
        return c

    def prepare_for_bench(self, inputs):
        c = torch.empty(self.M, self.N, device="cuda", dtype=torch.bfloat16)
        args = (
            from_dlpack(inputs[0], assumed_align=16),
            from_dlpack(self.w_q, assumed_align=16),
            from_dlpack(self.scales, assumed_align=16),
            from_dlpack(self.zeros, assumed_align=16),
            from_dlpack(c, assumed_align=16),
            cutlass.Int32(self.group_size),
        )
        if self._compiled is None:
            self._compiled = cute.compile(_gemm, *args)
        return lambda: self._compiled(*args)
