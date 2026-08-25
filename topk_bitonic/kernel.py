from benchmark.bench import app
import torch
import torch.nn as nn
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _topk_kernel(gX, gV, gI):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    row = bidx * bdim + tidx
    b_dim, n_dim = gX.shape
    k_dim = gV.shape[1]
    if row < b_dim:
        prev_v = cutlass.Float32(3.402823466e38)
        prev_i = cutlass.Int32(-1)
        for t in range(k_dim):
            best_v = cutlass.Float32(-3.402823466e38)
            best_i = cutlass.Int32(n_dim)
            for j in range(n_dim):
                v = gX[row, j]
                if ((v < prev_v) | ((v == prev_v) & (j > prev_i))) & ((v > best_v) | ((v == best_v) & (j < best_i))):
                    best_v = v
                    best_i = j
            gV[row, t] = best_v
            gI[row, t] = cutlass.Int64(best_i)
            prev_v = best_v
            prev_i = best_i


@cute.jit
def _topk(mX, mV, mI):
    b_dim, n_dim = mX.shape
    _topk_kernel(mX, mV, mI).launch(grid=(cute.ceil_div(b_dim, 128), 1, 1), block=(128, 1, 1))


class Model(nn.Module):
    """Top-k over the last dim of a (batch, n) fp32 tensor."""

    def __init__(self, batch: int, n: int, k: int):
        super().__init__()
        self.batch, self.n, self.k = batch, n, k
        self.register_buffer("_dummy", torch.zeros(1))
        self._compiled = None

    def forward(self, x: torch.Tensor):
        values = torch.empty(self.batch, self.k, device=x.device, dtype=torch.float32)
        indices = torch.empty(self.batch, self.k, device=x.device, dtype=torch.int64)
        x_ = from_dlpack(x, assumed_align=16)
        v_ = from_dlpack(values, assumed_align=16)
        i_ = from_dlpack(indices, assumed_align=16)
        if self._compiled is None:
            self._compiled = cute.compile(_topk, x_, v_, i_)
        self._compiled(x_, v_, i_)
        return values, indices
