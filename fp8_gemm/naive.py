from benchmark.bench import app
import torch
import torch.nn as nn
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _gemm_kernel(gA, gB, gC):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx
    m_dim, n_dim = gC.shape
    k_dim = gA.shape[1]
    if idx < m_dim * n_dim:
        mi = idx // n_dim
        ni = idx % n_dim
        acc = cutlass.Float32(0.0)
        for ki in range(k_dim):
            acc += cutlass.Float32(gA[mi, ki]) * cutlass.Float32(gB[ni, ki])
        gC[mi, ni] = gC.element_type(acc)


@cute.jit
def _gemm(mA, mB, mC):
    mA = cute.make_tensor(cute.recast_ptr(mA.iterator, dtype=cutlass.Float8E4M3FN), mA.layout)
    mB = cute.make_tensor(cute.recast_ptr(mB.iterator, dtype=cutlass.Float8E4M3FN), mB.layout)
    m_dim, n_dim = mC.shape
    _gemm_kernel(mA, mB, mC).launch(grid=(cute.ceil_div(m_dim * n_dim, 256), 1, 1), block=(256, 1, 1))


class Model(nn.Module):
    """y = (x @ w.T).to(bf16), x fp8_e4m3 (M, K), w fp8_e4m3 (N, K)."""
    def __init__(self, M: int, N: int, K: int):
        super().__init__()
        self.M, self.N, self.K = M, N, K
        self.weight = nn.Parameter(torch.empty(N, K, dtype=torch.float8_e4m3fn))
        self._compiled = None

    def _callables(self, x: torch.Tensor):
        c = torch.empty(self.M, self.N, device=x.device, dtype=torch.bfloat16)
        a_ = from_dlpack(x.view(torch.uint8), assumed_align=16)
        b_ = from_dlpack(self.weight.detach().view(torch.uint8), assumed_align=16)
        c_ = from_dlpack(c, assumed_align=16)
        if self._compiled is None:
            self._compiled = cute.compile(_gemm, a_, b_, c_)
        return self._compiled, (a_, b_, c_), c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compiled, args, c = self._callables(x)
        compiled(*args)
        return c

    def prepare_for_bench(self, inputs):
        compiled, args, _ = self._callables(inputs[0])
        return lambda: compiled(*args)
