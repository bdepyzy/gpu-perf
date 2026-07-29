"""Naive CuTeDSL paged-attention decode: one block per (batch, head), head_dim threads; serial smem reduction for q.k, online softmax in smem, exp as exp2(x * log2e)."""
import torch
import torch.nn as nn
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator

LOG2E = 1.4426950408889634


@cute.kernel
def _attn_kernel(gQ, gKV, gBT, gSL, gO, scale: cutlass.Float32):
    tidx, _, _ = cute.arch.thread_idx(); bidx, _, _ = cute.arch.block_idx()
    b_dim, h_dim, d_dim = gQ.shape
    p_dim = gKV.shape[1]; hkv_dim = gKV.shape[2]; g_dim = h_dim // hkv_dim
    b = bidx // h_dim; h = bidx % h_dim; hkv = h // g_dim
    smem = SmemAllocator()
    s_q = smem.allocate_tensor(cutlass.Float32, cute.make_layout(d_dim), byte_alignment=16); s_acc = smem.allocate_tensor(cutlass.Float32, cute.make_layout(d_dim), byte_alignment=16); s_red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(d_dim), byte_alignment=16); s_stat = smem.allocate_tensor(cutlass.Float32, cute.make_layout(4), byte_alignment=16)
    s_q[tidx] = cutlass.Float32(gQ[b, h, tidx]); s_acc[tidx] = cutlass.Float32(0.0)
    if tidx == 0: s_stat[0] = cutlass.Float32(-3.402823466e38); s_stat[1] = cutlass.Float32(0.0); s_stat[2] = cutlass.Float32(0.0); s_stat[3] = cutlass.Float32(1.0)
    cute.arch.sync_threads()
    for l in range(gSL[b]):
        page = gBT[b, l // p_dim]; slot = l % p_dim
        s_red[tidx] = s_q[tidx] * cutlass.Float32(gKV[page, slot, hkv, tidx]); cute.arch.sync_threads()
        if tidx == 0:
            total = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(d_dim): total += s_red[i]
            score = total * scale
            m_old = s_stat[0]; m_new = m_old
            if score > m_old: m_new = score
            corr = cute.math.exp2((m_old - m_new) * LOG2E); p = cute.math.exp2((score - m_new) * LOG2E)
            s_stat[0] = m_new; s_stat[1] = s_stat[1] * corr + p; s_stat[2] = p; s_stat[3] = corr
        cute.arch.sync_threads()
        s_acc[tidx] = s_acc[tidx] * s_stat[3] + s_stat[2] * cutlass.Float32(gKV[page, slot, hkv, d_dim + tidx]); cute.arch.sync_threads()
    if tidx == 0: s_stat[1] = cutlass.Float32(1.0) / s_stat[1]
    cute.arch.sync_threads()
    gO[b, h, tidx] = gO.element_type(s_acc[tidx] * s_stat[1])


@cute.jit
def _attn(mQ, mKV, mBT, mSL, mO, scale: cutlass.Float32):
    b_dim, h_dim, d_dim = mQ.shape
    _attn_kernel(mQ, mKV, mBT, mSL, mO, scale).launch(grid=(b_dim * h_dim, 1, 1), block=(d_dim, 1, 1))


class Model(nn.Module):
    """Single-query paged attention decode over a packed [K | V] page pool."""

    def __init__(self, batch: int, num_heads: int, num_kv_heads: int, head_dim: int, seq_len: int, page_size: int):
        super().__init__(); assert num_heads % num_kv_heads == 0
        self.batch, self.num_heads, self.num_kv_heads, self.head_dim, self.seq_len, self.page_size = batch, num_heads, num_kv_heads, head_dim, seq_len, page_size
        self.group_size = num_heads // num_kv_heads; self.scale = 1.0 / float(head_dim) ** 0.5
        self.register_buffer("_dummy", torch.zeros(1, dtype=torch.bfloat16), persistent=False); self._compiled = None

    def forward(self, query: torch.Tensor, kv_cache: torch.Tensor, block_table: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        B, H, D = query.shape
        out = torch.empty(B, H, D, dtype=query.dtype, device=query.device)
        q_ = from_dlpack(query, assumed_align=16); kv_ = from_dlpack(kv_cache, assumed_align=16); bt_ = from_dlpack(block_table, assumed_align=16); sl_ = from_dlpack(seq_lens, assumed_align=16); o_ = from_dlpack(out, assumed_align=16)
        scale = cutlass.Float32(self.scale)
        if self._compiled is None: self._compiled = cute.compile(_attn, q_, kv_, bt_, sl_, o_, scale)
        self._compiled(q_, kv_, bt_, sl_, o_, scale)
        return out
