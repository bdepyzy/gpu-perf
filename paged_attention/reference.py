import math

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        batch: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        page_size: int,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads (GQA)"
        self.batch = batch
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.page_size = page_size
        self.group_size = num_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, query: torch.Tensor, kv_cache: torch.Tensor, block_table: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        B, H, D = query.shape
        Hkv = self.num_kv_heads
        G = self.group_size
        P = self.page_size

        out = torch.empty(B, H, D, dtype=query.dtype, device=query.device)

        for b in range(B):
            L = int(seq_lens[b].item())
            num_pages = (L + P - 1) // P
            pages = block_table[b, :num_pages].long()

            kv = kv_cache.index_select(0, pages)
            kv = kv.reshape(num_pages * P, Hkv, 2 * D)
            kv = kv[:L]
            k = kv[..., :D]
            v = kv[..., D:]

            k = k.repeat_interleave(G, dim=1)
            v = v.repeat_interleave(G, dim=1)

            q = query[b]

            qf = q.float()
            kf = k.float()
            vf = v.float()

            scores = torch.einsum("hd,lhd->hl", qf, kf) * self.scale
            probs = torch.softmax(scores, dim=-1)

            o = torch.einsum("hl,lhd->hd", probs, vf)
            out[b] = o.to(query.dtype)

        return out


def get_inputs(batch, num_heads, num_kv_heads, head_dim, seq_len, page_size):
    B = batch
    H = num_heads
    Hkv = num_kv_heads
    D = head_dim
    L = seq_len
    P = page_size

    pages_per_seq = (L + P - 1) // P

    total_pages = max(B * pages_per_seq + 8, 64)

    query = torch.randn(B, H, D, dtype=torch.bfloat16) * 0.1
    kv_cache = torch.randn(total_pages, P, Hkv, 2 * D, dtype=torch.bfloat16) * 0.1

    perm = torch.randperm(total_pages)[: B * pages_per_seq].reshape(B, pages_per_seq).int()

    block_table = perm.contiguous()
    seq_lens = torch.full((B,), L, dtype=torch.int32)

    return [query, kv_cache, block_table, seq_lens]
