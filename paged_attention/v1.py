import cutlass
from cutlass import cute


def workspace_shape(batch, heads, dim, length):
    return (batch, heads, cute.ceil_div(length, 256) * 4, dim + 2)


class PagedAttention:
    def __init__(self):
        self.tokens_per_warp = 64
        self.threads = 128

    @cute.jit
    def __call__(self, q, cache, table, lengths, out, scratch):
        assert q.shape[2] % 32 == 0
        self.partial(q, cache, table, lengths, scratch).launch(
            grid=(q.shape[0], q.shape[1], scratch.shape[2] // 4),
            block=(self.threads, 1, 1),
        )
        self.merge(scratch, out).launch(grid=(q.shape[0], q.shape[1], 1), block=(32, 1, 1))

    @cute.kernel
    def partial(self, q, cache, table, lengths, scratch):
        tid, _, _ = cute.arch.thread_idx()
        b, h, tile = cute.arch.block_idx()
        lane = tid % 32
        part = tile * 4 + tid // 32
        D = q.shape[2]
        P = cache.shape[1]
        hkv = h // (q.shape[1] // cache.shape[2])
        query = cutlass.Array(cutlass.Float32, D // 32, space=cutlass.AddressSpace.rmem)
        acc = cutlass.Array(cutlass.Float32, D // 32, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(D // 32):
            query.store(cutlass.Float32(q[b, h, lane + i * 32]), i)
            acc.store(0.0, i)
        maximum = cutlass.Float32(-float("inf"))
        total = cutlass.Float32(0.0)
        begin = part * self.tokens_per_warp
        end = cutlass.min(begin + self.tokens_per_warp, lengths[b])
        for token in cutlass.range(begin, end, unroll=1):
            page = table[b, token // P]
            score = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(D // 32):
                score += query.load(i) * cutlass.Float32(cache[page, token % P, hkv, lane + i * 32])
            score = cute.arch.warp_reduction_sum(score) * (D ** -0.5)
            new_max = cutlass.max(maximum, score)
            correction = cute.math.exp(maximum - new_max)
            weight = cute.math.exp(score - new_max)
            total = total * correction + weight
            for i in cutlass.range_constexpr(D // 32):
                value = cutlass.Float32(cache[page, token % P, hkv, D + lane + i * 32])
                acc.store(acc.load(i) * correction + weight * value, i)
            maximum = new_max
        for i in cutlass.range_constexpr(D // 32):
            scratch[b, h, part, lane + i * 32] = acc.load(i)
        if lane == 0:
            scratch[b, h, part, D] = maximum
            scratch[b, h, part, D + 1] = total

    @cute.kernel
    def merge(self, scratch, out):
        lane, _, _ = cute.arch.thread_idx()
        b, h, _ = cute.arch.block_idx()
        D = out.shape[2]
        maximum = cutlass.Float32(-float("inf"))
        for part in cutlass.range(lane, scratch.shape[2], 32):
            maximum = cutlass.max(maximum, scratch[b, h, part, D])
        maximum = cute.arch.warp_reduction_max(maximum)
        for i in cutlass.range_constexpr(D // 32):
            acc = cutlass.Float32(0.0)
            total = cutlass.Float32(0.0)
            for part in cutlass.range(scratch.shape[2], unroll=1):
                count = scratch[b, h, part, D + 1]
                if count > 0.0:
                    weight = cute.math.exp(scratch[b, h, part, D] - maximum)
                    acc += weight * scratch[b, h, part, lane + i * 32]
                    total += weight * count
            result = cutlass.Float32(0.0)
            if total > 0.0:
                result = acc / total
            out[b, h, lane + i * 32] = out.element_type(result)


paged_attention = PagedAttention()
