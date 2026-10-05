import cutlass
from cutlass import cute

MUTATES_INPUT = False


def workspace_shape(batch, n, k):
    return (batch, cute.ceil_div(n, 1024), k)


class TopK:
    def __init__(self):
        self.tile = 1024
        self.threads = 256

    @cute.jit
    def __call__(self, x, values, indices, partial_values, partial_indices):
        assert values.shape[1] <= self.threads
        assert partial_values.shape[1] <= self.threads
        self.partial(x, partial_values, partial_indices).launch(
            grid=(partial_values.shape[1], x.shape[0], 1), block=(self.threads, 1, 1),
        )
        self.merge(partial_values, partial_indices, values, indices).launch(
            grid=(x.shape[0], 1, 1), block=(self.threads, 1, 1),
        )

    @cute.kernel
    def partial(self, x, values, indices):
        tid, _, _ = cute.arch.thread_idx()
        tile, b, _ = cute.arch.block_idx()
        sv = cutlass.Array(cutlass.Float32, self.tile, space=cutlass.AddressSpace.smem)
        si = cutlass.Array(cutlass.Int32, self.tile, space=cutlass.AddressSpace.smem)
        rv = cutlass.Array(cutlass.Float32, 4, space=cutlass.AddressSpace.rmem)
        ri = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(4):
            index = tile * self.tile + tid + j * self.threads
            value = cutlass.Float32(-float("inf"))
            if index < x.shape[1]:
                value = x[b, index]
            rv.store(value, j)
            ri.store(index, j)
        for level in cutlass.range_constexpr(1, 11):
            for step in cutlass.range_constexpr(level):
                distance = 1 << (level - step - 1)
                if cutlass.const_expr(distance >= self.threads):
                    for j in cutlass.range_constexpr(4):
                        peer = j ^ (distance // self.threads)
                        if cutlass.const_expr(j < peer):
                            a, ai = rv.load(j), ri.load(j)
                            v, idx = rv.load(peer), ri.load(peer)
                            better = (v > a) | ((v == a) & (idx < ai))
                            descending = ((tid + j * self.threads) & (1 << level)) == 0
                            if better == descending:
                                rv.store(v, j)
                                ri.store(idx, j)
                                rv.store(a, peer)
                                ri.store(ai, peer)
                else:
                    if cutlass.const_expr(distance >= 32):
                        for j in cutlass.range_constexpr(4):
                            sv.store(rv.load(j), tid + j * self.threads)
                            si.store(ri.load(j), tid + j * self.threads)
                        cute.arch.sync_threads()
                    for j in cutlass.range_constexpr(4):
                        i = tid + j * self.threads
                        a, ai = rv.load(j), ri.load(j)
                        if cutlass.const_expr(distance < 32):
                            v = cute.arch.shuffle_sync_bfly(a, offset=distance)
                            idx = cute.arch.shuffle_sync_bfly(ai, offset=distance)
                        else:
                            v, idx = sv.load(i ^ distance), si.load(i ^ distance)
                        better = (v > a) | ((v == a) & (idx < ai))
                        descending = (i & (1 << level)) == 0
                        want_high = descending == ((i & distance) == 0)
                        if better == want_high:
                            rv.store(v, j)
                            ri.store(idx, j)
                    if cutlass.const_expr(distance >= 32):
                        cute.arch.sync_threads()
        if tid < values.shape[2]:
            values[b, tile, tid] = rv.load(0)
            indices[b, tile, tid] = ri.load(0)

    @cute.kernel
    def merge(self, partial_values, partial_indices, values, indices):
        tid, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        rv = cutlass.Array(cutlass.Float32, 8, space=cutlass.AddressSpace.smem)
        ri = cutlass.Array(cutlass.Int32, 9, space=cutlass.AddressSpace.smem)
        pos = cutlass.Int32(0)
        head = cutlass.Float32(-float("inf"))
        head_i = cutlass.Int32(0x7fffffff)
        if tid < partial_values.shape[1]:
            head = partial_values[b, tid, 0]
            head_i = partial_indices[b, tid, 0]
        for j in cutlass.range(values.shape[1], unroll=1):
            best, best_i = head, head_i
            for step in cutlass.range_constexpr(5):
                v = cute.arch.shuffle_sync_bfly(best, offset=1 << step)
                idx = cute.arch.shuffle_sync_bfly(best_i, offset=1 << step)
                if (v > best) | ((v == best) & (idx < best_i)):
                    best, best_i = v, idx
            if lane == 0:
                rv.store(best, warp)
                ri.store(best_i, warp)
            cute.arch.sync_threads()
            if warp == 0:
                best = cutlass.Float32(-float("inf"))
                best_i = cutlass.Int32(0x7fffffff)
                if lane < 8:
                    best, best_i = rv.load(lane), ri.load(lane)
                for step in cutlass.range_constexpr(5):
                    v = cute.arch.shuffle_sync_bfly(best, offset=1 << step)
                    idx = cute.arch.shuffle_sync_bfly(best_i, offset=1 << step)
                    if (v > best) | ((v == best) & (idx < best_i)):
                        best, best_i = v, idx
                if lane == 0:
                    values[b, j] = best
                    indices[b, j] = cutlass.Int64(best_i)
                    ri.store(best_i, 8)
            cute.arch.sync_threads()
            if head_i == ri.load(8):
                pos += 1
                head = cutlass.Float32(-float("inf"))
                head_i = cutlass.Int32(0x7fffffff)
                if pos < values.shape[1]:
                    head = partial_values[b, tid, pos]
                    head_i = partial_indices[b, tid, pos]


topk = TopK()
