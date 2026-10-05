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
        for i in cutlass.range(tid, self.tile, self.threads):
            index = tile * self.tile + i
            value = cutlass.Float32(-float("inf"))
            if index < x.shape[1]:
                value = x[b, index]
            sv.store(value, i)
            si.store(index, i)
        cute.arch.sync_threads()
        for level in cutlass.range_constexpr(1, 11):
            for step in cutlass.range_constexpr(level):
                distance = 1 << (level - step - 1)
                for i in cutlass.range(tid, self.tile, self.threads):
                    peer = i ^ distance
                    if peer > i:
                        a, ai = sv.load(i), si.load(i)
                        bval, bi = sv.load(peer), si.load(peer)
                        better = (bval > a) | ((bval == a) & (bi < ai))
                        descending = (i & (1 << level)) == 0
                        if better == descending:
                            sv.store(bval, i)
                            si.store(bi, i)
                            sv.store(a, peer)
                            si.store(ai, peer)
                cute.arch.sync_threads()
        if tid < values.shape[2]:
            values[b, tile, tid] = sv.load(tid)
            indices[b, tile, tid] = si.load(tid)

    @cute.kernel
    def merge(self, partial_values, partial_indices, values, indices):
        tid, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        k = values.shape[1]
        count = partial_values.shape[1] * k
        rv = cutlass.Array(cutlass.Float32, 8, space=cutlass.AddressSpace.smem)
        ri = cutlass.Array(cutlass.Int32, 8, space=cutlass.AddressSpace.smem)
        rp = cutlass.Array(cutlass.Int32, 8, space=cutlass.AddressSpace.smem)
        for j in cutlass.range(k, unroll=1):
            best = cutlass.Float32(-float("inf"))
            best_i = cutlass.Int32(0x7fffffff)
            best_p = cutlass.Int32(0)
            for p in cutlass.range(tid, count, self.threads):
                value = partial_values[b, p // k, p % k]
                index = partial_indices[b, p // k, p % k]
                if (value > best) | ((value == best) & (index < best_i)):
                    best, best_i, best_p = value, index, p
            for step in cutlass.range_constexpr(5):
                v = cute.arch.shuffle_sync_bfly(best, offset=1 << step)
                idx = cute.arch.shuffle_sync_bfly(best_i, offset=1 << step)
                pos = cute.arch.shuffle_sync_bfly(best_p, offset=1 << step)
                if (v > best) | ((v == best) & (idx < best_i)):
                    best, best_i, best_p = v, idx, pos
            if lane == 0:
                rv.store(best, warp)
                ri.store(best_i, warp)
                rp.store(best_p, warp)
            cute.arch.sync_threads()
            if warp == 0:
                best = cutlass.Float32(-float("inf"))
                best_i = cutlass.Int32(0x7fffffff)
                best_p = cutlass.Int32(0)
                if lane < 8:
                    best, best_i, best_p = rv.load(lane), ri.load(lane), rp.load(lane)
                for step in cutlass.range_constexpr(5):
                    v = cute.arch.shuffle_sync_bfly(best, offset=1 << step)
                    idx = cute.arch.shuffle_sync_bfly(best_i, offset=1 << step)
                    pos = cute.arch.shuffle_sync_bfly(best_p, offset=1 << step)
                    if (v > best) | ((v == best) & (idx < best_i)):
                        best, best_i, best_p = v, idx, pos
                if lane == 0:
                    values[b, j] = best
                    indices[b, j] = cutlass.Int64(best_i)
                    partial_values[b, best_p // k, best_p % k] = cutlass.Float32(-float("inf"))
                    partial_indices[b, best_p // k, best_p % k] = cutlass.Int32(0x7fffffff)
            cute.arch.sync_threads()


topk = TopK()
