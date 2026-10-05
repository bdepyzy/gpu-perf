import cutlass
from cutlass import cute


class KDA:
    def __init__(self):
        self.threads = 128
        self.chunk = 16
        self.values = 16

    @cute.jit
    def __call__(self, q, k, v, g, beta, out):
        assert q.shape[3] % 32 == 0
        assert v.shape[3] % self.values == 0
        self.kernel(q, k, v, g, beta, out).launch(
            grid=(q.shape[0], q.shape[2], v.shape[3] // self.values),
            block=(self.threads, 1, 1),
        )

    @cute.kernel
    def kernel(self, q, k, v, g, beta, out):
        tid, _, _ = cute.arch.thread_idx()
        b, h, tile = cute.arch.block_idx()
        ki = tid % 8
        vi = tid // 8
        K = q.shape[3]
        state = cutlass.Array(cutlass.Float32, K // 8, space=cutlass.AddressSpace.rmem)
        sq = cutlass.Array(cutlass.Float32, (self.chunk, K), space=cutlass.AddressSpace.smem)
        sk = cutlass.Array(cutlass.Float32, (self.chunk, K), space=cutlass.AddressSpace.smem)
        sg = cutlass.Array(cutlass.Float32, (self.chunk, K), space=cutlass.AddressSpace.smem)
        sv = cutlass.Array(cutlass.Float32, (self.chunk, self.values), space=cutlass.AddressSpace.smem)
        sb = cutlass.Array(cutlass.Float32, self.chunk, space=cutlass.AddressSpace.smem)
        for r in cutlass.range_constexpr(K // 8):
            state.store(0.0, r)
        for begin in cutlass.range(0, q.shape[1], self.chunk, unroll=1):
            count = cutlass.min(self.chunk, q.shape[1] - begin)
            for i in cutlass.range(tid, count * K, self.threads):
                t, d = i // K, i % K
                sq.store(cutlass.Float32(q[b, begin + t, h, d]) * (K ** -0.5), (t, d))
                sk.store(cutlass.Float32(k[b, begin + t, h, d]), (t, d))
                sg.store(cute.math.exp(cutlass.Float32(g[b, begin + t, h, d])), (t, d))
            for i in cutlass.range(tid, count * self.values, self.threads):
                t, d = i // self.values, i % self.values
                sv.store(cutlass.Float32(v[b, begin + t, h, tile * self.values + d]), (t, d))
            if tid < count:
                sb.store(cutlass.Float32(beta[b, begin + tid, h]), tid)
            cute.arch.sync_threads()
            for t in cutlass.range(count, unroll=1):
                predicted = cutlass.Float32(0.0)
                projected = cutlass.Float32(0.0)
                qk = cutlass.Float32(0.0)
                for r in cutlass.range_constexpr(K // 8):
                    d = ki + r * 8
                    decayed = state.load(r) * sg.load((t, d))
                    state.store(decayed, r)
                    key = sk.load((t, d))
                    query = sq.load((t, d))
                    predicted += key * decayed
                    projected += query * decayed
                    qk += query * key
                predicted = cute.arch.warp_reduction_sum(predicted, threads_in_group=8)
                projected = cute.arch.warp_reduction_sum(projected, threads_in_group=8)
                qk = cute.arch.warp_reduction_sum(qk, threads_in_group=8)
                delta = sb.load(t) * (sv.load((t, vi)) - predicted)
                for r in cutlass.range_constexpr(K // 8):
                    state.store(state.load(r) + sk.load((t, ki + r * 8)) * delta, r)
                if ki == 0:
                    out[b, begin + t, h, tile * self.values + vi] = out.element_type(projected + qk * delta)
            cute.arch.sync_threads()


kda = KDA()
