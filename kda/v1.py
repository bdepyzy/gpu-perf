import cutlass
from cutlass import cute


class KDA:
    def __init__(self):
        self.threads = 128
        self.values_per_warp = 4

    @cute.jit
    def __call__(self, q, k, v, g, beta, out):
        assert q.shape[3] % 32 == 0
        assert v.shape[3] % 16 == 0
        self.kernel(q, k, v, g, beta, out).launch(
            grid=(q.shape[0], q.shape[2], v.shape[3] // 16),
            block=(self.threads, 1, 1),
        )

    @cute.kernel
    def kernel(self, q, k, v, g, beta, out):
        tid, _, _ = cute.arch.thread_idx()
        b, h, tile = cute.arch.block_idx()
        lane = tid % 32
        column = tile * 16 + tid // 32 * self.values_per_warp
        K = q.shape[3]
        R = K // 32
        state = cutlass.Array(cutlass.Float32, (R, 4), space=cutlass.AddressSpace.rmem)
        keys = cutlass.Array(cutlass.Float32, R, space=cutlass.AddressSpace.rmem)
        queries = cutlass.Array(cutlass.Float32, R, space=cutlass.AddressSpace.rmem)
        for r in cutlass.range_constexpr(R):
            for j in cutlass.range_constexpr(4):
                state.store(0.0, (r, j))

        for t in cutlass.range(q.shape[1], unroll=1):
            bt = cutlass.Float32(beta[b, t, h])
            for r in cutlass.range_constexpr(R):
                ki = lane + r * 32
                keys.store(cutlass.Float32(k[b, t, h, ki]), r)
                queries.store(cutlass.Float32(q[b, t, h, ki]), r)
                decay = cute.math.exp(cutlass.Float32(g[b, t, h, ki]))
                for j in cutlass.range_constexpr(4):
                    state.store(state.load((r, j)) * decay, (r, j))

            for j in cutlass.range_constexpr(4):
                predicted = cutlass.Float32(0.0)
                for r in cutlass.range_constexpr(R):
                    predicted += keys.load(r) * state.load((r, j))
                predicted = cute.arch.warp_reduction_sum(predicted)
                delta = bt * (cutlass.Float32(v[b, t, h, column + j]) - predicted)
                output = cutlass.Float32(0.0)
                for r in cutlass.range_constexpr(R):
                    updated = state.load((r, j)) + keys.load(r) * delta
                    state.store(updated, (r, j))
                    output += queries.load(r) * updated
                output = cute.arch.warp_reduction_sum(output)
                if lane == 0:
                    out[b, t, h, column + j] = out.element_type(output * (K ** -0.5))


kda = KDA()
