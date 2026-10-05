import cutlass
from cutlass import cute


@cute.kernel
def _kda_kernel(q, k, v, g, beta, out):
    tidx, _, _ = cute.arch.thread_idx()
    b, h, _ = cute.arch.block_idx()

    _, T, _, K = q.shape
    V = v.shape[3]
    scale = 1.0 / cute.math.sqrt(cutlass.Float32(K))

    S = cutlass.Array(cutlass.Float32, (K, V), space=cutlass.AddressSpace.smem)
    for ki in cutlass.range(0, K, 1):
        S.store(cutlass.Float32(0.0), (ki, tidx))
    cute.arch.sync_threads()

    for t in cutlass.range(0, T, 1):
        b_t = cutlass.Float32(beta[b, t, h])
        kv = cutlass.Float32(0.0)
        qS = cutlass.Float32(0.0)
        qk = cutlass.Float32(0.0)
        for ki in cutlass.range(0, K, 1):
            kt = cutlass.Float32(k[b, t, h, ki])
            dg = cute.math.exp(cutlass.Float32(g[b, t, h, ki]))
            s_kv = S.load((ki, tidx))
            kv += kt * dg * s_kv
            qt = cutlass.Float32(q[b, t, h, ki])
            qS += qt * dg * s_kv
            qk += qt * kt
        u = b_t * (cutlass.Float32(v[b, t, h, tidx]) - kv)
        out[b, t, h, tidx] = out.element_type(scale * (qS + qk * u))
        for ki in cutlass.range(0, K, 1):
            kt = cutlass.Float32(k[b, t, h, ki])
            dg = cute.math.exp(cutlass.Float32(g[b, t, h, ki]))
            S.store(dg * S.load((ki, tidx)) + kt * u, (ki, tidx))


@cute.jit
def kda(q, k, v, g, beta, out):
    _, _, H, _ = q.shape
    V = v.shape[3]
    _kda_kernel(q, k, v, g, beta, out).launch(grid=(q.shape[0], H, 1), block=(V, 1, 1))
