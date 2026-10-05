import cutlass
from cutlass import cute

BLOCK = 128


@cute.kernel
def _paged_kernel(q, kv_cache, block_table, seq_lens, out):
    tidx, _, _ = cute.arch.thread_idx()
    b, h, _ = cute.arch.block_idx()

    _, _, D = out.shape
    Hkv = kv_cache.shape[2]
    P = kv_cache.shape[1]
    G = out.shape[1] // Hkv
    hkv = h // G
    L = seq_lens[b]
    scale = 1.0 / cute.math.sqrt(cutlass.Float32(D))

    red = cutlass.Array(cutlass.Float32, BLOCK, space=cutlass.AddressSpace.smem)

    qd = cutlass.Float32(0.0)
    acc = cutlass.Float32(0.0)
    if tidx < D:
        qd = cutlass.Float32(q[b, h, tidx])

    max_s = cutlass.Float32(-float("inf"))
    for l in cutlass.range(0, L, 1):
        blk = block_table[b, l // P]
        part = cutlass.Float32(0.0)
        if tidx < D:
            part = qd * cutlass.Float32(kv_cache[blk, l % P, hkv, tidx])
        red.store(part, tidx)
        cute.arch.sync_threads()
        for step in cutlass.range_constexpr(7):
            stride = BLOCK // 2 >> step
            if tidx < stride:
                a = red.load(tidx)
                o = red.load(tidx + stride)
                red.store(a + o, tidx)
            cute.arch.sync_threads()
        s = red.load(0) * scale
        if s > max_s:
            max_s = s

    total = cutlass.Float32(0.0)
    for l in cutlass.range(0, L, 1):
        blk = block_table[b, l // P]
        part = cutlass.Float32(0.0)
        if tidx < D:
            part = qd * cutlass.Float32(kv_cache[blk, l % P, hkv, tidx])
        red.store(part, tidx)
        cute.arch.sync_threads()
        for step in cutlass.range_constexpr(7):
            stride = BLOCK // 2 >> step
            if tidx < stride:
                a = red.load(tidx)
                o = red.load(tidx + stride)
                red.store(a + o, tidx)
            cute.arch.sync_threads()
        w = cute.math.exp(red.load(0) * scale - max_s)
        total += w
        if tidx < D:
            acc += w * cutlass.Float32(kv_cache[blk, l % P, hkv, D + tidx])

    if tidx < D:
        out[b, h, tidx] = out.element_type(acc / total)


@cute.jit
def paged_attention(q, kv_cache, block_table, seq_lens, out):
    _, H, _ = out.shape
    D = out.shape[2]
    _paged_kernel(q, kv_cache, block_table, seq_lens, out).launch(
        grid=(out.shape[0], H, 1), block=(BLOCK, 1, 1)
    )
