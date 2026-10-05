import cutlass
from cutlass import cute

THREADS = 256

@cute.kernel
def _topk_kernel(x, values, indices):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    n = x.shape[1]
    k = values.shape[1]

    red_v = cutlass.Array(cutlass.Float32, THREADS, space=cutlass.AddressSpace.smem)
    red_i = cutlass.Array(cutlass.Int32, THREADS, space=cutlass.AddressSpace.smem)

    for j in cutlass.range(0, k, 1):
        best = cutlass.Float32(-float("inf"))
        best_i = cutlass.Int32(0)
        for i in cutlass.range(tidx, n, THREADS):
            v = x[bidx, i]
            if v > best:
                best = v
                best_i = i
            else:
                if v == best:
                    if i < best_i:
                        best_i = i
        red_v.store(best, tidx)
        red_i.store(best_i, tidx)
        cute.arch.sync_threads()

        for step in cutlass.range_constexpr(8):
            stride = THREADS // 2 >> step
            if tidx < stride:
                mine_v = red_v.load(tidx)
                mine_i = red_i.load(tidx)
                other_v = red_v.load(tidx + stride)
                other_i = red_i.load(tidx + stride)
                if other_v > mine_v:
                    red_v.store(other_v, tidx)
                    red_i.store(other_i, tidx)
                else:
                    if other_v == mine_v:
                        if other_i < mine_i:
                            red_i.store(other_i, tidx)
            cute.arch.sync_threads()

        if tidx == 0:
            win_v = red_v.load(0)
            win_i = red_i.load(0)
            values[bidx, j] = win_v
            indices[bidx, j] = cutlass.Int64(win_i)

            x[bidx, win_i] = cutlass.Float32(-float("inf"))
        cute.arch.sync_threads()


@cute.jit
def topk(x, values, indices):
    batch = x.shape[0]
    _topk_kernel(x, values, indices).launch(grid=(batch, 1, 1), block=(THREADS, 1, 1))
