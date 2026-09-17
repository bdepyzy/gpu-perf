import cutlass
from cutlass import cute

THREADS = 256


def _e3m2(code):
    """Decode one e3m2 code (6 bits: sign, 3 exp, 2 mantissa) to fp32. Branch-free."""
    sign = (code >> 5) & 1
    e = (code >> 2) & 7
    m = cutlass.Float32(code & 3)
    e0 = cutlass.Float32(e == 0)
    # normal: 2^(e-3) * (1 + m/4); subnormal (e==0): m/16
    mag_norm = cute.math.exp2(cutlass.Float32(cutlass.Int32(e) - 3)) * (cutlass.Float32(1.0) + cutlass.Float32(0.25) * m)
    mag = e0 * (cutlass.Float32(0.0625) * m) + (cutlass.Float32(1.0) - e0) * mag_norm
    return mag * (cutlass.Float32(1.0) - cutlass.Float32(2.0) * cutlass.Float32(sign))


@cute.kernel
def _fp6_kernel(x, w_q, y):
    """One thread per output element; unpack 4 e3m2 codes per 3 bytes along K.

    y[m, n] = sum_k x[m, k] * e3m2(code_k)
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    M, N = y.shape
    K = x.shape[1]
    mi = bidy
    ni = bidx * THREADS + tidx

    if ni < N:
        acc = cutlass.Float32(0.0)
        for k4 in cutlass.range(0, K // 4, 1):
            b0 = w_q[3 * k4, ni]
            b1 = w_q[3 * k4 + 1, ni]
            b2 = w_q[3 * k4 + 2, ni]
            acc += cutlass.Float32(x[mi, 4 * k4]) * _e3m2(b0 & 0x3F)
            acc += cutlass.Float32(x[mi, 4 * k4 + 1]) * _e3m2((b0 >> 6) | ((b1 & 0xF) << 2))
            acc += cutlass.Float32(x[mi, 4 * k4 + 2]) * _e3m2((b1 >> 4) | ((b2 & 0x3) << 4))
            acc += cutlass.Float32(x[mi, 4 * k4 + 3]) * _e3m2(b2 >> 2)
        y[mi, ni] = y.element_type(acc)


@cute.jit
def fp6(x, w_q, y):
    M, N = y.shape
    _fp6_kernel(x, w_q, y).launch(
        grid=(cute.ceil_div(N, THREADS), M, 1), block=(THREADS, 1, 1)
    )
