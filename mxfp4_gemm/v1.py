from cuda.bindings.driver import CUstream

import cutlass
from cutlass import cute

THREADS = 256
BLOCK = 32  # elements per E8M0 scale block (= 16 packed bytes)


def _e2m1(nib):
    """Decode one e2m1 code (4 bits: sign, 2 exp, 1 mantissa) to fp32. Branch-free."""
    sign = (nib >> 3) & 1
    e = (nib >> 1) & 3
    m = cutlass.Float32(nib & 1)
    e0 = cutlass.Float32(e == 0)
    # normal: 2^(e-1) * (1 + 0.5m) -> {1,1.5,2,3,4,6}; subnormal (e==0): 0.5*m
    mag_norm = cute.math.exp2(cutlass.Float32(cutlass.Int32(e) - 1)) * (cutlass.Float32(1.0) + cutlass.Float32(0.5) * m)
    mag = e0 * (cutlass.Float32(0.5) * m) + (cutlass.Float32(1.0) - e0) * mag_norm
    return mag * (cutlass.Float32(1.0) - cutlass.Float32(2.0) * cutlass.Float32(sign))


@cute.kernel
def _mxfp4_kernel(x, w_q, w_scales, y):
    """One thread per output element; dequantize weights on the fly.

    y[m, n] = sum_k x[m, k] * e2m1(nibble) * 2^(scale_bits - 127)
    w_q packs two e2m1 codes per byte along K: even K -> low nibble.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    M, N = y.shape
    K = x.shape[1]
    mi = bidy
    ni = bidx * THREADS + tidx

    if ni < N:
        acc = cutlass.Float32(0.0)
        for kh in cutlass.range(0, K // 2, 1):
            byte = w_q[kh, ni]
            blk = kh // (BLOCK // 2)
            s = cute.math.exp2(cutlass.Float32(cutlass.Int32(w_scales[blk, ni])) - 127.0)
            acc += cutlass.Float32(x[mi, 2 * kh]) * _e2m1(byte & 0xF) * s
            acc += cutlass.Float32(x[mi, 2 * kh + 1]) * _e2m1(byte >> 4) * s
        y[mi, ni] = y.element_type(acc)


@cute.jit
def mxfp4(x, w_q, w_scales, y, stream: CUstream):
    M, N = y.shape
    _mxfp4_kernel(x, w_q, w_scales, y).launch(grid=(cute.ceil_div(N, THREADS), M, 1), block=(THREADS, 1, 1), stream=stream)
