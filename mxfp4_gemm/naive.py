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
def _mxfp4_kernel(A, B, sfa, sfb, C):
    """One thread per output element; decode both MXFP4 operands into FP32.

    C[m, n] = sum_k decode(A[m, k]) * sfa[m, k//32]
                      * decode(B[k, n]) * sfb[k//32, n]
    A and B pack two E2M1 codes per byte along K: even K -> low nibble.
    Each scale byte represents 2^(bits - 127); scales use packed 128x4 blocks.
    The scale indices in the equation above are logical, before packing.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    M, N = C.shape
    K = A.shape[1] * 2
    mi = bidy
    ni = bidx * THREADS + tidx

    if ni < N:
        acc = cutlass.Float32(0.0)
        for kh in cutlass.range(0, K // 2, 1):
            a = A[mi, kh]
            b = B[kh, ni]
            blk = kh // (BLOCK // 2)
            # Each scale block holds 128 rows/columns x 4 scales in 512 bytes.
            # Within it, rows are interleaved as [row % 32, row // 32, scale % 4].
            sf_a_offset = (blk // 4) * 512 + (mi % 32) * 16 + ((mi % 128) // 32) * 4 + blk % 4
            sf_b_offset = (blk // 4) * 512 + (ni % 32) * 16 + ((ni % 128) // 32) * 4 + blk % 4
            # Before: sfa[mi, blk] and sfb[blk, ni] read linear scales.
            sa = cute.math.exp2(cutlass.Float32(cutlass.Int32(sfa[mi // 128, sf_a_offset])) - 127.0)
            sb = cute.math.exp2(cutlass.Float32(cutlass.Int32(sfb[ni // 128, sf_b_offset])) - 127.0)
            acc += (_e2m1(a & 0xF) * sa) * (_e2m1(b & 0xF) * sb)
            acc += (_e2m1(a >> 4) * sa) * (_e2m1(b >> 4) * sb)
        C[mi, ni] = C.element_type(acc)


@cute.jit
def mxfp4(A, B, sfa, sfb, C, stream: CUstream):
    M, N = C.shape
    _mxfp4_kernel(A, B, sfa, sfb, C).launch(
        grid=(cute.ceil_div(N, THREADS), M, 1), block=(THREADS, 1, 1), stream=stream
    )
