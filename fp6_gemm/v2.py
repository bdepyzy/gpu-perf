import cutlass
from cutlass import cute
from cutlass.experimental import primitives as prims


@cute.jit
def decode(code):
    exponent = (code >> 2) & 7
    bits = cutlass.Uint32(((code & 31) << 21) + (124 << 23))
    magnitude = bits.bitcast(cutlass.Float32)
    if exponent == 0:
        magnitude = cutlass.Float32(code & 3) * 0.0625
    if (code & 32) != 0:
        magnitude = -magnitude
    return cutlass.BFloat16(magnitude)


class FP6MM:
    def __init__(self):
        self.tile_m = 128
        self.tile_n = 128
        self.tile_k = 32
        self.threads = 128

    @cute.jit
    def __call__(self, x, packed, out: cutlass.Array):
        assert x.shape[1] % 4 == 0
        self.kernel(x, packed, out).launch(
            grid=(cute.ceil_div(out.shape[1], self.tile_n), cute.ceil_div(out.shape[0], self.tile_m), 1),
            block=(self.threads, 1, 1),
        )

    @cute.kernel
    def kernel(self, x, packed, out: cutlass.Array):
        tid, _, _ = cute.arch.thread_idx()
        bn, bm, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        M, N = out.shape
        K = x.shape[1]
        sa = cutlass.Array(cutlass.BFloat16, self.tile_m * self.tile_k, space=cutlass.AddressSpace.smem, alignment=512)
        sb = cutlass.Array(cutlass.BFloat16, self.tile_n * self.tile_k, space=cutlass.AddressSpace.smem, alignment=512)
        barrier = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
        address = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
        if tid == 0:
            prims.mbarrier_init(barrier, 1)
        prims.fence_mbarrier_init()
        if warp == 0:
            prims.tcgen05_alloc(address, self.tile_n)
            prims.tcgen05_relinquish_alloc_permit()
        cute.arch.sync_threads()
        tmem = prims.make_tmem_ptr(address.load(), cutlass.Int8)
        instruction = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=cutlass.BFloat16, b_dtype=cutlass.BFloat16,
            m_dim=self.tile_m, n_dim=self.tile_n,
        )
        for kt in cutlass.range(cute.ceil_div(K, self.tile_k), unroll=1):
            for i in cutlass.range(tid, self.tile_m * self.tile_k, self.threads):
                row, col = i // self.tile_k, i % self.tile_k
                value = cutlass.BFloat16(0.0)
                if (bm * self.tile_m + row < M) & (kt * self.tile_k + col < K):
                    value = x[bm * self.tile_m + row, kt * self.tile_k + col]
                offset = ((col // 32) * self.tile_m + row) * 32 + col % 32
                sa.store(value, offset ^ ((offset >> 3) & 24))
            for i in cutlass.range(tid, self.tile_n * (self.tile_k // 4), self.threads):
                group, col = i // self.tile_n, i % self.tile_n
                ni = bn * self.tile_n + col
                k4 = kt * (self.tile_k // 4) + group
                b0, b1, b2 = cutlass.Uint32(0), cutlass.Uint32(0), cutlass.Uint32(0)
                if (ni < N) & (k4 * 4 < K):
                    b0 = cutlass.Uint32(packed[k4 * 3, ni])
                    b1 = cutlass.Uint32(packed[k4 * 3 + 1, ni])
                    b2 = cutlass.Uint32(packed[k4 * 3 + 2, ni])
                bits = b0 | (b1 << 8) | (b2 << 16)
                for j in cutlass.range_constexpr(4):
                    ki = group * 4 + j
                    offset = ((ki // 32) * self.tile_n + col) * 32 + ki % 32
                    sb.store(decode((bits >> (6 * j)) & 63), offset ^ ((offset >> 3) & 24))
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
            cute.arch.sync_threads()
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if tid == 0:
                for group in cutlass.range_constexpr(self.tile_k // 32):
                    da = prims.Tcgen05SmemDesc.build(sa.subview(group * self.tile_m * 32), leading_byte_offset=0, stride_byte_offset=512, layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B)
                    db = prims.Tcgen05SmemDesc.build(sb.subview(group * self.tile_n * 32), leading_byte_offset=0, stride_byte_offset=512, layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B)
                    for i in cutlass.range_constexpr(2):
                        prims.tcgen05_mma(prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1,
                                         tmem, da, db, instruction, (kt + group + i) > 0)
                        da = da.advance_start_address(32)
                        db = db.advance_start_address(32)
                prims.tcgen05_commit(barrier)
            while not prims.mbarrier_try_wait_parity(barrier, kt & 1, time_limit=10000000):
                pass
            cute.arch.sync_threads()

        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        row = bm * self.tile_m + tid
        for n in cutlass.range_constexpr(0, self.tile_n, 16):
            pointer = cutlass.inttoptr(address.load() + (warp * 32 << 16) + n,
                                      mem_space=cutlass.AddressSpace.tmem, dtype=cutlass.Float32)
            values = prims.tcgen05_ld(prims.Tcgen05LdStShape.SHAPE_32X32B, pointer, num=16).to(cutlass.BFloat16)
            if row < M:
                if cutlass.const_expr(N % 8 == 0):
                    for j in cutlass.range_constexpr(0, 16, 8):
                        if bn * self.tile_n + n + j < N:
                            out.store(values[j:j + 8], (row, bn * self.tile_n + n + j))
                else:
                    for j in cutlass.range_constexpr(16):
                        if bn * self.tile_n + n + j < N:
                            out[row, bn * self.tile_n + n + j] = values[j]
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.sync_threads()
        if warp == 0:
            prims.tcgen05_dealloc(tmem, self.tile_n)


fp6 = FP6MM()
