import cutlass
from cutlass import cute
from cutlass.utils import get_smem_capacity_in_bytes
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import TensorMap, TensorMapSwizzle, create_tensor_map_tiled_from_view
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.typing import Uint32


@dsl_user_op
def lane_id(*, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %laneid;",
            "=r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class FP8MM:
    def __init__(self):
        self.tile_m = 128
        self.tile_n = 128
        self.tile_k = 64
        self.mma_k = 32
        self.threads_per_cta = 192
        stage_bytes = self.tile_m * self.tile_k + self.tile_n * self.tile_k
        self.num_ab_stage = (get_smem_capacity_in_bytes("sm_100") - 4096) // stage_bytes

    @cute.jit
    def __call__(self, matrix_a, matrix_b, matrix_c: cutlass.Array):
        matrix_a = cute.make_tensor(cute.recast_ptr(matrix_a.iterator, dtype=cutlass.Float8E4M3FN), matrix_a.layout)
        matrix_b = cute.make_tensor(cute.recast_ptr(matrix_b.iterator, dtype=cutlass.Float8E4M3FN), matrix_b.layout)

        M, K = matrix_a.shape
        N = matrix_b.shape[0]

        tma_desc_a = create_tensor_map_tiled_from_view(matrix_a, box_dims=(self.tile_m, self.tile_k), swizzle=TensorMapSwizzle.s64b)
        tma_desc_b = create_tensor_map_tiled_from_view(matrix_b, box_dims=(self.tile_n, self.tile_k), swizzle=TensorMapSwizzle.s64b)

        grid = (N // self.tile_n, M // self.tile_m, 1)
        self.gemm_kernel(tma_desc_a, tma_desc_b, matrix_c, K).launch(grid=grid, block=(self.threads_per_cta, 1, 1))

    @cute.kernel
    def gemm_kernel(
        self,
        tma_desc_a: cutlass.GridConstant[TensorMap],
        tma_desc_b: cutlass.GridConstant[TensorMap],
        matrix_c: cutlass.Array,
        K: cutlass.Int32,
    ) -> None:
        k_tiles = K // self.tile_k
        S = self.num_ab_stage

        tx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        m_tile = bidy * self.tile_m
        n_tile = bidx * self.tile_n

        smem_a = cutlass.Array(cutlass.Float8E4M3FN, (S * self.tile_m, self.tile_k), space=cutlass.AddressSpace.smem, alignment=64)
        smem_b = cutlass.Array(cutlass.Float8E4M3FN, (S * self.tile_n, self.tile_k), space=cutlass.AddressSpace.smem, alignment=64)

        ab_full_mbar = cutlass.Array(cutlass.Int64, S, space=cutlass.AddressSpace.smem)
        ab_empty_mbar = cutlass.Array(cutlass.Int64, S, space=cutlass.AddressSpace.smem)
        ab_full_phase = cutlass.Array(cutlass.Int64, S, space=cutlass.AddressSpace.smem)
        ab_empty_phase = cutlass.Array(cutlass.Int64, S, space=cutlass.AddressSpace.smem)
        mbar_mma = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
        tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

        if prims.elect_sync():
            prims.prefetch_tensormap(tma_desc_a.get_ptr())
            prims.prefetch_tensormap(tma_desc_b.get_ptr())
            for s in range(S):
                prims.mbarrier_init(ab_full_mbar.subview(s), 1)
                prims.mbarrier_init(ab_empty_mbar.subview(s), 1)
            prims.mbarrier_init(mbar_mma, 1)
        prims.fence_mbarrier_init()

        if lane_id() == 0:
            for s in range(S):
                ab_empty_phase.store(1, s)
                ab_full_phase.store(0, s)
        prims.barrier_cta_sync(0)

        is_epi_warp = warp_id < 4
        is_tma_warp = warp_id == 4
        is_tc_warp = warp_id == 5

        if is_tc_warp:
            prims.tcgen05_alloc(tmem_ptr_i32, self.tile_n)
            prims.tcgen05_relinquish_alloc_permit()
        prims.barrier_cta_sync(0)

        tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=cutlass.Float8E4M3FN,
            b_dtype=cutlass.Float8E4M3FN,
            n_dim=self.tile_n,
            m_dim=self.tile_m,
        )

        for kt in cutlass.range(0, k_tiles, 1, unroll=1):
            stage = kt % S
            smem_a_stage = smem_a.subview(stage * self.tile_m * self.tile_k)
            smem_b_stage = smem_b.subview(stage * self.tile_n * self.tile_k)
            full_mbar = ab_full_mbar.subview(stage)
            empty_mbar = ab_empty_mbar.subview(stage)

            if is_tma_warp:
                prims.setmaxregister(40, prims.SetMaxRegisterAction.DECREASE)
                prims.bar_warp_sync(cute.arch.FULL_MASK)
                empty_bit = ab_empty_phase.load(stage)
                while not prims.mbarrier_try_wait_parity(empty_mbar, empty_bit, time_limit=10000000):
                    pass
                ab_empty_phase.store(empty_bit ^ 1, stage)
                if prims.elect_sync():
                    sz = tma_desc_a.global_tx_bytes() + tma_desc_b.global_tx_bytes()
                    prims.mbarrier_arrive_expect_tx(full_mbar, sz)

                    prims.cp_async_bulk_tensor_shared_cta_global(smem_a_stage, tma_desc_a.get_ptr(), (kt * self.tile_k, m_tile), full_mbar)
                    prims.cp_async_bulk_tensor_shared_cta_global(smem_b_stage, tma_desc_b.get_ptr(), (kt * self.tile_k, n_tile), full_mbar)

            elif is_tc_warp:
                prims.bar_warp_sync(cute.arch.FULL_MASK)
                if prims.elect_sync():
                    full_bit = ab_full_phase.load(stage)
                    while not prims.mbarrier_try_wait_parity(full_mbar, full_bit, time_limit=10000000):
                        pass
                    ab_full_phase.store(full_bit ^ 1, stage)

                    desc_a = prims.Tcgen05SmemDesc.build(smem_a_stage, leading_byte_offset=0, stride_byte_offset=512, layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B)
                    desc_b = prims.Tcgen05SmemDesc.build(smem_b_stage, leading_byte_offset=0, stride_byte_offset=512, layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B)

                    for i in cutlass.range_constexpr(self.tile_k // self.mma_k):
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F8F6F4,
                            prims.CTAGroup.CTA_1,
                            tmem_ptr,
                            desc_a,
                            desc_b,
                            idesc,
                            (kt + i) > 0,
                        )

                        desc_a = desc_a.advance_start_address(self.mma_k * 1)
                        desc_b = desc_b.advance_start_address(self.mma_k * 1)

                    prims.tcgen05_commit(empty_mbar)

        if is_tc_warp:
            if prims.elect_sync():
                prims.tcgen05_commit(mbar_mma)

        if is_epi_warp:
            while not prims.mbarrier_try_wait_parity(mbar_mma, 0, time_limit=10000000): pass

            tid_in_epi_wg = tx % 128
            warpid_in_epi_wg = warp_id % 4
            tmem_raw_addr = tmem_ptr_i32.load()

            base_col_id = tmem_raw_addr & 0xFFFF
            base_row_id = tmem_raw_addr >> 16

            row_id = base_row_id + warpid_in_epi_wg * 32
            c_row = m_tile + tid_in_epi_wg

            tmem_x = 32
            for n in range(0, self.tile_n, tmem_x):
                tmem_offset = (row_id << 16) | (base_col_id + n)
                tmem_addr_ptr = cutlass.inttoptr(tmem_offset, mem_space=cutlass.AddressSpace.tmem, dtype=cutlass.Float32)

                c_f32 = prims.tcgen05_ld(prims.Tcgen05LdStShape.SHAPE_32X32B, tmem_addr_ptr, num=tmem_x, pack=False)
                c_bf16 = c_f32.to(cutlass.BFloat16)
                matrix_c.store(c_bf16, (c_row, n_tile + n), vector_size=tmem_x)

        prims.barrier_cta_sync(0)
        if is_tc_warp: prims.tcgen05_dealloc(tmem_ptr, self.tile_n)


gemm = FP8MM()
