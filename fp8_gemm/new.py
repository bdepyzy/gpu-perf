import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.cute as cute
import cuda.bindings.driver as cuda_driver

import torch
from typing import Callable, List
from cutlass.experimental import primitives as prims


def _get_default_stream() -> cuda_driver.CUstream:
    return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)


@cute.kernel
def gemm_kernel(
    tma_desc_a: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_b: cutlass.GridConstant[cuda.TensorMap],
    matrix_c_arr: cutlass.Array,
    problem_size: cutlass.Constexpr[List[int]],
) -> None:
    M, K, N = problem_size

    tx, _, _ = cute.arch.thread_idx()
    warp_id = cute.arch.warp_idx()

    tmem_num_col = N

    smem_a = cutlass.Array(
        cutlass.Float8E4M3FN, (M, K), space=cutlass.AddressSpace.smem, alignment=64
    )
    smem_b = cutlass.Array(
        cutlass.Float8E4M3FN, (N, K), space=cutlass.AddressSpace.smem, alignment=64
    )

    mbar_tma = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    mbar_mma = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    if prims.elect_sync():
        prims.prefetch_tensormap(tma_desc_a.get_ptr())
        prims.prefetch_tensormap(tma_desc_b.get_ptr())
        prims.mbarrier_init(mbar_tma, 1)
        prims.mbarrier_init(mbar_mma, 1)

    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    is_epi_warp = warp_id < 4
    is_tma_warp = warp_id == 4
    is_tc_warp = warp_id == 5

    if is_tc_warp:
        prims.tcgen05_alloc(tmem_ptr_i32, tmem_num_col)
        prims.tcgen05_relinquish_alloc_permit()

    prims.barrier_cta_sync(0)

    tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    if is_tma_warp:
        prims.setmaxregister(40, prims.SetMaxRegisterAction.DECREASE)
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        if prims.elect_sync():
            sz = tma_desc_a.global_tx_bytes() + tma_desc_b.global_tx_bytes()
            prims.mbarrier_arrive_expect_tx(mbar_tma, sz)

            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_a, tma_desc_a.get_ptr(), (0, 0), mbar_tma
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_b, tma_desc_b.get_ptr(), (0, 0), mbar_tma
            )

    elif is_tc_warp:
        prims.bar_warp_sync(cute.arch.FULL_MASK)

        if prims.elect_sync():
            while not prims.mbarrier_try_wait_parity(mbar_tma, 0, time_limit=10000000):
                pass

            desc_a = prims.Tcgen05SmemDesc.build(
                smem_a, leading_byte_offset=0, stride_byte_offset=512, layout=4
            )
            desc_b = prims.Tcgen05SmemDesc.build(
                smem_b, leading_byte_offset=0, stride_byte_offset=512, layout=4
            )

            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=cutlass.Float16,
                a_dtype=cutlass.Float16,
                b_dtype=cutlass.Float16,
                n_dim=128,
                m_dim=128,
            )

            scale_d = False

            for i in cutlass.range_constexpr(K // 32):
                prims.tcgen05_mma(
                    prims.Tcgen05MMAKind.F8F6F4,
                    prims.CTAGroup.CTA_1,
                    tmem_ptr,
                    desc_a,
                    desc_b,
                    idesc,
                    scale_d,
                )

                scale_d = True

                desc_a = desc_a.advance_start_address(32 * 1)
                desc_b = desc_b.advance_start_address(32 * 1)

            prims.tcgen05_commit(mbar_mma)

    elif is_epi_warp:
        while not prims.mbarrier_try_wait_parity(mbar_mma, 0, time_limit=10000000):
            pass

        tid_in_epi_wg = tx % 128
        warpid_in_epi_wg = warp_id % 4
        tmem_raw_addr = tmem_ptr_i32.load()

        base_col_id = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16

        row_id = base_row_id + warpid_in_epi_wg * 32

        tmem_x = 32
        for n in range(0, tmem_num_col, tmem_x):
            c_row = matrix_c_arr[tid_in_epi_wg, :]

            col_id = base_col_id + n
            tmem_offset = (row_id << 16) | col_id

            shape = "32x32b"

            tmem_addr_ptr = cutlass.inttoptr(
                tmem_offset, mem_space=6, dtype=cutlass.Float16
            )

            c_rmem_fp16 = prims.tcgen05_ld(
                shape, tmem_addr_ptr, num=tmem_x // 2, pack=True
            )
            c_row[n:tmem_x] = c_rmem_fp16

    prims.barrier_cta_sync(0)

    if is_tc_warp:
        prims.tcgen05_dealloc(tmem_ptr, tmem_num_col)


@cute.jit
def gemm(
    matrix_a: cute.Tensor,
    matrix_b: cute.Tensor,
    matrix_c: cutlass.Array,
    problem_size: cutlass.Constexpr[List[int]],
    stream: cuda_driver.CUstream,
) -> None:
    M, K, N = problem_size

    tma_desc_a = cuda.create_tensor_map_tiled_from_view(
        matrix_a, box_dims=(M, K), swizzle=cuda.TensorMapSwizzle.s64b
    )
    tma_desc_b = cuda.create_tensor_map_tiled_from_view(
        matrix_b, box_dims=(N, K), swizzle=cuda.TensorMapSwizzle.s64b
    )

    block = (256, 1, 1)
    grid = (1, 1, 1)
    gemm_kernel(tma_desc_a, tma_desc_b, matrix_c, problem_size).launch(
        grid=grid, block=block, stream=stream
    )


def get_compiled_gemm(M: int, K: int, N: int) -> Callable:
    cutlass.cuda.initialize_cuda_context()

    A_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (M, K),
        stride_order=(1, 0),
        assumed_align=16,
    )
    B_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (N, K),
        stride_order=(1, 0),
        assumed_align=16,
    )
    C_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float16,
        (M, N),
        stride_order=(1, 0),
        assumed_align=16,
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)

    compiled_fn = cute.compile(
        gemm,
        A_fake,
        B_fake,
        C_fake,
        (M, K, N),
        fake_stream,
        options="--enable-tvm-ffi",
    )

    return compiled_fn


if __name__ == "__main__":
    M, N, K = (128, 128, 64)
    a = torch.randn(M, K, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(N, K, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    c = torch.zeros(M, N, dtype=torch.float16, device="cuda")

    print(f"Matrix: {M}x{K} @ {K}x{N} = {M}x{N}")

    print("Compiling kernel...")
    compiled_gemm = get_compiled_gemm(M, K, N)

    host_c = a.to(torch.float16) @ b.to(torch.float16).T
    print("@torch result:  ", host_c)

    stream = _get_default_stream()

    print("\nRunning kernel...")
    compiled_gemm(a, b, c, stream)
    torch.cuda.synchronize()

    print("@cutlass result:   ", c)
    torch.testing.assert_close(c, host_c, atol=1e-01, rtol=1e-01)

    print("PASS")
