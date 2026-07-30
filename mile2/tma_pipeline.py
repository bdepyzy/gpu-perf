import modal

from bench import benchmark_gemm

image = modal.Image.debian_slim(python_version="3.13").uv_pip_install("torch==2.11.0", "cutlass", "nvidia-cutlass", "nvidia-cutlass-dsl").add_local_python_source("bench")
app = modal.App("milestone2", image=image)


@app.function(gpu="B200", timeout=1800)
def run():
    from typing import Type, Union

    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.nvgpu import cpasync
    from cutlass.cute.runtime import from_dlpack
    import torch

    class Kernel:
        def __init__(self):
            self.tile_shape = (128, 128)
            self.tile_m, self.tile_n = self.tile_shape
            self.tile_k = 32
            self.thread_tile_shape = (8, 8)
            self.thread_tile_m, self.thread_tile_n = self.thread_tile_shape
            self.threads_x, self.threads_y = 16, 16
            self.cluster_shape_mn = (1, 1)
            self.threads_per_cta = self.threads_x * self.threads_y
            self.buffer_align_bytes = 1024

        @cute.jit
        def __call__(self, srcA, srcB, dstC):
            self.dtype: Type[cutlass.Numeric] = srcA.element_type

            # These are the two trays used by one K iteration.
            sA_layout = cute.make_layout((self.tile_m, self.tile_k), stride=(self.tile_k, 1))
            sB_layout = cute.make_layout((self.tile_k, self.tile_n), stride=(self.tile_n, 1))

            tma_load_bytes = cute.size_in_bytes(self.dtype, sA_layout) + cute.size_in_bytes(self.dtype, sB_layout)

            tma_atom_src_A, tma_tensor_src_A = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                srcA,
                sA_layout,
                cute.product_each(sA_layout.shape),
            )

            tma_atom_src_B, tma_tensor_src_B = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                srcB,
                sB_layout,
                cute.product_each(sB_layout.shape),
            )

            blocks_m = dstC.shape[0] // self.tile_m
            blocks_n = dstC.shape[1] // self.tile_n
            self.kernel(
                srcA,
                srcB,
                dstC,
                tma_atom_src_A,
                tma_tensor_src_A,
                tma_atom_src_B,
                tma_tensor_src_B,
                sA_layout,
                sB_layout,
                tma_load_bytes,
            ).launch(
                grid=(blocks_n, blocks_m, 1),
                block=(self.threads_x, self.threads_y, 1),
                cluster=(*self.cluster_shape_mn, 1),
            )

        @cute.kernel
        def kernel(
            self,
            gA,
            gB,
            gC,
            tma_atom_src_A: cute.CopyAtom,
            tma_tensor_src_A: cute.Tensor,
            tma_atom_src_B: cute.CopyAtom,
            tma_tensor_src_B: cute.Tensor,
            sA_layout: Union[cute.Layout, cute.ComposedLayout],
            sB_layout: Union[cute.Layout, cute.ComposedLayout],
            tma_load_bytes: cutlass.Constexpr,
        ):
            # A[M,K] B[K,N]
            tx, ty, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()
            # Top-left C coordinate owned by this thread
            m0 = bidy * self.tile_m + ty * self.thread_tile_m
            n0 = bidx * self.tile_n + tx * self.thread_tile_n
            Kdim = gA.shape[1]

            # This thread's output tile uses FP32 accumulators.
            acc = cute.make_rmem_tensor(self.thread_tile_shape, cutlass.Float32)

            for mi in cutlass.range_constexpr(self.thread_tile_m):
                for ni in cutlass.range_constexpr(self.thread_tile_n):
                    acc[(mi, ni)] = cutlass.Float32(0.0)

            smem = utils.SmemAllocator()
            barrier_ptr = smem.allocate(cutlass.Int64, byte_alignment=8)

            sA = smem.allocate_tensor(
                gA.element_type,
                sA_layout,
                byte_alignment=self.buffer_align_bytes,
            )
            sB = smem.allocate_tensor(
                gB.element_type,
                sB_layout,
                byte_alignment=self.buffer_align_bytes,
            )

            with cute.arch.elect_one():
                cute.arch.mbarrier_init(barrier_ptr, 1)

            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            # A is tiled over (M, K); B is tiled over (K, N).
            gAsrc_tiled = cute.local_tile(
                tma_tensor_src_A,
                (self.tile_m, self.tile_k),
                (None, None),
            )
            gBsrc_tiled = cute.local_tile(
                tma_tensor_src_B,
                (self.tile_k, self.tile_n),
                (None, None),
            )

            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                tma_atom_src_A,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gAsrc_tiled, 0, 2),
            )

            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                tma_atom_src_B,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gBsrc_tiled, 0, 2),
            )

            # Per-k register fragments of the original micro-tile.
            a_frag = cute.make_rmem_tensor((self.thread_tile_m, 1), gA.element_type)
            b_frag = cute.make_rmem_tensor((1, self.thread_tile_n), gB.element_type)

            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            phase = cutlass.Int32(0)
            for k0 in range(0, Kdim, self.tile_k):
                k_tile = k0 // self.tile_k

                # One warp dispatches the two deliveries. All warps compute below.
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(barrier_ptr, tma_load_bytes)

                    cute.copy(
                        tma_atom_src_A,
                        tAgA[(None, bidy, k_tile)],
                        tAsA,
                        tma_bar_ptr=barrier_ptr,
                    )

                    cute.copy(
                        tma_atom_src_B,
                        tBgB[(None, k_tile, bidx)],
                        tBsB,
                        tma_bar_ptr=barrier_ptr,
                    )

                cute.arch.mbarrier_wait(barrier_ptr, phase)

                for k in cutlass.range_constexpr(self.tile_k):
                    # cute.local_tile(tensor, (tile_height, tile_width), (tile_row, tile_col)) == 
                    #tensor[tile_row * tile_height : (tile_row + 1) * tile_height,
                    #       tile_col * tile_width  : (tile_col + 1) * tile_width,
                    #]
                    cute.autovec_copy(
                        cute.local_tile(sA, (self.thread_tile_m, 1), (ty, k)),
                        a_frag,
                    )
                    cute.autovec_copy( 
                        cute.local_tile(sB, (1, self.thread_tile_n), (k, tx)), 
                        b_frag,
                    )

                    for mi in cutlass.range_constexpr(self.thread_tile_m):
                        a = cutlass.Float32(a_frag[(mi, 0)])
                        for ni in cutlass.range_constexpr(self.thread_tile_n):
                            acc[(mi, ni)] = acc[(mi, ni)] + a * cutlass.Float32(b_frag[(0, ni)])
                cute.arch.sync_threads()
                phase = phase ^ 1

            for mi in cutlass.range_constexpr(self.thread_tile_m):
                for ni in cutlass.range_constexpr(self.thread_tile_n):
                    gC[(m0 + mi, n0 + ni)] = gC.element_type(acc[(mi, ni)])

    shapes = [256, 512, 768, 1024, 2048, 4096, 8192]
    warmup = 5
    iterations = 100

    print(f"Shapes={len(shapes)} | warmup={warmup} | iterations={iterations} | C[M,N]=A[M,K]@B[K,N]")

    torch.manual_seed(0)
    results = []

    for size in shapes:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        c = torch.empty(size, size, device="cuda", dtype=torch.float16)
        a_cute = from_dlpack(a, assumed_align=16)
        b_cute = from_dlpack(b, assumed_align=16)
        c_cute = from_dlpack(c, assumed_align=16)
        gemm = cute.compile(Kernel(), a_cute, b_cute, c_cute)
        gemm(a_cute, b_cute, c_cute)

        reference = torch.matmul(a, b)
        abs_error = (c - reference).abs()
        max_abs_error = abs_error.max().item()
        mean_abs_error = abs_error.float().mean().item()
        try:
            torch.testing.assert_close(c, reference, rtol=1e-2, atol=5e-1)
        except AssertionError:
            print(f"CORRECTNESS FAILED {size}x{size}x{size} | max_abs={max_abs_error:.6f} | mean_abs={mean_abs_error:.6f}")
            raise

        result = benchmark_gemm(gemm, a_cute, b_cute, c_cute, a, b, size, size, size, warmup=warmup, iterations=iterations)
        result["max_abs_error"] = max_abs_error
        result["mean_abs_error"] = mean_abs_error
        results.append(result)
        del a, b, c, reference
        torch.cuda.empty_cache()

    return results
