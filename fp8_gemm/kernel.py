from benchmark.bench import app
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def kernel(gA, gB, gC):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx
    m_dim, n_dim = gC.shape
    k_dim = gA.shape[1]
    if idx < m_dim * n_dim:
        mi = idx // n_dim
        ni = idx % n_dim
        acc = cutlass.Float32(0.0)
        for ki in range(k_dim):
            acc += cutlass.Float32(gA[mi, ki]) * cutlass.Float32(gB[ni, ki])
        gC[mi, ni] = gC.element_type(acc)


@cute.jit
def gemm_setup(mA, mB, mC, problem_size):
    M, K, N = problem_size
    mA = cute.make_tensor(cute.recast_ptr(mA.iterator, dtype=cutlass.Float8E4M3FN), mA.layout)

    mB = cute.make_tensor(cute.recast_ptr(mB.iterator, dtype=cutlass.Float8E4M3FN), mB.layout)

    m_dim, n_dim = mC.shape

    kernel(mA, mB, mC).launch(grid=(cute.ceil_div(m_dim * n_dim, 256), 1, 1), block=(256, 1, 1))


if __name__ == "__main__":
    M, N, K = (128, 128, 128)
    torch.manual_seed(2026)
    a = (torch.rand(M, K, device="cuda") * 8 - 4).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device="cuda") * 0.02).to(torch.float8_e4m3fn)
    c = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    args = (
        from_dlpack(a.view(torch.uint8), assumed_align=16),
        from_dlpack(b.view(torch.uint8), assumed_align=16),
        from_dlpack(c, assumed_align=16),
    )

    print("Compiling kernel...")
    compiled_gemm = cute.compile(kernel, *args)
    host_c = a.to(torch.bfloat16) @ b.to(torch.bfloat16).T

    compiled_gemm(*args)
    torch.cuda.synchronize()

    torch.testing.assert_close(c, host_c, atol=5e-2, rtol=5e-2)

    print("PASS")
