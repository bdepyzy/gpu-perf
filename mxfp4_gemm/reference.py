import torch

BLOCK = 32
LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def get_inputs(M, N, K):
    from flashinfer import SfLayout, mxfp4_quantize

    torch.manual_seed(2026)
    A, sfa = mxfp4_quantize(
        torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),
        sfLayout=SfLayout.layout_linear,
    )
    B, sfb = mxfp4_quantize(
        torch.randn(N, K, dtype=torch.bfloat16, device="cuda"),
        sfLayout=SfLayout.layout_linear,
    )

    return A, B.T, sfa.reshape(M, K // BLOCK), sfb.reshape(N, K // BLOCK).T


def dequantize(packed, scales):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    levels = torch.tensor(LEVELS, device=packed.device)
    values = levels[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -values, values)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(BLOCK, dim=1)


def reference(A, B, sfa, sfb):
    torch.backends.cuda.matmul.allow_tf32 = False
    return (dequantize(A, sfa) @ dequantize(B.T, sfb.T).T).to(torch.bfloat16)
