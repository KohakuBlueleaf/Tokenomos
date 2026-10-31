"""NVFP4 producers: codes and scales decoded back against the input (an independent bound),
and byte equality with vLLM's ``scaled_fp4_quant`` when vLLM is installed."""

import pytest
import torch

from tokenomos.kernels.nvfp4 import ops

pytestmark = [pytest.mark.gpu, pytest.mark.sm100]
DEV = "cuda"
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)


def unswizzle(s: torch.Tensor, m: int, k: int) -> torch.Tensor:
    kb = k // 16
    kbp = -(-kb // 4) * 4
    rows = -(-m // 128) * 128
    t = s.view(torch.uint8).view(rows // 128, kbp // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)
    return t.reshape(rows, kbp)[:m, :kb].contiguous().view(torch.float8_e4m3fn).float()


def decode(q: torch.Tensor, s: torch.Tensor, gs: float, m: int, k: int) -> torch.Tensor:
    lo, hi = (q & 0xF).long(), (q >> 4).long()
    codes = torch.stack([lo, hi], dim=-1).view(m, k)
    vals = E2M1.to(q.device)[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    return (vals.view(m, k // 16, 16) * (unswizzle(s, m, k) / gs)[..., None]).view(m, k)


@pytest.mark.parametrize("m,k", [(1, 2048), (37, 4096), (300, 2560)])
def test_quant_roundtrip_bound(m, k):
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    gs = torch.tensor([448 * 6 / x.abs().max().item()], device=DEV)
    q, s = ops.nvfp4_quant(x, gs)
    back = decode(q, s, gs.item(), m, k)
    blocks = x.float().view(m, k // 16, 16)
    step = blocks.abs().amax(-1, keepdim=True) / 6
    err = (back.view(m, k // 16, 16) - blocks).abs() / step.clamp_min(1e-12)
    assert err.max().item() <= 1.0 + 0.07


@pytest.mark.parametrize("m,k", [(1, 2048), (37, 4096), (300, 2560), (5, 12288)])
@pytest.mark.parametrize("wdtype", [torch.bfloat16, torch.float32])
def test_rmsnorm_hf_rounding(m, k, wdtype):
    """The fused add + norm + quantize equals quantizing the Hugging Face-rounded norm."""
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    r = torch.randn_like(x)
    w = (torch.rand(k, device=DEV) + 0.5).to(wdtype)
    s = x + r
    h = s.float()
    n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    y = (w * n.to(torch.bfloat16)) if wdtype == torch.bfloat16 else (n * w).to(torch.bfloat16)
    gs = torch.tensor([448 * 6 / y.float().abs().max().item()], device=DEV)
    q, sc, _, _, res = ops.rmsnorm_nvfp4(x, r, w, 1e-6, gs, None)
    rq, rs = ops.nvfp4_quant(y, gs)
    assert torch.equal(res, s)
    assert (q != rq).float().mean().item() < 1e-3
    assert (sc.view(torch.uint8) != rs.view(torch.uint8)).float().mean().item() < 1e-3


@pytest.mark.parametrize("m,k", [(1, 2048), (37, 4096), (300, 2560)])
def test_bytes_vs_vllm(m, k):
    custom = pytest.importorskip("vllm._custom_ops")
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    gs = torch.tensor([448 * 6 / x.abs().max().item()], device=DEV)
    q, s = ops.nvfp4_quant(x, gs)
    rq, rs = custom.scaled_fp4_quant(x, gs)
    assert torch.equal(q, rq.view(torch.uint8))
    assert torch.equal(s.view(torch.uint8).view(-1), rs.view(torch.uint8).view(-1))
