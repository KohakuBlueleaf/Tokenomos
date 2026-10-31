"""Fused QK norm + RoPE against the model's own torch chain (``RMSNorm`` then ``Rope``), from
strided views of one fused projection, full and zero-tail partial rotation. The reduction order
differs, so a rare bf16 rounding flip is allowed; no norm must be exact.
"""

import pytest
import torch

from tokenomos.kernels.attention import qk
from tokenomos.models.blocks.rope import Rope
from tokenomos.models.contract import Step

pytestmark = pytest.mark.gpu
DEV = "cuda"


def torch_norm(x, w, eps):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)


@pytest.mark.parametrize("h,hkv,d", [(16, 4, 128), (8, 1, 256), (32, 8, 64)])
@pytest.mark.parametrize("angles", ["full", "partial"])
@pytest.mark.parametrize("norm", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_norm_rope(h, hkv, d, angles, norm, dtype):
    torch.manual_seed(d)
    t = 300
    proj = torch.randn(t, (h + 2 * hkv) * d, device=DEV, dtype=dtype) * 3
    q = proj[:, : h * d].view(t, h, d)
    k = proj[:, h * d : (h + hkv) * d].view(t, hkv, d)
    wq = torch.rand(d, device=DEV, dtype=dtype) + 0.5
    wk = torch.rand(d, device=DEV, dtype=dtype) + 0.5
    rope = Rope(d, d // 2 if angles == "full" else d // 4, 1e6, DEV)
    step = Step(torch.arange(t, device=DEV) * 13, *(None,) * 5)
    rq, rk = (torch_norm(q, wq, 1e-5), torch_norm(k, wk, 1e-6)) if norm else (q, k)
    rq, rk = rope(step, rq, rk)
    cos, sin = rope.tables(step, dtype)
    gq, gk = qk.norm_rope(
        q, k, wq if norm else None, wk if norm else None, 1e-5, 1e-6, cos[:, 0], sin[:, 0]
    )
    for got, ref in ((gq, rq), (gk, rk)):
        flips = (got != ref).float().mean().item()
        assert flips < (2e-3 if norm else 1e-9), flips
