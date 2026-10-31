"""The fused residual norm against the Hugging Face chain in torch: ``post_norm(x)`` (bf16),
``+ residual`` (bf16), ``norm`` (bf16). Reduction order differs, so a rare bf16 rounding flip is
allowed (a skipped rounding step flips about a quarter of the elements); the stream must be
exact when no norm precedes the add. A skipped post-norm must fail the same check.
"""

import pytest
import torch

from tokenomos.kernels.norm import fused

pytestmark = pytest.mark.gpu
DEV = "cuda"


def hf_norm(x, w, eps):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)


def flips(a, b):
    return (a != b).float().mean().item()


@pytest.mark.parametrize("m", [1, 33, 256])
@pytest.mark.parametrize("k", [128, 2048, 2560, 5376])
@pytest.mark.parametrize("pre", [False, True])
@pytest.mark.parametrize("res", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_add_norm(m, k, pre, res, dtype):
    torch.manual_seed(k + m)
    x = torch.randn(m, k, device=DEV, dtype=dtype) * 4
    r = torch.randn_like(x) if res else None
    w = torch.rand(k, device=DEV, dtype=dtype) + 0.5
    wp = torch.rand(k, device=DEV, dtype=dtype) + 0.5
    stream = hf_norm(x, wp, 1e-6) if pre else x
    if res:
        stream = stream + r
    out, xq, got = fused.add_norm(x, r, w, 1e-5, (wp, 1e-6) if pre else None)
    assert xq is None
    if pre:
        assert flips(got, stream) < 1e-3
    else:
        assert torch.equal(got, stream)
    assert flips(out, hf_norm(got, w, 1e-5)) < 1e-3


@pytest.mark.bf16
def test_control_pre_matters():
    x = torch.randn(64, 2048, device=DEV, dtype=torch.bfloat16) * 4
    r = torch.randn_like(x)
    w = torch.rand(2048, device=DEV, dtype=torch.bfloat16) + 0.5
    wp = torch.rand(2048, device=DEV, dtype=torch.bfloat16) + 0.5
    _, _, with_pre = fused.add_norm(x, r, w, 1e-5, (wp, 1e-6))
    assert flips(with_pre, x + r) > 0.5
