"""W8A8 kernels against plain torch references."""

import pytest
import torch

i8 = pytest.importorskip("tokenomos.kernels.linear.int8")

from tokenomos.models.blocks.linear import Linear, regular_hadamard  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


def rotated(x, g):
    k = x.shape[1] // g * g
    out = x.clone()
    h = regular_hadamard(g, x.device)
    out[:, :k] = (x[:, :k].reshape(-1, g) @ h.t()).reshape(x.shape[0], k)
    return out


def test_regular_hadamard_is_orthonormal_with_equal_row_sums():
    for g in (4, 16, 64, 256):
        h = regular_hadamard(g, "cpu")
        assert torch.allclose(h @ h.t(), torch.eye(g), atol=1e-6)
        assert h.sum(1).abs().max() - h.sum(1).abs().min() < 1e-5


@pytest.mark.parametrize("rotate", [0, 4, 16, 64, 256])
@pytest.mark.parametrize(("m", "k"), [(5, 256), (32, 640), (70, 1024), (200, 2048)])
def test_quant_matches_the_reference(m, k, rotate):
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    q, s = i8.quant(x, rotate)
    f = rotated(x.float(), rotate) if rotate else x.float()
    amax = f.abs().amax(1).clamp(min=1e-12) / 127
    assert torch.allclose(s[:m], amax, rtol=1e-3)
    ref = torch.round(f / amax[:, None]).clamp(-127, 127)
    assert (q[:m, :k].float() - ref).abs().max() <= 1
    assert q.shape == (i8.rows(m), i8.pad_k(k))
    assert (q[m:] == 0).all() and (q[:, k:] == 0).all()


@pytest.mark.parametrize(("m", "n", "k"), [(32, 128, 256), (64, 512, 1024), (128, 192, 640)])
def test_gemm_is_exact(m, n, k):
    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device=DEV)
    b = torch.randint(-127, 128, (n, k), dtype=torch.int8, device=DEV)
    out = i8.gemm(a, b)
    assert (out.double() == a.double() @ b.double().t()).all()


@pytest.mark.parametrize("rotate", [0, 16])
@pytest.mark.parametrize(("m", "n", "k"), [(7, 300, 512), (100, 1024, 1024), (130, 77, 640)])
def test_a_linear_stays_near_its_float_product(m, n, k, rotate):
    w = torch.randn(n, k, device=DEV).to(torch.bfloat16)
    bias = torch.randn(n, device=DEV).to(torch.bfloat16)
    x = torch.randn(m, k, device=DEV).to(torch.bfloat16)
    ref = x.float() @ w.float().t() + bias.float()
    lin = Linear("bf16", w.clone(), bias=bias)
    lin.to_int8(rotate)
    out = lin(x)
    assert out.dtype == torch.bfloat16
    err = (out.float() - ref).norm() / (ref - bias.float()).norm()
    assert err < 0.02
