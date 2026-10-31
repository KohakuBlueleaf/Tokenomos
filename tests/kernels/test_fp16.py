"""The fp32 work dtype's kernels against a double-precision reference.

The fp16 matmul: the float32-accumulated product of the rounded activation within float32 error of
the exact one, and its output holding values no 16-bit float can (a 16-bit output fails the bound).
Norms and the skinny linear keep fp32 with no 16-bit rounding step.
"""

import pytest
import torch

from tokenomos.core import floats
from tokenomos.kernels.linear import fp16, int8, skinny
from tokenomos.kernels.norm import fused
from tokenomos.models.blocks.linear import Linear

pytestmark = pytest.mark.gpu
DEV = "cuda"
SHAPES = [(1, 256, 64), (7, 300, 512), (33, 4096, 2048), (130, 2048, 2560), (300, 4096, 2048)]


@pytest.fixture
def fp32():
    floats.set_work("fp32")
    yield
    floats.set_work("bf16")


def bound(x, w, bias=None):
    """Exact product of the fp16-rounded input and the error budget per element. Products of two
    fp16 values are exact in float32, so a sum of ``K`` of them in any order is within
    ``(K - 1) * u * sum |x_i w_i|`` (``u = 2^-24``), plus one rounding of the bias add."""
    h, wd = x.to(torch.float16).double(), w.double()
    ref = h @ wd.t()
    tol = (x.shape[1] - 1) * 2.0**-24 * (h.abs() @ wd.abs().t())
    if bias is not None:
        ref = ref + bias.double()
        tol = tol + 2.0**-24 * ref.abs()
    return ref, tol


def error(out, ref):
    return (out.double() - ref).abs()


@pytest.mark.parametrize("m,n,k", SHAPES)
def test_pack(m, n, k):
    x = torch.randn(m, k, device=DEV) * 3
    r, padk = fp16.rows(m, fp16.block_m(m, DEV)), -(-k // 64) * 64
    q = fp16.pack(x, r, padk)
    assert q.dtype == torch.float16
    assert torch.equal(q[:m, :k], x.half())
    assert not q[m:].any() and not q[:, k:].any()


@pytest.mark.parametrize("m,n,k", SHAPES)
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.usefixtures("fp32")
def test_route_tilelang(m, n, k, with_bias):
    torch.manual_seed(m + n)
    w = torch.randn(n, k, device=DEV).half() * 0.2
    x = torch.randn(m, k + 8, device=DEV)[:, 4:-4]
    bias = torch.randn(n, device=DEV) if with_bias else None
    lin = Linear("bf16", w, bias=bias)
    assert lin.weight.dtype == torch.float16
    lin.set_route("tilelang")
    assert lin.weight.shape[0] % fp16.pad_size(DEV)[0] == 0
    out = lin(x)
    assert out.dtype == torch.float32 and out.shape == (m, n)
    ref, tol = bound(x, w, bias)
    assert (error(out, ref) <= tol).all()
    assert (out != out.half().float()).float().mean() > 0.5
    # float32 accumulation is worth keeping: well under a 16-bit output's rounding
    assert error(out, ref).max() * 8 < error(out.half(), ref).max()


@pytest.mark.usefixtures("fp32")
def test_route_cublas_is_the_16_bit_product():
    w = torch.randn(512, 1024, device=DEV).half() * 0.2
    x = torch.randn(32, 1024, device=DEV)
    lin = Linear("bf16", w)
    lin.set_route("cublas")
    out = lin(x)
    assert out.dtype == torch.float32
    assert torch.equal(out, out.half().float())
    ref, tol = bound(x, w)
    assert (error(out, ref) <= tol * 2.0**13).all()  # the same bound at 16-bit unit roundoff


@pytest.mark.usefixtures("fp32")
def test_set_route_names():
    with pytest.raises(ValueError, match="matmul route"):
        Linear("bf16", torch.ones(4, 4)).set_route("nvcc")


@pytest.mark.parametrize("n", [8, 64])
def test_skinny_fp32_input(n):
    w = torch.randn(n, 1024, device=DEV).half() * 0.2
    x = torch.randn(100, 1032, device=DEV)[:, 8:]
    bias = torch.randn(n, device=DEV)
    out = skinny.linear(x, w, bias)
    assert out.dtype == torch.float32
    ref, tol = bound(x, w, bias)
    assert (error(out, ref) <= tol).all()


def test_add_norm_keeps_fp32():
    x, r = torch.randn(2, 33, 2048, device=DEV).float() * 4
    w = torch.rand(2048, device=DEV) + 0.5
    out, _, res = fused.add_norm(x, r, w, 1e-5)
    h = x.double() + r.double()
    assert (res.double() - h).abs().max() <= 1e-5 * h.abs().max()
    ref = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5) * w.double()
    assert (out.double() - ref).abs().max() <= 1e-5 * ref.abs().max()
    assert (out != out.half().float()).float().mean() > 0.5


def test_int8_scale_writes_fp32():
    q = torch.randint(-1000, 1000, (64, 128), dtype=torch.int32, device=DEV)
    amax, wmul = torch.rand(64, device=DEV), torch.rand(128, device=DEV)
    out = int8.scale(q, 64, 128, amax, wmul, None, torch.float32)
    ref = q.double() * amax.double()[:, None] * wmul.double()[None, :]
    assert out.dtype == torch.float32
    assert (out.double() - ref).abs().max() <= 1e-5 * ref.abs().max()
