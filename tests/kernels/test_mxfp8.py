"""MXFP8 producers and GEMM against independent references: values and exponents from a torch
round-up quantizer, swizzled scale bytes placed by the layout formula one byte at a time (and by
vLLM's swizzle when vLLM is installed), the fused inputs computed in torch, the GEMM against a
dequantized float32 product.
"""

import pytest
import torch
import torch.nn.functional as F

from tokenomos.kernels.mxfp8 import ops as mx
from tokenomos.kernels.norm import fused
from tokenomos.models.blocks.linear import Linear

pytestmark = pytest.mark.gpu
DEV = "cuda"


def ref_quant(y: torch.Tensor):
    """float32 [M, K] -> (e4m3 [M, K], biased exponents [M, K / 32] uint8)."""
    m, k = y.shape
    g = y.view(m, k // 32, 32)
    amax = g.abs().amax(-1).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp_min(-127)
    q = (g / torch.exp2(e)[..., None]).to(torch.float8_e4m3fn).view(m, k)
    return q, (e + 127).to(torch.uint8)


def ref_swizzle(e: torch.Tensor) -> torch.Tensor:
    """The layout formula, byte by byte: row r, scale column c of an [M, K/32] table lands at
    tile (r // 128, c // 4) * 512 + (r % 32) * 16 + (r // 32 % 4) * 4 + c % 4."""
    m, kb = e.shape
    rows = -(-m // 128) * 128
    out = torch.zeros(rows * kb, dtype=torch.uint8)
    host = e.cpu()
    for r in range(m):
        for c in range(kb):
            tile = (r // 128) * (kb // 4) + c // 4
            out[tile * 512 + (r % 32) * 16 + (r // 32 % 4) * 4 + c % 4] = host[r, c]
    return out.to(e.device)


def compare(got, y: torch.Tensor):
    m, k = y.shape
    q, e = ref_quant(y)
    assert torch.equal(got[0].view(torch.uint8), q.view(torch.uint8))
    assert got[1].numel() == mx.scale_numel(m, k)
    assert torch.equal(got[1].view(-1), ref_swizzle(e))


@pytest.mark.fp8
@pytest.mark.parametrize("m", [1, 7, 128, 300])
def test_producers(m):
    x = torch.randn(m, 2048, device=DEV, dtype=torch.bfloat16) * 3
    x[0, :32] = 0
    compare(mx.quant(x), x.float())
    b = torch.randn_like(x)
    compare(mx.mul(x, b), x.float() * b.float())
    wide = torch.randn(m, 6144, device=DEV, dtype=torch.bfloat16)
    c = wide[:, 2048:4096]
    compare(mx.mul(c, b), c.float() * b.float())
    h = torch.randn(m, 15360, device=DEV, dtype=torch.bfloat16)
    a, u = h.float().chunk(2, -1)
    compare(mx.silu_mul(h), F.silu(a) * u)
    g = torch.randn(m, 16, device=DEV, dtype=torch.bfloat16)
    gy = x.float().view(m, 16, 128) * torch.sigmoid(g.float())[..., None]
    compare(mx.gate(x, g, 128), gy.view(m, 2048))


def test_norm_quant_rejects_width():
    x = torch.randn(4, 2080, device=DEV, dtype=torch.bfloat16)
    w = torch.ones(2080, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fused.add_norm(x, None, w, 1e-5, quant=True)


@pytest.mark.fp8
@pytest.mark.parametrize("m", [1, 7, 300])
@pytest.mark.parametrize("with_res", [False, True])
@pytest.mark.parametrize("k", [2048, 2560, 4096])
def test_norm_quant(m, with_res, k):
    """The fused norm's quantized output is exactly the producer quantization of its bf16
    output, scales included (2560: a width padded in registers)."""
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    r = torch.randn_like(x) if with_res else None
    w = torch.rand(k, device=DEV, dtype=torch.bfloat16) + 0.5
    out, (q, sc), res = fused.add_norm(x, r, w, 1e-5, quant=True)
    if with_res:
        assert torch.equal(res, x + r)
    compare((q, sc), out.float())
    _, (q2, s2), _ = fused.add_norm(x, r, w, 1e-5, write=False, quant=True)
    assert torch.equal(q2.view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(s2, sc)


def test_swizzle_weight():
    e = torch.randint(100, 140, (300, 64), dtype=torch.uint8, device=DEV)
    assert torch.equal(mx.swizzle(e), ref_swizzle(e))
    utils = pytest.importorskip("vllm.model_executor.layers.quantization.utils.mxfp8_utils")
    assert torch.equal(mx.swizzle(e), utils.swizzle_mxfp8_scale(e, M=300, K=2048).reshape(-1))


def dequant(q, e, m, k):
    return q.float().view(m, k // 32, 32).mul(torch.exp2(e.float() - 127)[..., None]).view(m, k)


def test_dequantized_linear():
    """``Linear.dequantized`` against the reference dequantization, N not a multiple of 128."""
    wt = torch.randn(300, 2048, device=DEV, dtype=torch.bfloat16)
    wq, we = ref_quant(wt.float())
    lin = Linear("mxfp8", wq, mx.swizzle(we))
    assert torch.equal(lin.dequantized().weight, dequant(wq, we, 300, 2048).to(torch.bfloat16))


@pytest.mark.parametrize(
    "n,k",
    [(32, 2048), (96, 2048), (2048, 2048), (3072, 2048), (6144, 2048), (15360, 2048), (2048, 7680)],
)
@pytest.mark.sm100
@pytest.mark.parametrize("m", [1, 37, 256, 512, 1000, 4096])
def test_gemm(n, k, m):
    wt = torch.randn(n, k, device=DEV, dtype=torch.bfloat16) * 0.05
    wq, we = ref_quant(wt.float())
    ws = mx.swizzle(we)
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    q, s = mx.quant(x)
    _, e = ref_quant(x.float())
    ref = dequant(q, e, m, k) @ dequant(wq, we, n, k).t()
    got = mx.gemm(q, s, wq, ws).float()
    err = ((got - ref).norm() / ref.norm()).item()
    assert err < 5e-3, f"{mx.backend(n, k, m)}: rel err {err:.2e}"
