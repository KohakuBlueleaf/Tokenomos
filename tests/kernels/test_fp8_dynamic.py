"""Dynamic FP8: scales normalized from each checkpoint granularity, the quantized activation against
E4M3's rounding bound, the GEMM against a float64 product of the same codes and scales."""

import pytest
import torch

from tokenomos.kernels.linear import fp8
from tokenomos.models.load.checkpoint import QuantScheme

DEV = "cuda"
SHAPES = [(1, 256, 2048), (7, 300, 512), (33, 4096, 2048), (130, 200, 300), (257, 1024, 1024)]
# E4M3 keeps 3 mantissa bits: half a step at the top of a binade over the value
ROUNDING = 1 / 16
# below 2^-6 the codes step by 2^-9: half a step, in units of the group's scale
SUBNORMAL = 2.0**-10


def test_float_quantized_is_read():
    cfg = {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {"g": {"targets": ["Linear"]}},
        "ignore": ["lm_head"],
    }
    scheme = QuantScheme(cfg)
    assert scheme.format_of("model.layers.0.w1") == "fp8_dynamic"
    assert scheme.format_of("lm_head") == "bf16"


def test_scales_normalize():
    n, k = 300, 512
    one = fp8.expand_scales(torch.tensor(2.0), n, k)
    channel = fp8.expand_scales(torch.arange(n, dtype=torch.float32)[:, None], n, k)
    block = fp8.expand_scales(torch.arange(12.0).view(3, 4), n, k)
    assert one.shape == channel.shape == block.shape == (n, 4)
    assert (one == 2).all()
    assert torch.equal(channel[7], torch.full((4,), 7.0))
    assert torch.equal(block[129], torch.tensor([4.0, 5.0, 6.0, 7.0]))
    assert torch.equal(block[299], torch.tensor([8.0, 9.0, 10.0, 11.0]))
    with pytest.raises(ValueError, match="FP8 weight scale"):
        fp8.expand_scales(torch.ones(2, 2), n, k)


@pytest.mark.gpu
@pytest.mark.fp8
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_quant_holds_each_group(m, n, k):
    x = (torch.randn(m, k + 8, device=DEV) * 3)[:, 4:-4]
    q, s = fp8.dynamic_quant(x)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (m, -(-k // fp8.GROUP))
    sc = s.double().repeat_interleave(fp8.GROUP, 1)[:, :k]
    back = q.double() * sc
    assert ((back - x.double()).abs() <= ROUNDING * x.double().abs() + SUBNORMAL * sc).all()
    # the row's largest value in each group lands on the top code
    assert q.float().abs().amax(1).max() == fp8.E4M3_MAX


@pytest.mark.gpu
@pytest.mark.fp8
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_gemm_against_double(m, n, k, with_bias):
    x = torch.randn(m, k, device=DEV)
    w = torch.randn(n, k, device=DEV)
    kb = -(-k // fp8.GROUP)
    # every row a scale of its own per group, as fused parts hold them
    ws = (torch.rand(n, kb, device=DEV) + 0.5) / fp8.E4M3_MAX
    code = (w / ws.repeat_interleave(fp8.GROUP, 1)[:, :k]).clamp(-448, 448)
    code = code.to(torch.float8_e4m3fn)
    bias = torch.randn(n, device=DEV, dtype=torch.bfloat16) if with_bias else None
    q, s = fp8.dynamic_quant(x)
    out = fp8.dynamic_gemm(q, s, code, ws, bias)
    a = q.double() * s.double().repeat_interleave(fp8.GROUP, 1)[:, :k]
    b = code.double() * ws.double().repeat_interleave(fp8.GROUP, 1)[:, :k]
    ref = a @ b.t() + (bias.double() if with_bias else 0)
    assert out.dtype == torch.bfloat16 and out.shape == (m, n)
    # float32 accumulation, one bf16 rounding
    assert (out.double() - ref).abs().max() <= 2.0**-7 * ref.abs().max() + 1e-3
