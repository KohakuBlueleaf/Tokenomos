"""``round_as`` against torch's casts: every fp32 value rounded to bf16 / fp16 exactly as
``x.to(dtype)`` rounds it (nearest even, halfway cases included), inside an expression whose
rounding is observable (the rounded product plus an addend). For fp32 it is the identity: the
kernel compiles to the same expression without it, whatever the compiler fuses.
"""

import pytest
import torch
import triton
import triton.language as tl

from tokenomos.core import floats
from tokenomos.kernels.rounding import round_as

pytestmark = pytest.mark.gpu
DEV = "cuda"


@triton.jit
def _kernel(a_ptr, b_ptr, o_ptr, n, DTYPE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(a_ptr + i, mask=m)
    b = tl.load(b_ptr + i, mask=m)
    tl.store(o_ptr + i, round_as(a * 1.5, DTYPE) + b, mask=m)


@triton.jit
def _plain_kernel(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(a_ptr + i, mask=m)
    b = tl.load(b_ptr + i, mask=m)
    tl.store(o_ptr + i, a * 1.5 + b, mask=m)


def test_round_as_fp32_is_the_identity():
    torch.manual_seed(0)
    n = 1 << 20
    a = torch.randn(n, device=DEV) * 8
    b = torch.randn(n, device=DEV) * 1e-3
    got, plain = torch.empty_like(a), torch.empty_like(a)
    grid = (triton.cdiv(n, 1024),)
    _kernel[grid](a, b, got, n, DTYPE=floats.TL[torch.float32], BLOCK=1024)
    _plain_kernel[grid](a, b, plain, n, BLOCK=1024)
    assert torch.equal(got, plain)
    exact = a.double() * 1.5 + b.double()
    assert (got.double() - exact).abs().max() <= (exact.abs() * 2.0**-23).max()
    assert (got != got.half().float()).float().mean() > 0.5


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_round_as(dtype):
    torch.manual_seed(0)
    n = 1 << 20
    a = torch.randn(n, device=DEV) * 8
    # exact halfway points between two representable values
    lo = torch.randn(4096, device=DEV).to(dtype)
    a[:4096] = (lo.float() + torch.finfo(dtype).eps * lo.float().abs() / 2) / 1.5
    b = torch.randn(n, device=DEV) * 1e-3
    out = torch.empty_like(a)
    _kernel[(triton.cdiv(n, 1024),)](a, b, out, n, DTYPE=floats.TL[dtype], BLOCK=1024)
    ref = (a * 1.5).to(dtype).float() + b
    assert torch.equal(out, ref)
