"""Sampler against exact references.

Greedy is the first maximum. With temperature the empirical distribution over many seeds must
match the softmax restricted to the allowed set (``top_k`` / ``top_p`` / ``min_p`` computed in
float64 on the host) within a total-variation bound, and never leave that set. A row samples the
same token alone or inside a larger batch at another row index.
"""

import pytest
import torch

from tokenomos.kernels.sampling.sample import sample

pytestmark = pytest.mark.gpu
DEV = "cuda"


def params(n, temp=0.0, top_p=1.0, top_k=0, min_p=0.0, seed0=0):
    s1 = n + 1
    return dict(
        temp=torch.full((s1,), temp, device=DEV),
        top_p=torch.full((s1,), top_p, device=DEV),
        top_k=torch.full((s1,), top_k, dtype=torch.int32, device=DEV),
        min_p=torch.full((s1,), min_p, device=DEV),
        seed=torch.arange(seed0, seed0 + s1, dtype=torch.int32, device=DEV),
        step=torch.zeros(s1, dtype=torch.int32, device=DEV),
    )


@pytest.mark.parametrize("v", [7, 2048, 65537, 151936])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_greedy(v, dtype):
    torch.manual_seed(0)
    n = 64
    logits = torch.randn(n, v, device=DEV).to(dtype)
    logits[3, :] = 1.0
    logits[5, v // 2 :] = 9.0
    slot = torch.arange(1, n + 1, dtype=torch.int32, device=DEV)
    out = sample(logits, slot, **params(n))
    ref = logits.float().argmax(-1)
    assert torch.equal(out.long(), ref)
    assert out[3] == 0 and out[5] == v // 2


def allowed(x, top_k, top_p, min_p):
    """Allowed set of scaled logits ``x`` [V] (float64)."""
    keep = torch.ones_like(x, dtype=torch.bool)
    if min_p > 0:
        keep &= x >= x.max() + torch.log(torch.tensor(min_p, dtype=x.dtype))
    if top_k > 0:
        keep &= x >= x.sort(descending=True).values[top_k - 1]
    if top_p < 1:
        p = torch.where(keep, x, float("-inf")).softmax(-1)
        sp, idx = p.sort(descending=True)
        csum = sp.cumsum(0)
        k = int((csum < top_p).sum()) + 1
        sel = torch.zeros_like(keep)
        sel[idx[:k]] = True
        keep &= sel
    return keep


@pytest.mark.parametrize(
    "temp,top_k,top_p,min_p",
    [
        (1.0, 0, 1.0, 0.0),
        (0.7, 0, 1.0, 0.0),
        (1.0, 5, 1.0, 0.0),
        (1.0, 0, 0.8, 0.0),
        (1.0, 0, 1.0, 0.05),
        (0.8, 10, 0.9, 0.02),
    ],
)
def test_distribution(temp, top_k, top_p, min_p):
    torch.manual_seed(1)
    v, n = 40, 40000
    base = torch.randn(v, device=DEV, dtype=torch.float64) * 2
    logits = base.float()[None].expand(n, v).contiguous()
    slot = torch.arange(1, n + 1, dtype=torch.int32, device=DEV)
    out = sample(logits, slot, **params(n, temp, top_p, top_k, min_p))
    x = logits[0].double() / temp
    keep = allowed(x, top_k, top_p, min_p)
    assert keep[out.long()].all(), "sampled outside the allowed set"
    want = torch.where(keep, x, float("-inf")).softmax(-1)
    got = torch.bincount(out.long(), minlength=v).double() / n
    tv = 0.5 * (got - want).abs().sum().item()
    assert tv < 0.02, tv


def test_batch_invariance():
    torch.manual_seed(2)
    v = 32000
    logits = torch.randn(300, v, device=DEV)
    p = params(300, temp=1.0, top_p=0.9)
    slot = torch.arange(1, 301, dtype=torch.int32, device=DEV)
    full = sample(logits, slot, **p)
    for r in (0, 17, 299):
        alone = sample(logits[r : r + 1], slot[r : r + 1], **p)
        assert alone[0] == full[r]
    p["step"][1:] = 1
    assert not torch.equal(sample(logits, slot, **p), full), "step must change the draw"
