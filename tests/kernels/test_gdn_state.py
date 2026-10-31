"""GDN decode step in every state format against the gated delta rule in float64 on the
dequantized state; ``load`` / ``store`` round trips against an independent row pick."""

import pytest
import torch
import torch.nn.functional as F

from tokenomos.kernels.gdn import state as gstate

pytestmark = pytest.mark.gpu
DEV = "cuda"
# relative state error after one step (the 8-bit formats requantize)
TOL = {"fp32": 1e-5, "bf16": 1e-2, "int8": 0.02, "fp8": 0.08}


def pool_of(state: torch.Tensor, fmt: str) -> torch.Tensor:
    shape, dtype = gstate.shape(*state.shape[1:], fmt)
    return gstate.quantize(state, torch.empty(state.shape[0], *shape, dtype=dtype, device=DEV), fmt)


@pytest.mark.parametrize(
    "fmt", ["fp32", "bf16", "int8", pytest.param("fp8", marks=pytest.mark.fp8)]
)
def test_decode(fmt):
    n, h, hv, k, v = 5, 4, 8, 128, 128
    slots = torch.tensor([3, 1, 0, 6, 2], dtype=torch.int32, device=DEV)
    pool = pool_of(torch.randn(7, hv, v, k, device=DEV) * 0.2, fmt)
    h0 = gstate.dequantize(pool, k, fmt).double()
    q = torch.randn(n, h * k, device=DEV, dtype=torch.bfloat16)
    kk = torch.randn(n, h * k, device=DEV, dtype=torch.bfloat16)
    vv = torch.randn(n, hv * v, device=DEV, dtype=torch.bfloat16)
    a = torch.randn(n, hv, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(n, hv, device=DEV, dtype=torch.bfloat16)
    a_log = torch.randn(hv, device=DEV)
    dt_bias = torch.randn(hv, device=DEV)
    out = torch.empty(n, hv * v, device=DEV, dtype=torch.bfloat16)
    scale = k**-0.5
    gstate.decode(q, kk, vv, a, b, a_log, dt_bias, pool, slots, out, scale, True, fmt)

    after = gstate.dequantize(pool, k, fmt).double()
    untouched = [r for r in range(7) if r not in [s for s in slots.tolist() if s > 0]]
    assert torch.equal(after[untouched], h0[untouched])
    for i in range(n):
        row = int(slots[i])
        if row <= 0:
            assert torch.equal(out[i], torch.zeros_like(out[i]))
            continue
        for j in range(hv):
            hk = j // (hv // h)
            qv = F.normalize(q[i].double().view(h, k)[hk], dim=0, eps=0) * scale
            kv = F.normalize(kk[i].double().view(h, k)[hk], dim=0, eps=0)
            vj = vv[i].double().view(hv, v)[j]
            g = -a_log[j].double().exp() * F.softplus(a[i, j].double() + dt_bias[j].double())
            beta = torch.sigmoid(b[i, j].double())
            st = h0[row, j] * g.exp()
            delta = (vj - st @ kv) * beta
            st = st + delta[:, None] * kv[None, :]
            ref = st @ qv
            got = out[i].double().view(hv, v)[j]
            assert ((got - ref).norm() / ref.norm()).item() < 2e-2
            assert ((after[row, j] - st).norm() / st.norm()).item() < TOL[fmt]


@pytest.mark.parametrize("fmt", ["fp32", "bf16", "int8", "fp8"])
def test_load_store(fmt):
    hv, v, k = 4, 16, 64
    pool = pool_of(torch.randn(6, hv, v, k, device=DEV), fmt)
    full = gstate.dequantize(pool, k, fmt)
    slots = torch.tensor([4, 2, 5], dtype=torch.int32, device=DEV)
    init = torch.tensor([1, 0, 1], dtype=torch.int32, device=DEV)
    rows = gstate.load(pool, slots, init, k, fmt)
    assert torch.equal(rows[0], full[4]) and torch.equal(rows[2], full[5])
    assert not rows[1].any()
    new = torch.randn(3, hv, v, k, device=DEV)
    gstate.store(pool, slots, new, fmt)
    back = gstate.dequantize(pool, k, fmt)
    for i, s in enumerate((4, 2, 5)):
        assert ((back[s] - new[i]).norm() / new[i].norm()).item() < TOL[fmt] + 1e-3
    for s in (0, 1, 3):
        assert torch.equal(back[s], full[s])
