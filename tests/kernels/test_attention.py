"""Paged and ring attention against a float32 reference over the whole sequence.

Each case: sequences with ``ctx`` cached positions and ``n`` new tokens, the cached K / V placed
in a paged cache through a shuffled page table (or in a ring), every cache row the kernels must
not read filled with NaN. The output must match softmax attention over positions ``[0, ctx + n)``
with the causal / window mask and softcap; after ``write`` the cache must hold every position a
later step can read.
"""

import random

import pytest
import torch

from tokenomos.core import floats
from tokenomos.kernels.attention import paged

pytestmark = pytest.mark.gpu
DEV = "cuda"
WORK = floats.DTYPES[floats.default(DEV)] if torch.cuda.is_available() else torch.bfloat16


def reference(q, k, v, ctx, window, softcap, scale):
    """``q`` [n, H, D] at positions ctx.., ``k`` / ``v`` [ctx + n, Hkv, D] -> [n, H, D] fp32."""
    n, h, _ = q.shape
    g = h // k.shape[1]
    kf = k.float().repeat_interleave(g, 1).transpose(0, 1)
    vf = v.float().repeat_interleave(g, 1).transpose(0, 1)
    s = (q.float().transpose(0, 1) @ kf.transpose(1, 2)) * scale
    if softcap is not None:
        s = softcap * torch.tanh(s / softcap)
    i = torch.arange(ctx, ctx + n, device=q.device)[:, None]
    j = torch.arange(ctx + n, device=q.device)[None, :]
    ok = j <= i
    if window is not None:
        ok &= i - j < window
    s = s.masked_fill(~ok, float("-inf"))
    return (s.softmax(-1) @ vf).transpose(0, 1)


class Case:
    """Sequences (ctx, n) with full K / V, a cache holding positions < ctx, device metadata."""

    def __init__(self, lens, hkv, dim, ring, window, page=16, same=False):
        self.lens, self.ring, self.same = lens, ring, same
        S = len(lens)
        self.k = [torch.randn(c + n, hkv, dim, device=DEV, dtype=WORK) for c, n in lens]
        self.v = self.k if same else [torch.randn_like(x) for x in self.k]
        slots = list(range(1, S + 1))
        random.shuffle(slots)
        self.slot = torch.tensor(slots, dtype=torch.int32, device=DEV)
        self.ctx = torch.tensor([c for c, _ in lens], dtype=torch.int32, device=DEV)
        nan = float("nan")
        if ring:
            self.R = window + 5
            self.kc = torch.full((S + 1, self.R, hkv, dim), nan, device=DEV, dtype=WORK)
            self.vc = self.kc if same else torch.full_like(self.kc, nan)
            self.table = None
            for (c, _), s, k, v in zip(lens, slots, self.k, self.v):
                for p in range(max(0, c - self.R), c):
                    self.kc[s, p % self.R], self.vc[s, p % self.R] = k[p], v[p]
        else:
            need = [-(-(c + n) // page) for c, n in lens]
            pages = list(range(1, sum(need) + 1))
            random.shuffle(pages)
            shape = (sum(need) + 1, page, hkv, dim)
            self.kc = torch.full(shape, nan, device=DEV, dtype=WORK)
            self.vc = self.kc if same else torch.full_like(self.kc, nan)
            self.table = torch.zeros(S + 1, max(need), dtype=torch.int32, device=DEV)
            for (c, _), s, k, v, m in zip(lens, slots, self.k, self.v, need):
                own = [pages.pop() for _ in range(m)]
                self.table[s, :m] = torch.tensor(own, dtype=torch.int32)
                for p in range(c):
                    self.kc[own[p // page], p % page] = k[p]
                    self.vc[own[p // page], p % page] = v[p]
        self.page = page

    def new(self, t):
        return torch.cat([x[c:] for x, (c, _) in zip(t, self.lens)])

    def cached(self, s, p, cache):
        if self.ring:
            return cache[s, p % self.R]
        return cache[self.table[s, p // self.page], p % self.page]


CONFIGS = [(32, 8, 64), (16, 2, 128), (8, 8, 256)]


@pytest.mark.parametrize("h,hkv,dim", CONFIGS)
@pytest.mark.parametrize("mode", ["full", "window", "softcap", "ring", "same"])
def test_decode(h, hkv, dim, mode):
    torch.manual_seed(0)
    random.seed(0)
    window = 37 if mode in ("window", "ring") else None
    softcap = 30.0 if mode == "softcap" else None
    lens = [(c, 1) for c in (0, 1, 15, 16, 17, 100, 300, 64)]
    case = Case(lens, hkv, dim, mode == "ring", window or 37, same=mode == "same")
    q = torch.randn(len(lens), h, dim, device=DEV, dtype=WORK)
    k, v = case.new(case.k), case.new(case.v)
    scale = dim**-0.5
    out = paged.decode(
        q,
        k,
        v,
        case.kc,
        case.vc,
        case.slot,
        case.ctx,
        case.table,
        scale,
        window,
        softcap,
        case.ring,
    )
    for i, (c, _) in enumerate(lens):
        ref = reference(q[i : i + 1], case.k[i], case.v[i], c, window, softcap, scale)
        torch.testing.assert_close(out[i : i + 1].float(), ref, atol=2e-2, rtol=2e-2)


def check_written(case):
    for i, ((c, n), s) in enumerate(zip(case.lens, case.slot.tolist())):
        lo = max(0, c + n - case.R) if case.ring else 0
        for p in range(lo, c + n):
            assert torch.equal(case.cached(s, p, case.kc), case.k[i][p]), (i, p)
            assert torch.equal(case.cached(s, p, case.vc), case.v[i][p]), (i, p)


@pytest.mark.parametrize(
    "dim,half,block", [(128, 0, 64), (128, 1, 32), (256, 1, 16), (512, 0, 16), (512, 1, 16)]
)
def test_prefill_tiles_fit_shared_memory_and_dot(monkeypatch, dim, half, block):
    monkeypatch.setattr(paged, "_half_tiles", lambda: half)
    monkeypatch.setattr(paged, "_on_volta", lambda: False)
    assert paged.prefill_block(dim) == (block, block)


@pytest.mark.parametrize("h,hkv,dim", CONFIGS)
@pytest.mark.parametrize("mode", ["full", "window", "softcap", "ring", "same"])
def test_prefill_and_write(h, hkv, dim, mode):
    torch.manual_seed(1)
    random.seed(1)
    window = 50 if mode in ("window", "ring") else None
    softcap = 30.0 if mode == "softcap" else None
    lens = [(0, 1), (0, 130), (17, 3), (64, 64), (200, 77), (5, 0), (40, 260)]
    case = Case(lens, hkv, dim, mode == "ring", window or 50, same=mode == "same")
    k, v = case.new(case.k), case.new(case.v)
    T = k.shape[0]
    q = torch.randn(T, h, dim, device=DEV, dtype=WORK)
    cu_host = [0]
    for _, n in lens:
        cu_host.append(cu_host[-1] + n)
    seq, q0 = paged.tiles(cu_host, paged.prefill_block(dim)[0])
    i32 = dict(dtype=torch.int32, device=DEV)
    cu = torch.tensor(cu_host, **i32)
    scale = dim**-0.5
    out = paged.prefill(
        q,
        k,
        v,
        case.kc,
        case.vc,
        cu,
        case.slot,
        case.ctx,
        case.table,
        torch.tensor(seq, **i32),
        torch.tensor(q0, **i32),
        scale,
        window,
        softcap,
        case.ring,
    )
    for i, (c, n) in enumerate(lens):
        a, b = cu_host[i], cu_host[i + 1]
        if n:
            ref = reference(q[a:b], case.k[i], case.v[i], c, window, softcap, scale)
            torch.testing.assert_close(out[a:b].float(), ref, atol=2e-2, rtol=2e-2)
    tok_seq = torch.tensor([i for i, (_, n) in enumerate(lens) for _ in range(n)], **i32)
    paged.write(
        k,
        v,
        case.kc,
        case.vc,
        tok_seq,
        cu,
        case.slot,
        case.ctx,
        torch.tensor([T], **i32),
        case.table,
        case.ring,
    )
    check_written(case)


def test_reads_poisoned_rows_fail():
    """Control: a cached position overwritten with NaN must show in the output."""
    torch.manual_seed(2)
    random.seed(2)
    case = Case([(100, 1)], 8, 64, False, 37)
    case.kc[case.table[case.slot[0], 1], 3] = float("nan")
    q = torch.randn(1, 32, 64, device=DEV, dtype=WORK)
    out = paged.decode(
        q,
        case.new(case.k),
        case.new(case.v),
        case.kc,
        case.vc,
        case.slot,
        case.ctx,
        case.table,
        0.125,
    )
    assert out.isnan().any()
