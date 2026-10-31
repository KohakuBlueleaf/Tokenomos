"""Attention over a KV cache plus the tokens of the current forward, from device metadata only.

Layouts: paged ``[pages, P, Hkv, D]``, position ``p`` of slot ``s`` on page ``table[s, p // P]``
row ``p % P`` (page 0 null); ring ``[slots, R, Hkv, D]``, position ``p`` at row ``p % R`` (a
sliding window ``W <= R``; slot 0 null).

A sequence has ``ctx`` positions in the cache and brings new tokens (positions ``ctx ..``) whose
K / V come from the forward's own tensors, so a chunk never reads what it is about to overwrite.
``decode``: one new token per sequence, the query heads of a KV head in one program.
``prefill``: tiles of ``BM`` queries of one sequence, one query head per program. ``write``: the
new K / V into the cache after attention (on a ring, the chunk's last ``R`` only). Optional
score softcap ``c * tanh(s / c)``; the window keeps keys ``j`` with ``i - j < W``. Prefill
``blocks`` (per token a run id, -1 outside a run, and the run's last position): a token of a run
also sees the run's later tokens (Gemma 4's bidirectional image runs; a run never spans the
cache and the forward's own tokens). ``MM``: the operand dtype of the tensor-core products (fp16
for fp32 tensors, float32 accumulation either way).
"""

from functools import cache

import torch
import triton
import triton.language as tl

from tokenomos.core.floats import matmul, tl_dtype


@cache
def _half_tiles() -> int:
    """1 where shared memory is 64 KiB (Turing), half of what sm80 and later hold; 0 elsewhere."""
    return int(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8)


def _stages(dtype: torch.dtype) -> int:
    """Triton's three, except that it would pipeline fp32 loads through shared memory unconverted
    (past the 100 KiB of a tile on sm86)."""
    return 3 if matmul(dtype) == dtype else 1


@triton.jit
def _rows(table, slot, p, stride_t, P: tl.constexpr, R: tl.constexpr, RING: tl.constexpr):
    """Cache row index (into ``[rows, Hkv, D]``) of positions ``p`` of slot ``slot``."""
    if RING:
        return slot.to(tl.int64) * R + p % R
    page = tl.load(table + slot.to(tl.int64) * stride_t + p // P)
    return page.to(tl.int64) * P + p % P


@triton.jit
def _score(s, scale, softcap, SOFTCAP: tl.constexpr):
    s = s * scale
    if SOFTCAP:
        s = softcap * (2.0 * tl.sigmoid(2.0 * s / softcap) - 1.0)
    return s


@triton.jit
def _online(m_i, l_i, acc, sc, vt):
    """One online-softmax block; a row with every score masked so far stays at zero."""
    m_new = tl.maximum(m_i, tl.max(sc, 1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp(m_i - m_safe)
    pr = tl.exp(sc - m_safe[:, None])
    l_i = l_i * alpha + tl.sum(pr, 1)
    acc = acc * alpha[:, None] + tl.dot(pr.to(vt.dtype), vt)
    return m_new, l_i, acc


@triton.jit
def _decode_kernel(
    q,
    k,
    v,
    kc,
    vc,
    out,
    slot_ptr,
    ctx_ptr,
    table,
    scale,
    softcap,
    window,
    stride_qt,
    stride_kt,
    stride_vt,
    stride_ot,
    stride_t,
    G: tl.constexpr,
    GP: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    RING: tl.constexpr,
    WINDOW: tl.constexpr,
    SOFTCAP: tl.constexpr,
    MM: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(slot_ptr + row)
    pos = tl.load(ctx_ptr + row)
    g = tl.arange(0, GP)
    d = tl.arange(0, D)
    gm = g < G
    qv = tl.load(
        q + row * stride_qt + (h * G + g)[:, None] * D + d[None, :], mask=gm[:, None], other=0.0
    )
    lo = 0
    if WINDOW:
        lo = tl.maximum(pos - window + 1, 0)
    m_i = tl.full((GP,), float("-inf"), tl.float32)
    l_i = tl.zeros((GP,), tl.float32)
    acc = tl.zeros((GP, D), tl.float32)
    qm = qv.to(MM)
    for start in range(lo, pos, BN):
        p = start + tl.arange(0, BN)
        pm = p < pos
        rows = _rows(table, slot, tl.where(pm, p, 0), stride_t, P, R, RING)
        off = (rows * HKV + h)[:, None] * D + d[None, :]
        kt = tl.load(kc + off, mask=pm[:, None], other=0.0).to(MM)
        s = _score(tl.dot(qm, tl.trans(kt)), scale, softcap, SOFTCAP)
        s = tl.where(pm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        pr = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(pr, 1)
        vt = tl.load(vc + off, mask=pm[:, None], other=0.0).to(MM)
        acc = acc * alpha[:, None] + tl.dot(pr.to(vt.dtype), vt)
        m_i = m_new
    kn = tl.load(k + row * stride_kt + h * D + d).to(tl.float32)
    vn = tl.load(v + row * stride_vt + h * D + d).to(tl.float32)
    sn = _score(tl.sum(qv.to(tl.float32) * kn[None, :], 1), scale, softcap, SOFTCAP)
    m_new = tl.maximum(m_i, sn)
    alpha = tl.exp(m_i - m_new)
    pn = tl.exp(sn - m_new)
    l_i = l_i * alpha + pn
    acc = acc * alpha[:, None] + pn[:, None] * vn[None, :]
    o = acc / l_i[:, None]
    tl.store(
        out + row * stride_ot + (h * G + g)[:, None] * D + d[None, :],
        o.to(out.dtype.element_ty),
        mask=gm[:, None],
    )


@triton.jit
def _prefill_kernel(
    q,
    k,
    v,
    kc,
    vc,
    out,
    tile_seq,
    tile_q0,
    cu,
    slot_ptr,
    ctx_ptr,
    table,
    blk_ptr,
    hi_ptr,
    scale,
    softcap,
    window,
    stride_qt,
    stride_kt,
    stride_vt,
    stride_ot,
    stride_t,
    G: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    RING: tl.constexpr,
    WINDOW: tl.constexpr,
    SOFTCAP: tl.constexpr,
    CAUSAL: tl.constexpr,
    BIDIR: tl.constexpr,
    MM: tl.constexpr,
):
    """``DP``: ``D`` rounded up to a power of two (the padded columns load as zeros)."""
    tile = tl.program_id(0)
    hq = tl.program_id(1)
    hk = hq // G
    s = tl.load(tile_seq + tile)
    q0 = tl.load(tile_q0 + tile)
    a = tl.load(cu + s)
    b = tl.load(cu + s + 1)
    slot = tl.load(slot_ptr + s)
    ctx = tl.load(ctx_ptr + s)
    d = tl.arange(0, DP)
    dm = d < D
    qi = q0 + tl.arange(0, BM)
    qm = qi < b
    qpos = ctx + (qi - a)
    qv = tl.load(
        q + qi[:, None] * stride_qt + hq * D + d[None, :], mask=qm[:, None] & dm[None, :], other=0.0
    ).to(MM)
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, DP), tl.float32)
    first = ctx + (q0 - a)
    last = ctx + (b - a) - 1
    bq = tl.full((BM,), -1, tl.int32)
    if BIDIR:
        bq = tl.load(blk_ptr + qi, mask=qm, other=-1)
    if CAUSAL:
        reach = ctx + (q0 + BM - 1 - a)
        if BIDIR:
            reach = tl.maximum(reach, tl.max(tl.load(hi_ptr + qi, mask=qm, other=0)))
        last = tl.minimum(reach, last)
    lo = 0
    if WINDOW:
        lo = tl.maximum(first - window + 1, 0)
    # cached positions [lo, ctx)
    for start in range(lo, ctx, BN):
        p = start + tl.arange(0, BN)
        pm = p < ctx
        rows = _rows(table, slot, tl.where(pm, p, 0), stride_t, P, R, RING)
        off = (rows * HKV + hk)[:, None] * D + d[None, :]
        km = pm[:, None] & dm[None, :]
        kt = tl.load(kc + off, mask=km, other=0.0).to(MM)
        sc = _score(tl.dot(qv, tl.trans(kt)), scale, softcap, SOFTCAP)
        ok = pm[None, :] & qm[:, None]
        if WINDOW:
            ok = ok & (qpos[:, None] - p[None, :] < window)
        sc = tl.where(ok, sc, float("-inf"))
        vt = tl.load(vc + off, mask=km, other=0.0).to(MM)
        m_i, l_i, acc = _online(m_i, l_i, acc, sc, vt)
    # this forward's own tokens [max(lo, ctx), last]
    flo = tl.maximum(lo, ctx)
    for start in range(flo, last + 1, BN):
        p = start + tl.arange(0, BN)
        pm = p <= last
        ti = a + (p - ctx)
        km = pm[:, None] & dm[None, :]
        kt = tl.load(k + ti[:, None] * stride_kt + hk * D + d[None, :], mask=km, other=0.0).to(MM)
        sc = _score(tl.dot(qv, tl.trans(kt)), scale, softcap, SOFTCAP)
        ok = pm[None, :] & qm[:, None]
        if CAUSAL:
            see = p[None, :] <= qpos[:, None]
            if BIDIR:
                bk = tl.load(blk_ptr + ti, mask=pm, other=-1)
                see = see | ((bq[:, None] == bk[None, :]) & (bq[:, None] >= 0))
            ok = ok & see
        if WINDOW:
            ok = ok & (qpos[:, None] - p[None, :] < window)
        sc = tl.where(ok, sc, float("-inf"))
        vt = tl.load(v + ti[:, None] * stride_vt + hk * D + d[None, :], mask=km, other=0.0).to(MM)
        m_i, l_i, acc = _online(m_i, l_i, acc, sc, vt)
    o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    tl.store(
        out + qi[:, None] * stride_ot + hq * D + d[None, :],
        o.to(out.dtype.element_ty),
        mask=qm[:, None] & dm[None, :],
    )


@triton.jit
def _write_kernel(
    k,
    v,
    kc,
    vc,
    tok_seq,
    cu,
    slot_ptr,
    ctx_ptr,
    n_tok,
    table,
    stride_kt,
    stride_vt,
    stride_t,
    HKV: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    RING: tl.constexpr,
    SAME: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    if t < tl.load(n_tok):
        s = tl.load(tok_seq + t)
        a = tl.load(cu + s)
        b = tl.load(cu + s + 1)
        ctx = tl.load(ctx_ptr + s)
        keep = True
        if RING:
            # a later token of the chunk takes the same ring row
            keep = t >= b - R
        if keep:
            slot = tl.load(slot_ptr + s)
            p = ctx + (t - a)
            row = _rows(table, slot, p, stride_t, P, R, RING)
            d = tl.arange(0, D)
            off = (row * HKV + h) * D + d
            tl.store(kc + off, tl.load(k + t * stride_kt + h * D + d))
            if not SAME:
                tl.store(vc + off, tl.load(v + t * stride_vt + h * D + d))


def _layout(cache: torch.Tensor, ring: bool) -> tuple[int, int]:
    """(P, R): page size or ring length."""
    return (1, cache.shape[1]) if ring else (cache.shape[1], 1)


def decode(
    q, k, v, kc, vc, slot, ctx, table, scale, window=None, softcap=None, ring=False, out=None
):
    """``q`` [N, H, D], ``k`` / ``v`` [N, Hkv, D] (one new token per sequence; rows may be
    strided views), ``slot`` / ``ctx`` [N] -> [N, H, D]."""
    n, h, dim = q.shape
    hkv = k.shape[1]
    g = h // hkv
    out = torch.empty_like(q) if out is None else out
    p, r = _layout(kc, ring)
    _decode_kernel[(n, hkv)](
        q,
        k,
        v,
        kc,
        vc,
        out,
        slot,
        ctx,
        table if table is not None else slot,
        scale,
        softcap or 1.0,
        window or 0,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        out.stride(0),
        table.stride(0) if table is not None else 0,
        G=g,
        GP=max(16, triton.next_power_of_2(g)),
        HKV=hkv,
        D=dim,
        BN=(64 if dim <= 128 else 32) >> _half_tiles(),
        P=p,
        R=r,
        RING=ring,
        WINDOW=window is not None,
        SOFTCAP=softcap is not None,
        MM=tl_dtype(matmul(q.dtype)),
        num_warps=4,
        num_stages=_stages(q.dtype),
    )
    return out


def tiles(cu_host: list[int], bm: int) -> tuple[list[int], list[int]]:
    """Query tiles of ``bm`` tokens, none crossing a sequence: (sequence, first token) each."""
    seq, q0 = [], []
    for s in range(len(cu_host) - 1):
        for t in range(cu_host[s], cu_host[s + 1], bm):
            seq.append(s)
            q0.append(t)
    return seq, q0


def prefill_block(dim: int) -> tuple[int, int]:
    """(BM, BN): the tiles' shared memory is about ``(2 * BM + BN) * dim`` float32 values."""
    block = 64 if dim <= 128 else 32 if dim <= 256 else 16
    block = max(block >> _half_tiles(), 16)  # tl.dot needs K >= 16
    return block, block


def prefill(
    q,
    k,
    v,
    kc,
    vc,
    cu,
    slot,
    ctx,
    table,
    tile_seq,
    tile_q0,
    scale,
    window=None,
    softcap=None,
    ring=False,
    causal=True,
    blocks=None,
):
    """``q`` [T, H, D], ``k`` / ``v`` [T, Hkv, D] packed by ``cu``; ``slot`` / ``ctx`` [S];
    ``tile_seq`` / ``tile_q0`` from ``tiles`` -> [T, H, D]. ``causal`` False: every token of a
    sequence sees all of it (an image encoder's full attention). ``blocks`` (run id [T], run's
    last position [T]): bidirectional runs within the causal mask."""
    t, h, dim = q.shape
    hkv = k.shape[1]
    out = torch.zeros_like(q)
    p, r = _layout(kc, ring)
    bm, bn = prefill_block(dim)
    _prefill_kernel[(tile_seq.shape[0], h)](
        q,
        k,
        v,
        kc,
        vc,
        out,
        tile_seq,
        tile_q0,
        cu,
        slot,
        ctx,
        table if table is not None else slot,
        blocks[0] if blocks is not None else slot,
        blocks[1] if blocks is not None else slot,
        scale,
        softcap or 1.0,
        window or 0,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        out.stride(0),
        table.stride(0) if table is not None else 0,
        G=h // hkv,
        HKV=hkv,
        D=dim,
        DP=triton.next_power_of_2(dim),
        BM=bm,
        BN=bn,
        P=p,
        R=r,
        RING=ring,
        WINDOW=window is not None,
        SOFTCAP=softcap is not None,
        CAUSAL=causal,
        BIDIR=blocks is not None,
        MM=tl_dtype(matmul(q.dtype)),
        num_warps=4,
        num_stages=_stages(q.dtype),
    )
    return out


def write(k, v, kc, vc, tok_seq, cu, slot, ctx, n_tok, table, ring=False):
    """The forward's new tokens' K / V into the cache (``vc is kc``: K = V, one tensor)."""
    t, hkv, dim = k.shape
    p, r = _layout(kc, ring)
    _write_kernel[(t, hkv)](
        k,
        v,
        kc,
        vc,
        tok_seq,
        cu,
        slot,
        ctx,
        n_tok,
        table if table is not None else slot,
        k.stride(0),
        v.stride(0),
        table.stride(0) if table is not None else 0,
        HKV=hkv,
        D=dim,
        P=p,
        R=r,
        RING=ring,
        SAME=vc is kc,
        num_warps=1,
    )
