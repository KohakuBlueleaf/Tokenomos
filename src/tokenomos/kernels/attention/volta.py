"""``paged``'s decode and prefill on Volta tensor cores (TileLang): Triton has no MMA path on sm70,
its ``tl.dot`` runs on vector units there. Same layouts and semantics: fp16 operands, float32
accumulation, scores and output arithmetic, probabilities rounded once to fp16 before P . V.

Tensors reach a kernel flat with explicit row strides (forward tensors may be strided views), the
caches as ``[rows, Hkv, D]``. ``DP``: ``D`` rounded up to a multiple of 16, the padding zeros.
``r_s`` holds a key block's cache rows, one page-table read per key.
"""

from functools import cache

import tilelang
import tilelang.language as T
import torch

from tokenomos.kernels.linear.int8 import arch

ACC = "float32"
MM = "float16"
LOG2E = 1.4426950408889634
NAMES = {torch.float16: "float16", torch.float32: "float32"}
PASSES = {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}
# head dim -> keys per block, threads: a decode tile has 16+ rows, one warp per 16 (FullRow)
DECODE = {64: (32, 32), 128: (32, 32), 256: (32, 32)}
# head dim -> query rows, keys per block, threads (FullRow: 16 rows per warp)
PREFILL = {64: (64, 64, 128), 128: (64, 64, 128), 256: (64, 32, 128)}


def round16(x: int) -> int:
    return -(-x // 16) * 16


def _tile(table: dict, dim: int) -> tuple:
    return table[min((d for d in table if d >= dim), default=max(table))]


def prefill_block(dim: int) -> tuple[int, int]:
    """(BM, BN) of a prefill tile."""
    return _tile(PREFILL, dim)[:2]


def _flat(x: torch.Tensor) -> torch.Tensor:
    """``x`` [N, ...] (row stride free, each row contiguous) as one dimension."""
    row = x[0].numel() if x.shape[0] else 0
    return x.as_strided(((x.shape[0] - 1) * x.stride(0) + row,), (1,))


def _cached(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1, x.shape[-2], x.shape[-1])


def _compile(prim, device: torch.device):
    return tilelang.compile(
        prim, target={"kind": "cuda", "arch": arch(device)}, pass_configs=PASSES
    )


@T.macro
def score_block(
    dots, s, m_i, m_prev, m_safe, alpha, row_sum, l_i, p_s, v_s, tile, acc, gemm_policy
):
    """One online-softmax block over scores ``s`` (masked -inf), a row with none so far at zero."""
    T.copy(m_i, m_prev)
    T.reduce_max(s, m_i, dim=1, clear=False)
    bm, bn = s.shape
    for i in T.Parallel(bm):
        m_safe[i] = T.if_then_else(m_i[i] == -T.infinity(ACC), 0, m_i[i])
        alpha[i] = T.exp2((m_prev[i] - m_safe[i]) * LOG2E)
    for i, j in T.Parallel(bm, bn):
        s[i, j] = T.exp2((s[i, j] - m_safe[i]) * LOG2E)
    T.reduce_sum(s, row_sum, dim=1)
    for i in T.Parallel(bm):
        l_i[i] = l_i[i] * alpha[i] + row_sum[i]
    T.copy(s, p_s)
    T.clear(tile)
    T.gemm(p_s, v_s, tile, policy=gemm_policy)
    dp = tile.shape[1]
    for i, j in T.Parallel(bm, dp):
        acc[i, j] = acc[i, j] * alpha[i] + tile[i, j]


@cache
def _decode_kernel(g, hkv, d, bn, threads, qt, ct, ring, has_window, has_softcap, device):
    """One sequence's query heads of one KV head per program."""
    gp, dp = round16(g), round16(d)
    n, lq, lk, lv, lo_, rows, nt = (
        T.dynamic(x) for x in ("n", "lq", "lk", "lv", "lo", "rows", "nt")
    )
    policy = T.GemmWarpPolicy.FullRow

    @T.prim_func
    def kernel(
        Q: T.Tensor((lq,), qt),
        K: T.Tensor((lk,), qt),
        V: T.Tensor((lv,), qt),
        KC: T.Tensor((rows, hkv, d), ct),
        VC: T.Tensor((rows, hkv, d), ct),
        Out: T.Tensor((lo_,), qt),
        Slot: T.Tensor((n,), "int32"),
        Ctx: T.Tensor((n,), "int32"),
        Table: T.Tensor((nt,), "int32"),
        stride_q: T.int32,
        stride_k: T.int32,
        stride_v: T.int32,
        stride_o: T.int32,
        stride_t: T.int32,
        page: T.int32,
        ring_len: T.int32,
        window: T.int32,
        scale: T.float32,
        softcap: T.float32,
    ):
        with T.Kernel(n, hkv, threads=threads) as (row, h):
            slot = Slot[row]
            pos = Ctx[row]
            lo = T.max(pos - window + 1, 0) if has_window else 0
            q_s = T.alloc_shared((gp, dp), MM)
            k_s = T.alloc_shared((bn, dp), MM)
            v_s = T.alloc_shared((bn, dp), MM)
            p_s = T.alloc_shared((gp, bn), MM)
            r_s = T.alloc_shared((bn,), "int32")
            qf = T.alloc_fragment((gp, dp), ACC)
            dots = T.alloc_fragment((gp, bn), ACC)
            s = T.alloc_fragment((gp, bn), ACC)
            tile = T.alloc_fragment((gp, dp), ACC)
            acc = T.alloc_fragment((gp, dp), ACC)
            prod = T.alloc_fragment((gp, dp), ACC)
            sn = T.alloc_fragment((gp,), ACC)
            m_i = T.alloc_fragment((gp,), ACC)
            m_prev = T.alloc_fragment((gp,), ACC)
            m_safe = T.alloc_fragment((gp,), ACC)
            alpha = T.alloc_fragment((gp,), ACC)
            row_sum = T.alloc_fragment((gp,), ACC)
            l_i = T.alloc_fragment((gp,), ACC)

            for i, j in T.Parallel(gp, dp):
                qf[i, j] = T.if_then_else(
                    T.And(i < g, j < d), Q[row * stride_q + (h * g + i) * d + j], 0
                )
            T.copy(qf, q_s)
            T.fill(acc, 0)
            T.fill(l_i, 0)
            T.fill(m_i, -T.infinity(ACC))
            for kb in T.serial(T.ceildiv(pos - lo, bn)):
                for i in T.Parallel(bn):
                    p = lo + kb * bn + i
                    if ring:
                        r_s[i] = slot * ring_len + p % ring_len
                    else:
                        r_s[i] = T.if_then_else(
                            p < pos, Table[slot * stride_t + p // page] * page + p % page, 0
                        )
                for i, j in T.Parallel(bn, dp):
                    live = T.And(lo + kb * bn + i < pos, j < d)
                    k_s[i, j] = T.if_then_else(live, KC[r_s[i], h, j], 0)
                    v_s[i, j] = T.if_then_else(live, VC[r_s[i], h, j], 0)
                T.clear(dots)
                T.gemm(q_s, k_s, dots, transpose_B=True, policy=policy)
                for i, j in T.Parallel(gp, bn):
                    sc = dots[i, j] * scale
                    if has_softcap:
                        sc = softcap * T.tanh(sc / softcap)
                    s[i, j] = T.if_then_else(lo + kb * bn + j < pos, sc, -T.infinity(ACC))
                score_block(
                    dots, s, m_i, m_prev, m_safe, alpha, row_sum, l_i, p_s, v_s, tile, acc, policy
                )
            for i, j in T.Parallel(gp, dp):
                kn = T.if_then_else(j < d, K[row * stride_k + h * d + j], 0)
                prod[i, j] = qf[i, j] * kn.astype(ACC)
            T.reduce_sum(prod, sn, dim=1)
            for i in T.Parallel(gp):
                sc = sn[i] * scale
                if has_softcap:
                    sc = softcap * T.tanh(sc / softcap)
                sn[i] = sc
                m_safe[i] = T.max(m_i[i], sn[i])
                alpha[i] = T.exp2((m_i[i] - m_safe[i]) * LOG2E)
                sn[i] = T.exp2((sn[i] - m_safe[i]) * LOG2E)
                l_i[i] = l_i[i] * alpha[i] + sn[i]
            for i, j in T.Parallel(gp, dp):
                vn = T.if_then_else(j < d, V[row * stride_v + h * d + j], 0)
                o = (acc[i, j] * alpha[i] + sn[i] * vn.astype(ACC)) / l_i[i]
                if i < g and j < d:
                    Out[row * stride_o + (h * g + i) * d + j] = o.astype(qt)

    return _compile(kernel, device)


def decode(
    q, k, v, kc, vc, slot, ctx, table, scale, window=None, softcap=None, ring=False, out=None
):
    """As ``paged.decode``."""
    n, h, dim = q.shape
    hkv = k.shape[1]
    out = torch.empty_like(q) if out is None else out
    bn, threads = _tile(DECODE, dim)
    kernel = _decode_kernel(
        h // hkv,
        hkv,
        dim,
        bn,
        threads,
        NAMES[q.dtype],
        NAMES[kc.dtype],
        ring,
        window is not None,
        softcap is not None,
        q.device,
    )
    page, ring_len = (1, kc.shape[1]) if ring else (kc.shape[1], 1)
    kernel(
        _flat(q),
        _flat(k),
        _flat(v),
        _cached(kc),
        _cached(vc),
        _flat(out),
        slot,
        ctx,
        table.view(-1) if table is not None else slot,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        out.stride(0),
        table.stride(0) if table is not None else 0,
        page,
        ring_len,
        window or 0,
        scale,
        softcap or 1.0,
    )
    return out


@cache
def _prefill_kernel(
    gh, hkv, d, bm, bn, threads, qt, ct, ring, has_window, has_softcap, causal, bidir, device
):
    """``BM`` queries of one sequence, one query head per program."""
    dp = round16(d)
    nt, ns, lq, lk, lv, rows, lo_, ltab, ncu, nb = (
        T.dynamic(x) for x in ("nt", "ns", "lq", "lk", "lv", "rows", "lo", "ltab", "ncu", "nb")
    )
    policy = T.GemmWarpPolicy.FullRow

    @T.prim_func
    def kernel(
        Q: T.Tensor((lq,), qt),
        K: T.Tensor((lk,), qt),
        V: T.Tensor((lv,), qt),
        KC: T.Tensor((rows, hkv, d), ct),
        VC: T.Tensor((rows, hkv, d), ct),
        Out: T.Tensor((lo_,), qt),
        TileSeq: T.Tensor((nt,), "int32"),
        TileQ0: T.Tensor((nt,), "int32"),
        Cu: T.Tensor((ncu,), "int32"),
        Slot: T.Tensor((ns,), "int32"),
        Ctx: T.Tensor((ns,), "int32"),
        Table: T.Tensor((ltab,), "int32"),
        Blk: T.Tensor((nb,), "int32"),
        Hi: T.Tensor((nb,), "int32"),
        stride_q: T.int32,
        stride_k: T.int32,
        stride_v: T.int32,
        stride_o: T.int32,
        stride_t: T.int32,
        page: T.int32,
        ring_len: T.int32,
        window: T.int32,
        scale: T.float32,
        softcap: T.float32,
    ):
        with T.Kernel(nt, hkv * gh, threads=threads) as (tile_id, hq):
            hk = hq // gh
            sq = TileSeq[tile_id]
            q0 = TileQ0[tile_id]
            a = Cu[sq]
            b = Cu[sq + 1]
            slot = Slot[sq]
            ctx = Ctx[sq]
            first = ctx + (q0 - a)
            last = ctx + (b - a) - 1
            if causal:
                reach = ctx + (q0 + bm - 1 - a)
                last = T.min(reach, last)
            lo = T.max(first - window + 1, 0) if has_window else 0
            flo = T.max(lo, ctx)
            q_s = T.alloc_shared((bm, dp), MM)
            k_s = T.alloc_shared((bn, dp), MM)
            v_s = T.alloc_shared((bn, dp), MM)
            p_s = T.alloc_shared((bm, bn), MM)
            r_s = T.alloc_shared((bn,), "int32")
            s = T.alloc_fragment((bm, bn), ACC)
            dots = T.alloc_fragment((bm, bn), ACC)
            tile = T.alloc_fragment((bm, dp), ACC)
            acc = T.alloc_fragment((bm, dp), ACC)
            m_i = T.alloc_fragment((bm,), ACC)
            m_prev = T.alloc_fragment((bm,), ACC)
            m_safe = T.alloc_fragment((bm,), ACC)
            alpha = T.alloc_fragment((bm,), ACC)
            row_sum = T.alloc_fragment((bm,), ACC)
            l_i = T.alloc_fragment((bm,), ACC)

            for i, j in T.Parallel(bm, dp):
                q_s[i, j] = T.if_then_else(
                    T.And(q0 + i < b, j < d), Q[(q0 + i) * stride_q + hq * d + j], 0
                )
            T.fill(acc, 0)
            T.fill(l_i, 0)
            T.fill(m_i, -T.infinity(ACC))

            for kb in T.serial(T.ceildiv(T.max(ctx - lo, 0), bn)):
                for i in T.Parallel(bn):
                    p = lo + kb * bn + i
                    if ring:
                        r_s[i] = slot * ring_len + p % ring_len
                    else:
                        r_s[i] = T.if_then_else(
                            p < ctx, Table[slot * stride_t + p // page] * page + p % page, 0
                        )
                for i, j in T.Parallel(bn, dp):
                    live = T.And(lo + kb * bn + i < ctx, j < d)
                    k_s[i, j] = T.if_then_else(live, KC[r_s[i], hk, j], 0)
                    v_s[i, j] = T.if_then_else(live, VC[r_s[i], hk, j], 0)
                T.clear(dots)
                T.gemm(q_s, k_s, dots, transpose_B=True, policy=policy)
                for i, j in T.Parallel(bm, bn):
                    p = lo + kb * bn + j
                    qp = first + i
                    sc = dots[i, j] * scale
                    if has_softcap:
                        sc = softcap * T.tanh(sc / softcap)
                    ok = T.And(p < ctx, q0 + i < b)
                    if has_window:
                        ok = T.And(ok, qp - p < window)
                    s[i, j] = T.if_then_else(ok, sc, -T.infinity(ACC))
                score_block(
                    dots, s, m_i, m_prev, m_safe, alpha, row_sum, l_i, p_s, v_s, tile, acc, policy
                )

            for kb in T.serial(T.ceildiv(T.max(last + 1 - flo, 0), bn)):
                for i, j in T.Parallel(bn, dp):
                    p = flo + kb * bn + i
                    ti = a + (p - ctx)
                    live = T.And(p <= last, j < d)
                    k_s[i, j] = T.if_then_else(live, K[ti * stride_k + hk * d + j], 0)
                    v_s[i, j] = T.if_then_else(live, V[ti * stride_v + hk * d + j], 0)
                T.clear(dots)
                T.gemm(q_s, k_s, dots, transpose_B=True, policy=policy)
                for i, j in T.Parallel(bm, bn):
                    p = flo + kb * bn + j
                    qp = first + i
                    sc = dots[i, j] * scale
                    if has_softcap:
                        sc = softcap * T.tanh(sc / softcap)
                    ok = T.And(p <= last, q0 + i < b)
                    if causal:
                        see = p <= qp
                        if bidir:
                            ti = a + (p - ctx)
                            same = T.And(Blk[q0 + i] == Blk[ti], Blk[q0 + i] >= 0)
                            see = T.Or(see, same)
                        ok = T.And(ok, see)
                    if has_window:
                        ok = T.And(ok, qp - p < window)
                    s[i, j] = T.if_then_else(ok, sc, -T.infinity(ACC))
                score_block(
                    dots, s, m_i, m_prev, m_safe, alpha, row_sum, l_i, p_s, v_s, tile, acc, policy
                )

            for i, j in T.Parallel(bm, dp):
                if q0 + i < b and j < d:
                    o = acc[i, j] / T.if_then_else(l_i[i] > 0, l_i[i], 1.0)
                    Out[(q0 + i) * stride_o + hq * d + j] = o.astype(qt)

    return _compile(kernel, device)


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
    """As ``paged.prefill``."""
    _, h, dim = q.shape
    hkv = k.shape[1]
    if kc is k:  # an image tower's forward tensors stand in for a cache it never reads
        kc = vc = k.new_empty(0, hkv, dim)
    out = torch.zeros_like(q)
    bm, bn, threads = _tile(PREFILL, dim)
    kernel = _prefill_kernel(
        h // hkv,
        hkv,
        dim,
        bm,
        bn,
        threads,
        NAMES[q.dtype],
        NAMES[kc.dtype],
        ring,
        window is not None,
        softcap is not None,
        causal,
        blocks is not None,
        q.device,
    )
    page, ring_len = (1, kc.shape[1]) if ring else (kc.shape[1], 1)
    kernel(
        _flat(q),
        _flat(k),
        _flat(v),
        _cached(kc),
        _cached(vc),
        _flat(out),
        tile_seq,
        tile_q0,
        cu,
        slot,
        ctx,
        table.view(-1) if table is not None else slot,
        blocks[0] if blocks is not None else slot,
        blocks[1] if blocks is not None else slot,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        out.stride(0),
        table.stride(0) if table is not None else 0,
        page,
        ring_len,
        window or 0,
        scale,
        softcap or 1.0,
    )
    return out
