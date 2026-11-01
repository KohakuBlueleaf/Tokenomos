"""The GDN chunked prefill in three Triton kernels, for every card FlashInfer's does not run on.

Per value head, per chunk of ``BT`` tokens (``c`` the chunk's running product of gates, the
compact WY form of the token recurrence): ``A[t, s] = beta[t] * k[t] @ k[s] * c[t] / c[s]``
(``t > s``), ``T = (I + A) ** -1``, ``u = T @ (beta * (v - c * k @ h))``,
``o = scale * (c * q @ h + (q @ k.T * c[t] / c[s]) @ u)`` (``t >= s``),
``h = c[-1] * h + k.T @ (u * c[-1] / c)``.

``gram`` writes ``A``, the gated ``q @ k.T`` and the running log gates; ``invert`` writes ``T`` in
fp16 tensor-core blocks over 16 x 16 row sweeps; ``advance`` holds ``V / BV`` state rows in
registers over a sequence's chunks and writes the output. Only ``gram`` reads the chunk table.
``config`` picks ``(BT, BV, warps, stages)``: the measured ``tables/<gpu>.json``
(``bench/profile/gdn_kernel.py``), else ``BT`` and ``TILES``.
"""

import functools
import json
from pathlib import Path

import torch
import triton
import triton.language as tl

from tokenomos.core.floats import is_volta, tl_dtype
from tokenomos.kernels.gdn.fla.index import chunk_indices_host
from tokenomos.kernels.mxfp8.ops import gpu_slug

try:
    from tokenomos.kernels.gdn import volta
except ImportError:  # Volta's products need TileLang
    volta = None

BT = 16
BT_SIZES = (16, 32, 64)
BK = 64
TABLES = Path(__file__).parent / "tables"
# compute capability major -> (state rows per program, warps, pipeline stages)
TILES = {7: (32, 4, 2), 8: (64, 4, 2)}


@functools.cache
def config(device: torch.device) -> tuple[int, int, int, int]:
    """``(BT, BV, warps, stages)`` of ``device``."""
    path = TABLES / f"{gpu_slug()}.json"
    if path.exists():
        return tuple(json.loads(path.read_text())["config"])
    return (BT, *TILES[min(torch.cuda.get_device_capability(device)[0], 8)])


def table(cu_host: list[int], bt: int) -> tuple[list[list[int]], list[int]]:
    """``[sequence, chunk]`` rows of chunks of ``bt`` tokens, and each sequence's first row."""
    starts, at = [], 0
    for a, b in zip(cu_host, cu_host[1:]):
        starts.append(at)
        at += triton.cdiv(b - a, bt)
    return chunk_indices_host(cu_host, bt), starts


@triton.jit
def _gram_kernel(
    q,
    k,
    beta,
    g,
    gc,
    A,
    W,
    cu,
    chunks,
    H: tl.constexpr,
    HK: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    i_h, i_c = tl.program_id(0), tl.program_id(1)
    i_n, i_t = tl.load(chunks + 2 * i_c), tl.load(chunks + 2 * i_c + 1)
    bos = tl.load(cu + i_n).to(tl.int64)
    n = tl.load(cu + i_n + 1).to(tl.int64) - bos
    o_b = tl.arange(0, BT)
    o_t = i_t * BT + o_b
    m_t = o_t < n
    q += (bos * HK + i_h // (H // HK)) * K
    k += (bos * HK + i_h // (H // HK)) * K
    o_k = tl.arange(0, BK)
    b_kk = tl.zeros([BT, BT], dtype=tl.float32)
    b_qk = tl.zeros([BT, BT], dtype=tl.float32)
    parts = K // BK
    whole = tl.where((i_t + 1) * BT <= n, parts, 0)
    for i_k in range(whole):
        o_d = i_k * BK + o_k
        b_q = tl.load(q + o_t[:, None] * (HK * K) + o_d[None, :])
        b_kt = tl.load(k + o_d[:, None] + o_t[None, :] * (HK * K))
        b_kk += tl.dot(tl.trans(b_kt), b_kt)
        b_qk += tl.dot(b_q, b_kt)
    for i_k in range(whole, parts):
        o_d = i_k * BK + o_k
        b_q = tl.load(q + o_t[:, None] * (HK * K) + o_d[None, :], mask=m_t[:, None], other=0.0)
        b_kt = tl.load(k + o_d[:, None] + o_t[None, :] * (HK * K), mask=m_t[None, :], other=0.0)
        b_kk += tl.dot(tl.trans(b_kt), b_kt)
        b_qk += tl.dot(b_q, b_kt)
    p_g = (bos + o_t) * H + i_h
    b_beta = tl.load(beta + p_g, mask=m_t, other=0.0)
    b_g = tl.cumsum(tl.load(g + p_g, mask=m_t, other=0.0), 0)
    tl.store(gc + p_g, b_g, mask=m_t)
    r = tl.exp(b_g[:, None] - b_g[None, :])
    b_a = tl.where(o_b[:, None] > o_b[None, :], b_kk * r, 0.0)
    b_w = tl.where(o_b[:, None] >= o_b[None, :], b_qk * r, 0.0)
    p_a = (i_c * H + i_h) * (BT * BT) + o_b[:, None] * BT + o_b[None, :]
    tl.store(A + p_a, (b_a * b_beta[:, None]).to(tl.float16))
    tl.store(W + p_a, b_w.to(W.dtype.element_ty))


@triton.jit
def _gates_kernel(beta, g, gc, A, W, KK, QK, cu, chunks, H: tl.constexpr, BT: tl.constexpr):
    """``_gram_kernel``'s end over products already made, vector-only (Volta)."""
    i_h, i_c = tl.program_id(0), tl.program_id(1)
    i_n, i_t = tl.load(chunks + 2 * i_c), tl.load(chunks + 2 * i_c + 1)
    bos = tl.load(cu + i_n).to(tl.int64)
    n = tl.load(cu + i_n + 1).to(tl.int64) - bos
    o_b = tl.arange(0, BT)
    o_t = i_t * BT + o_b
    m_t = o_t < n
    p_g = (bos + o_t) * H + i_h
    b_beta = tl.load(beta + p_g, mask=m_t, other=0.0)
    b_g = tl.cumsum(tl.load(g + p_g, mask=m_t, other=0.0), 0)
    tl.store(gc + p_g, b_g, mask=m_t)
    r = tl.exp(b_g[:, None] - b_g[None, :])
    p_a = (i_c * H + i_h) * (BT * BT) + o_b[:, None] * BT + o_b[None, :]
    b_a = tl.where(o_b[:, None] > o_b[None, :], tl.load(KK + p_a) * r, 0.0)
    b_w = tl.where(o_b[:, None] >= o_b[None, :], tl.load(QK + p_a) * r, 0.0)
    tl.store(A + p_a, (b_a * b_beta[:, None]).to(tl.float16))
    tl.store(W + p_a, b_w.to(W.dtype.element_ty))


@triton.jit
def _sweep(p, o_i, BT: tl.constexpr):
    """``(I + a) ** -1 - I`` of the 16 x 16 diagonal block of strictly lower ``a`` at ``p``."""
    b_i = -tl.load(p + o_i[:, None] * BT + o_i[None, :]).to(tl.float32)
    for i in range(2, 16):
        b_a = -tl.load(p + i * BT + o_i).to(tl.float32)
        b_a = b_a + tl.sum(b_a[:, None] * b_i, 0)
        b_i = tl.where((o_i == i)[:, None], b_a, b_i)
    return b_i


@triton.jit
def _put(Ti, r, c, b, o_i, BT: tl.constexpr):
    tl.store(Ti + r * BT + c + o_i[:, None] * BT + o_i[None, :], b.to(Ti.dtype.element_ty))


@triton.jit
def _inverse_kernel(A, Ti, BT: tl.constexpr):
    i_o = tl.program_id(0)
    A += i_o * (BT * BT)
    Ti += i_o * (BT * BT)
    o_i = tl.arange(0, 16)
    eye = (o_i[:, None] == o_i[None, :]).to(tl.float32)
    i11 = (_sweep(A, o_i, BT) + eye).to(tl.float16)
    if BT == 16:
        _put(Ti, 0, 0, i11, o_i, BT)
        return
    i22 = (_sweep(A + 16 * (BT + 1), o_i, BT) + eye).to(tl.float16)
    if BT == 32:
        a21 = tl.load(A + 16 * BT + o_i[:, None] * BT + o_i[None, :])
        i21 = -tl.dot(tl.dot(i22, a21).to(tl.float16), i11)
        _put(Ti, 0, 0, i11, o_i, BT)
        _put(Ti, 16, 16, i22, o_i, BT)
        _put(Ti, 16, 0, i21, o_i, BT)
        _put(Ti, 0, 16, tl.zeros([16, 16], dtype=tl.float32), o_i, BT)
        return
    i33 = (_sweep(A + 32 * (BT + 1), o_i, BT) + eye).to(tl.float16)
    i44 = (_sweep(A + 48 * (BT + 1), o_i, BT) + eye).to(tl.float16)
    a21 = tl.load(A + 16 * BT + o_i[:, None] * BT + o_i[None, :])
    a31 = tl.load(A + 32 * BT + o_i[:, None] * BT + o_i[None, :])
    a32 = tl.load(A + 32 * BT + 16 + o_i[:, None] * BT + o_i[None, :])
    a41 = tl.load(A + 48 * BT + o_i[:, None] * BT + o_i[None, :])
    a42 = tl.load(A + 48 * BT + 16 + o_i[:, None] * BT + o_i[None, :])
    a43 = tl.load(A + 48 * BT + 32 + o_i[:, None] * BT + o_i[None, :])
    i21 = -tl.dot(tl.dot(i22, a21).to(tl.float16), i11)
    i32 = -tl.dot(tl.dot(i33, a32).to(tl.float16), i22)
    i43 = -tl.dot(tl.dot(i44, a43).to(tl.float16), i33)
    f21, f32 = i21.to(tl.float16), i32.to(tl.float16)
    i31 = -tl.dot(i33, (tl.dot(a31, i11) + tl.dot(a32, f21)).to(tl.float16))
    i42 = -tl.dot(i44, (tl.dot(a42, i22) + tl.dot(a43, f32)).to(tl.float16))
    f31 = i31.to(tl.float16)
    i41 = -tl.dot(i44, (tl.dot(a41, i11) + tl.dot(a42, f21) + tl.dot(a43, f31)).to(tl.float16))
    _put(Ti, 0, 0, i11, o_i, BT)
    _put(Ti, 16, 16, i22, o_i, BT)
    _put(Ti, 32, 32, i33, o_i, BT)
    _put(Ti, 48, 48, i44, o_i, BT)
    _put(Ti, 16, 0, i21, o_i, BT)
    _put(Ti, 32, 16, i32, o_i, BT)
    _put(Ti, 48, 32, i43, o_i, BT)
    _put(Ti, 32, 0, i31, o_i, BT)
    _put(Ti, 48, 16, i42, o_i, BT)
    _put(Ti, 48, 0, i41, o_i, BT)
    z = tl.zeros([16, 16], dtype=tl.float32)
    _put(Ti, 0, 16, z, o_i, BT)
    _put(Ti, 0, 32, z, o_i, BT)
    _put(Ti, 0, 48, z, o_i, BT)
    _put(Ti, 16, 32, z, o_i, BT)
    _put(Ti, 16, 48, z, o_i, BT)
    _put(Ti, 32, 48, z, o_i, BT)


@triton.jit
def _chunk(b_q, b_kt, b_v, b_t, b_w, b_beta, b_g, g_end, s, m_t, scale, MM: tl.constexpr):
    """One chunk: the output [BT, BV] and the next state [K, BV]."""
    c = tl.exp(b_g)
    sm = s.to(MM)
    sk = tl.dot(tl.trans(b_kt), sm)
    x = ((b_v.to(tl.float32) - sk * c[:, None]) * b_beta[:, None]).to(MM)
    u = tl.dot(b_t, x)
    b_o = (tl.dot(b_q, sm) * c[:, None] + tl.dot(b_w, u.to(MM))) * scale
    r = tl.where(m_t, tl.exp(g_end - b_g), 0.0)
    return b_o, s * tl.exp(g_end) + tl.dot(b_kt, (u * r[:, None]).to(MM))


@triton.jit
def _state_kernel(
    q,
    k,
    v,
    g,
    beta,
    Ti,
    W,
    o,
    h0,
    ht,
    cu,
    starts,
    scale,
    H: tl.constexpr,
    HK: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    MM: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu + i_n).to(tl.int64)
    n = tl.load(cu + i_n + 1).to(tl.int64) - bos
    start = tl.load(starts + i_n).to(tl.int64)
    o_b = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    h0 += i_nh.to(tl.int64) * (V * K)
    ht += i_nh.to(tl.int64) * (V * K)
    s = tl.load(h0 + o_k[:, None] + o_v[None, :] * K)
    ik = (bos * HK + i_h // (H // HK)) * K
    q += ik
    k += ik
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    g += bos * H + i_h
    beta += bos * H + i_h
    Ti += (start * H + i_h) * (BT * BT)
    W += (start * H + i_h) * (BT * BT)
    full = n // BT
    for i_t in range(full):
        o_t = i_t * BT + o_b
        b_q = tl.load(q + o_t[:, None] * (HK * K) + o_k[None, :])
        b_kt = tl.load(k + o_k[:, None] + o_t[None, :] * (HK * K))
        b_v = tl.load(v + o_t[:, None] * (H * V) + o_v[None, :])
        b_t = tl.load(Ti + i_t * (H * BT * BT) + o_b[:, None] * BT + o_b[None, :])
        b_w = tl.load(W + i_t * (H * BT * BT) + o_b[:, None] * BT + o_b[None, :])
        b_g = tl.load(g + o_t * H)
        b_beta = tl.load(beta + o_t * H)
        g_end = tl.load(g + (i_t + 1) * BT * H - H)
        b_o, s = _chunk(b_q, b_kt, b_v, b_t, b_w, b_beta, b_g, g_end, s, o_b < BT, scale, MM)
        tl.store(o + o_t[:, None] * (H * V) + o_v[None, :], b_o.to(o.dtype.element_ty))
    for i_t in range(full, tl.cdiv(n, BT)):
        o_t = i_t * BT + o_b
        m_t = o_t < n
        m_o = m_t[:, None]
        b_q = tl.load(q + o_t[:, None] * (HK * K) + o_k[None, :], mask=m_o, other=0.0)
        b_kt = tl.load(k + o_k[:, None] + o_t[None, :] * (HK * K), mask=m_t[None, :], other=0.0)
        b_v = tl.load(v + o_t[:, None] * (H * V) + o_v[None, :], mask=m_o, other=0.0)
        b_t = tl.load(Ti + i_t * (H * BT * BT) + o_b[:, None] * BT + o_b[None, :])
        b_w = tl.load(W + i_t * (H * BT * BT) + o_b[:, None] * BT + o_b[None, :])
        b_g = tl.load(g + o_t * H, mask=m_t, other=0.0)
        b_beta = tl.load(beta + o_t * H, mask=m_t, other=0.0)
        g_end = tl.load(g + (n - 1) * H)
        b_o, s = _chunk(b_q, b_kt, b_v, b_t, b_w, b_beta, b_g, g_end, s, m_t, scale, MM)
        tl.store(o + o_t[:, None] * (H * V) + o_v[None, :], b_o.to(o.dtype.element_ty), mask=m_o)
    tl.store(ht + o_k[:, None] + o_v[None, :] * K, s)


def gram(q, k, beta, g, cu, chunks, bt: int):
    """The chunks' gated products: (running log gates, ``A``, ``W``)."""
    hk, kd = q.shape[1:]
    hv, nc = beta.shape[1], chunks.shape[0]
    gc = torch.empty_like(g)
    A = torch.empty(nc * hv, bt, bt, dtype=torch.float16, device=q.device)
    W = torch.empty(nc * hv, bt, bt, dtype=q.dtype, device=q.device)
    if is_volta(q.device):
        kk = torch.empty(nc * hv, bt, bt, dtype=torch.float32, device=q.device)
        qk = torch.empty_like(kk)
        volta.gram(q, k, cu, chunks, kk, qk, min(BK, kd), hv, bt)
        _gates_kernel[(hv, nc)](
            beta, g, gc, A, W, kk, qk, cu, chunks, H=hv, BT=bt, num_warps=4, num_stages=2
        )
        return gc, A, W
    _gram_kernel[(hv, nc)](
        q,
        k,
        beta,
        g,
        gc,
        A,
        W,
        cu,
        chunks,
        H=hv,
        HK=hk,
        K=kd,
        BK=min(BK, kd),
        BT=bt,
        num_warps=4,
        num_stages=2,
    )
    return gc, A, W


def invert(A: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``(I + A) ** -1`` of every ``[BT, BT]`` block, strictly lower ``A``."""
    n, bt = A.shape[:2]
    Ti = torch.empty(n, bt, bt, dtype=dtype, device=A.device)
    _inverse_kernel[(n,)](A, Ti, BT=bt, num_warps=4, num_stages=2)
    return Ti


def advance(q, k, v, gc, beta, Ti, W, out, initial_state, cu, starts, scale, tile) -> torch.Tensor:
    """Every sequence's chunks in order: fills ``out``, returns the final state."""
    hk, kd = q.shape[1:]
    hv, vd = v.shape[1:]
    bv, warps, stages = tile
    if vd % bv or bv > vd:
        raise ValueError(f"GDN prefill: V {vd} must be a multiple of {bv} state rows")
    final = torch.empty_like(initial_state)
    if is_volta(q.device):
        volta.state(q, k, v, gc, beta, Ti, W, out, initial_state, final, cu, starts, scale, bv)
        return final
    _state_kernel[(vd // bv, (cu.shape[0] - 1) * hv)](
        q,
        k,
        v,
        gc,
        beta,
        Ti,
        W,
        out,
        initial_state,
        final,
        cu,
        starts,
        scale,
        H=hv,
        HK=hk,
        K=kd,
        V=vd,
        BV=bv,
        BT=W.shape[1],
        MM=tl_dtype(q.dtype),
        num_warps=warps,
        num_stages=stages,
    )
    return final


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    cu: torch.Tensor,
    chunks: torch.Tensor,
    starts: torch.Tensor,
    out: torch.Tensor | None = None,
    tile: tuple[int, int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``q``, ``k`` [T, Hk, K], ``v`` [T, Hv, V] in one matmul dtype, ``g`` (log gates), ``beta``
    [T, Hv] float32, ``initial_state`` [N, Hv, V, K] float32, ``cu`` [N + 1], ``chunks``
    ``[sequence, chunk]`` rows and ``starts`` [N] from ``table``; ``out`` [T, Hv, V] (default:
    ``v``'s dtype); ``tile`` ``(BT, BV, warps, stages)`` as ``config`` gives it, the table's
    ``BT``. Returns the output and the final state [N, Hv, V, K] float32."""
    bt, *rest = tile or config(q.device)
    if out is None:
        out = torch.empty_like(v)
    gc, A, W = gram(q, k, beta, g, cu, chunks, bt)
    Ti = invert(A, q.dtype)
    return out, advance(q, k, v, gc, beta, Ti, W, out, initial_state, cu, starts, scale, rest)
