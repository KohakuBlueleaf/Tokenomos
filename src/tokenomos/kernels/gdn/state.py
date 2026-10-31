"""The GDN recurrent state pool in any of four formats, and its decode step.

Layout: per slot ``[HV, V, row]``: ``fp32`` / ``bf16`` rows of ``K`` values (``bf16``: the work
dtype); ``fp8`` / ``int8`` rows of ``K`` 8-bit values followed by that row's float32 scale
(``amax / 448`` for FP8, ``amax / 127`` for INT8), the pool in uint8 (``K + 4`` bytes a row; a
quarter of the fp32 bytes). Pool row 0 is the null row: a sequence with slot ``<= 0`` reads
nothing and gets a zero output.

``decode``: one Triton kernel per step reads a row tile, dequantizes, runs the gated delta rule
step in float32 (gates, q/k L2 norm and scale included) and writes the rows back (8-bit:
requantized with a fresh scale). A chunked prefill reads its sequences' rows as float32
(``load``) and writes its final states back (``store``). FP8 needs sm89 and up.
"""

import torch
import triton
import triton.language as tl

from tokenomos.core.floats import work

# format -> (pool dtype, None: the work dtype; 8-bit value dtype, 8-bit top value)
FORMATS = {
    "fp32": (torch.float32, None, None),
    "bf16": (None, None, None),
    "fp8": (torch.uint8, torch.float8_e4m3fn, 448.0),
    "int8": (torch.uint8, torch.int8, 127.0),
}
CODES = {"fp32": 0, "bf16": 1, "fp8": 2, "int8": 3}
SCALE_BYTES = 4
# The decode kernel's state rows and warps per program; 8x1 and the larger tiles are level on a
# 5090 (1.9x the bf16 pool)
TILE = {"rows": 8, "warps": 1}


def shape(heads_v: int, v: int, k: int, fmt: str) -> tuple[tuple[int, int, int], torch.dtype]:
    """One slot's pool shape and dtype."""
    dtype, q8, _ = FORMATS[fmt]
    return (heads_v, v, k + SCALE_BYTES if q8 is not None else k), dtype or work()


def split(pool: torch.Tensor, k: int, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """An 8-bit ``pool`` [..., V, K + 4] -> its values [..., V, K] and scales [..., V] float32
    (views)."""
    return pool[..., :k].view(FORMATS[fmt][1]), pool[..., k:].view(torch.float32)[..., 0]


def quantize(state: torch.Tensor, out: torch.Tensor, fmt: str) -> torch.Tensor:
    """``state`` [..., V, K] float -> ``out`` in ``fmt`` (8-bit: per row)."""
    _, q8, top = FORMATS[fmt]
    if q8 is None:
        return out.copy_(state)
    k = state.shape[-1]
    values, scales = split(out, k, fmt)
    s = state.float().abs().amax(-1).clamp(min=1e-30) / top
    scales.copy_(s)
    x = state.float() / s[..., None]
    values.copy_(x.round() if q8 == torch.int8 else x)
    return out


def dequantize(pool: torch.Tensor, k: int, fmt: str) -> torch.Tensor:
    """``pool`` rows in ``fmt`` -> float32 [..., V, K]."""
    if FORMATS[fmt][1] is None:
        return pool.float()
    values, scales = split(pool, k, fmt)
    return values.float() * scales[..., None]


def load(pool: torch.Tensor, slots: torch.Tensor, init: torch.Tensor, k: int, fmt: str):
    """The rows of ``slots`` as float32 [N, HV, V, K], zero where ``init`` is 0 (a fresh
    sequence on a reused slot)."""
    rows = dequantize(pool.index_select(0, slots.long()), k, fmt)
    return rows * (init != 0).float()[:, None, None, None]


def store(pool: torch.Tensor, slots: torch.Tensor, state: torch.Tensor, fmt: str) -> None:
    rows = torch.empty((state.shape[0], *pool.shape[1:]), dtype=pool.dtype, device=pool.device)
    pool.index_copy_(0, slots.long(), quantize(state, rows, fmt))


@triton.jit
def decode_kernel(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    o,
    pool,
    slots,
    scale,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    stride_o,
    stride_slot,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    L2NORM: tl.constexpr,
    FMT: tl.constexpr,
):
    """One token of one sequence, one value-head, ``BV`` state rows. ``FMT``: 0 fp32, 1 the work
    dtype, 2 fp8, 3 int8."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    p_o = o + i_n * stride_o + i_hv * V + o_v

    slot = tl.load(slots + i_n).to(tl.int64)
    if slot <= 0:
        tl.store(p_o, tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty))
        return

    ROW: tl.constexpr = K + 4 if FMT >= 2 else K
    p_row = pool + slot * stride_slot + (i_hv * V + o_v) * ROW
    raw = tl.load(p_row[:, None] + o_k[None, :])
    if FMT == 3:
        h = raw.to(tl.int8, bitcast=True).to(tl.float32)
    elif FMT == 2:
        h = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        h = raw.to(tl.float32)
    if FMT >= 2:
        p_scale = (p_row + K).to(tl.pointer_type(tl.float32))
        h *= tl.load(p_scale)[:, None]

    b_q = tl.load(q + i_n * stride_q + i_h * K + o_k).to(tl.float32)
    b_k = tl.load(k + i_n * stride_k + i_h * K + o_k).to(tl.float32)
    b_v = tl.load(v + i_n * stride_v + i_hv * V + o_v).to(tl.float32)
    if L2NORM:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    x = tl.load(a + i_n * stride_a + i_hv).to(tl.float32) + tl.load(dt_bias + i_hv).to(tl.float32)
    softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    g = -tl.exp(tl.load(A_log + i_hv).to(tl.float32)) * softplus
    beta = tl.sigmoid(tl.load(b + i_n * stride_b + i_hv).to(tl.float32))

    h *= tl.exp(g)
    b_v -= tl.sum(h * b_k[None, :], 1)
    b_v *= beta
    h += b_v[:, None] * b_k[None, :]
    tl.store(p_o, tl.sum(h * b_q[None, :], 1).to(p_o.dtype.element_ty))

    if FMT >= 2:
        amax = tl.maximum(tl.max(tl.abs(h), 1), 1e-30)
        if FMT == 3:
            s = amax / 127.0
            x8 = h / s[:, None]
            q8 = tl.where(x8 >= 0, x8 + 0.5, x8 - 0.5).to(tl.int8)
        else:
            s = amax / 448.0
            q8 = (h / s[:, None]).to(tl.float8e4nv)
        tl.store(p_row[:, None] + o_k[None, :], q8.to(tl.uint8, bitcast=True))
        tl.store(p_scale, s)
    else:
        tl.store(p_row[:, None] + o_k[None, :], h.to(pool.dtype.element_ty))


def decode(
    q, k, v, a, b, A_log, dt_bias, pool, slots, out, scale: float, l2norm: bool, fmt: str, tile=None
):
    """One decode step of ``N`` sequences on the pool. ``q``, ``k`` [N, H * K], ``v``
    [N, HV * V] (row strides free), ``a``, ``b`` [N, HV], ``slots`` [N], ``out`` [N, HV * V];
    ``tile``: ``{"rows", "warps"}``, ``TILE`` by default."""
    tile = tile or TILE
    n = q.shape[0]
    HV, V, row = pool.shape[-3:]
    K = row - SCALE_BYTES if FORMATS[fmt][1] is not None else row
    H = q.shape[1] // K
    BV = min(triton.next_power_of_2(V), tile["rows"])
    if triton.next_power_of_2(K) != K or V % BV:
        raise ValueError(f"GDN state: K {K} must be a power of two and V {V} a multiple of {BV}")
    grid = (triton.cdiv(V, BV), n * HV)
    decode_kernel[grid](
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        out,
        pool,
        slots,
        scale,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        a.stride(0),
        b.stride(0),
        out.stride(0),
        pool.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BV=BV,
        L2NORM=l2norm,
        FMT=CODES[fmt],
        num_warps=tile["warps"],
    )
