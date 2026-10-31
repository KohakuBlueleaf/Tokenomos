"""Per-head QK RMSNorm and RoPE in one kernel, straight from the fused projection's views.

Each program takes one head of one token (query heads first, then key heads, then, with ``v``,
value heads): optional RMSNorm with its weight (HF rounding: normed value to the 16-bit dtype
``DT`` of ``q``, times the weight, to ``DT``; without ``round_normed`` the float32 product rounded
once), then the neox rotation of the first ``R`` dims (``R`` the width of the step's cos / sin
tables, ``[T, R]``, already rounded as the model rounds them; ``rotate_half`` within each
``PERIOD`` dims): ``(y * cos + rotate_half(y) * sin)`` in float32, rounded to ``DT``; dims past
``R`` pass through. A value head is RMS-normalized without a weight and not rotated (Gemma 4's
``v_norm``). Writes contiguous ``q`` [T, H, D], ``k`` [T, Hkv, D] (and ``v`` [T, Hkv, D]).
"""

import torch
import triton
import triton.language as tl

from tokenomos.core.floats import tl_dtype
from tokenomos.kernels.rounding import round_as


@triton.jit
def _normed(src, w_ptr, cols, cm, rstd, NORM: tl.constexpr, RN: tl.constexpr, DT: tl.constexpr):
    x = tl.load(src + cols, mask=cm, other=0.0).to(tl.float32)
    if NORM:
        w = tl.load(w_ptr + cols, mask=cm, other=0.0).to(tl.float32)
        if RN:  # noqa: SIM108 - a constexpr branch: Triton compiles only the taken side
            x = round_as(round_as(x * rstd, DT) * w, DT)
        else:
            x = round_as(x * rstd * w, DT)
    return x


@triton.jit
def _qk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qo_ptr,
    ko_ptr,
    vo_ptr,
    wq_ptr,
    wk_ptr,
    cos_ptr,
    sin_ptr,
    eps_q,
    eps_k,
    eps_v,
    stride_q,
    stride_k,
    stride_v,
    stride_c,
    H,
    HKV,
    D: tl.constexpr,
    DP: tl.constexpr,
    R: tl.constexpr,
    PERIOD: tl.constexpr,
    NORM: tl.constexpr,
    RN: tl.constexpr,
    DT: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    cols = tl.arange(0, DP)
    cm = cols < D
    if h >= H + HKV:
        src = v_ptr + t * stride_v + (h - H - HKV) * D
        x = tl.load(src + cols, mask=cm, other=0.0).to(tl.float32)
        y = round_as(x / tl.sqrt(tl.sum(x * x) / D + eps_v), DT)
        tl.store(vo_ptr + (t * HKV + h - H - HKV) * D + cols, y.to(DT), mask=cm)
    else:
        rot_part = cols < R
        within = cols % PERIOD
        rcols = tl.where(rot_part, cols - within + (within + PERIOD // 2) % PERIOD, cols)
        if h < H:
            src = q_ptr + t * stride_q + h * D
            dst = qo_ptr + (t * H + h) * D
            w_ptr = wq_ptr
            eps = eps_q
        else:
            src = k_ptr + t * stride_k + (h - H) * D
            dst = ko_ptr + (t * HKV + h - H) * D
            w_ptr = wk_ptr
            eps = eps_k
        rstd = 1.0
        if NORM:
            x = tl.load(src + cols, mask=cm, other=0.0).to(tl.float32)
            rstd = 1.0 / tl.sqrt(tl.sum(x * x) / D + eps)
        y = _normed(src, w_ptr, cols, cm, rstd, NORM, RN, DT)
        yr = _normed(src, w_ptr, rcols, cm, rstd, NORM, RN, DT)
        rot = tl.where(within < PERIOD // 2, -yr, yr)
        c = tl.load(cos_ptr + t * stride_c + cols, mask=rot_part, other=1.0)
        s = tl.load(sin_ptr + t * stride_c + cols, mask=rot_part, other=0.0)
        tl.store(dst + cols, (y * c + rot * s).to(DT), mask=cm)


def norm_rope(
    q,
    k,
    wq,
    wk,
    eps_q,
    eps_k,
    cos,
    sin,
    round_normed: bool = True,
    v=None,
    eps_v: float = 0.0,
    chunks: int = 1,
):
    """``q`` [T, H, D] / ``k`` [T, Hkv, D] (token stride free, heads contiguous); ``wq`` / ``wk``
    the norm weights (None: no norm); ``cos`` / ``sin`` float32 ``[T, R]``, ``R <= D``, with ``R``
    contiguous per token; ``v`` [T, Hkv, D]: also RMS-normalized without weight (returned
    third), else ``(q, k)``. ``chunks``: the rotation runs within each of that many equal parts
    of the ``R`` dims (Gemma 4's vision RoPE: 2, one axis each)."""
    t, h, d = q.shape
    hkv = k.shape[1]
    assert q.stride(1) == d and q.stride(2) == 1 and k.stride(1) == d and k.stride(2) == 1
    qo = torch.empty(t, h, d, dtype=q.dtype, device=q.device)
    ko = torch.empty(t, hkv, d, dtype=k.dtype, device=k.device)
    vo = torch.empty(t, hkv, d, dtype=k.dtype, device=k.device) if v is not None else None
    if v is not None:
        assert v.stride(1) == d and v.stride(2) == 1
    if t:
        norm = wq is not None
        _qk_kernel[(t, h + hkv * (2 if v is not None else 1))](
            q,
            k,
            v if v is not None else k,
            qo,
            ko,
            vo if v is not None else ko,
            wq if norm else q,
            wk if norm else k,
            cos,
            sin,
            eps_q,
            eps_k,
            eps_v,
            q.stride(0),
            k.stride(0),
            v.stride(0) if v is not None else 0,
            cos.stride(0),
            h,
            hkv,
            D=d,
            DP=triton.next_power_of_2(d),
            R=cos.shape[-1],
            PERIOD=cos.shape[-1] // chunks,
            NORM=norm,
            RN=round_normed,
            DT=tl_dtype(q.dtype),
            num_warps=1 if d <= 128 else 2,
        )
    return (qo, ko) if v is None else (qo, ko, vo)
