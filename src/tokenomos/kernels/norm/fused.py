"""One kernel for every residual-stream norm: optional post-norm of the incoming block output,
residual add, RMSNorm, optional MXFP8 quantization of the result. One row in registers, any
width (padded to a power of two in registers).

``stream = (residual + pre_norm(x)) * scale`` (``pre_norm`` only with ``pre``, the residual only
when given, ``scale`` a one-element tensor, Gemma 4's per-layer scalar), ``y = rmsnorm(stream) *
w`` (none without a weight: the stream alone). Rounds as the Hugging Face models do, to ``x``'s
16-bit dtype ``DT``: each norm's normed value before its weight (``round_normed``; off: the
float32 product, Qwen3.5's ``(x * w).to(DT)``), its product, the sum before the statistics, the
scaled stream.
"""

import torch
import triton
import triton.language as tl

from tokenomos.core.floats import tl_dtype
from tokenomos.kernels.mxfp8.producers import _quant, _store_scales
from tokenomos.kernels.rounding import round_as


@triton.jit
def _rms(x, w_ptr, cols, cm, eps, K: tl.constexpr, RN: tl.constexpr, DT: tl.constexpr):
    rstd = 1.0 / tl.sqrt(tl.sum(x * x) / K + eps)
    w = tl.load(w_ptr + cols, mask=cm, other=0.0).to(tl.float32)
    if RN:
        return round_as(round_as(x * rstd, DT) * w, DT)
    return round_as(x * rstd * w, DT)


@triton.jit
def _add_norm_kernel(
    x_ptr,
    r_ptr,
    wp_ptr,
    w_ptr,
    out_ptr,
    res_ptr,
    q_ptr,
    s_ptr,
    scale_ptr,
    M,
    eps_pre,
    eps,
    col_tiles,
    stride_x,
    stride_r,
    stride_out,
    stride_res,
    stride_q,
    K: tl.constexpr,
    KP: tl.constexpr,
    PRE: tl.constexpr,
    HAS_RES: tl.constexpr,
    WRITE_RES: tl.constexpr,
    WRITE_OUT: tl.constexpr,
    QUANT: tl.constexpr,
    RN_PRE: tl.constexpr,
    RN: tl.constexpr,
    SCALE: tl.constexpr,
    NORM: tl.constexpr,
    DT: tl.constexpr,
):
    row = tl.program_id(0)
    live = row < M
    cols = tl.arange(0, KP)
    cm = cols < K
    m = live & cm
    x = tl.load(x_ptr + row * stride_x + cols, mask=m, other=0.0).to(tl.float32)
    if PRE:
        x = _rms(x, wp_ptr, cols, cm, eps_pre, K, RN_PRE, DT)
    if HAS_RES:
        x += tl.load(r_ptr + row * stride_r + cols, mask=m, other=0.0).to(tl.float32)
        x = round_as(x, DT)
    if SCALE:
        x = round_as(x * tl.load(scale_ptr).to(tl.float32), DT)
    if WRITE_RES:
        tl.store(res_ptr + row * stride_res + cols, x.to(DT), mask=m)
    if NORM:
        y = _rms(x, w_ptr, cols, cm, eps, K, RN, DT)
        if WRITE_OUT:
            tl.store(out_ptr + row * stride_out + cols, y.to(DT), mask=m)
        if QUANT:
            q, s = _quant(y, KP)
            tl.store(q_ptr + row * stride_q + cols, q, mask=m)
            _store_scales(s_ptr, row, 0, s, col_tiles, live, KP, K)


def add_norm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor | None,
    eps: float,
    pre: tuple | None = None,
    write: bool = True,
    quant: bool = False,
    round_normed: bool = True,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, tuple | None, torch.Tensor]:
    """``x`` [M, K] in a 16-bit dtype; ``pre`` (weight, eps[, round_normed]) of the post-norm on
    ``x``; ``weight`` None: no norm (the stream only). Returns (normed or None unless ``write``,
    (e4m3 values, swizzled scales) when ``quant``, the stream)."""
    m, k = x.shape
    if quant and k % 128:
        raise ValueError(f"MXFP8 norm: width {k} is not a multiple of 128")
    norm = weight is not None
    write, quant = write and norm, quant and norm
    dev = x.device
    write_res = residual is not None or pre is not None or scale is not None
    out = torch.empty_like(x) if write else x.new_empty(0)
    res = torch.empty_like(x) if write_res else x.new_empty(0)
    q = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=dev) if quant else x.new_empty(0)
    s = torch.empty(-(-m // 128) * 128 * (k // 32), dtype=torch.uint8, device=dev)
    if not quant:
        s = s[:0]
    if m:
        kp = triton.next_power_of_2(k)
        _add_norm_kernel[((-(-m // 128) * 128) if quant else m,)](
            x,
            residual if residual is not None else x,
            pre[0] if pre is not None else (weight if norm else x),
            weight if norm else x,
            out if write else x,
            res if write_res else x,
            q if quant else x,
            s if quant else x,
            scale if scale is not None else x,
            m,
            pre[1] if pre is not None else 0.0,
            eps,
            k // 128,
            x.stride(0),
            residual.stride(0) if residual is not None else 0,
            out.stride(0) if write else 0,
            res.stride(0) if write_res else 0,
            q.stride(0) if quant else 0,
            K=k,
            KP=kp,
            PRE=pre is not None,
            HAS_RES=residual is not None,
            WRITE_RES=write_res,
            WRITE_OUT=write,
            QUANT=quant,
            RN_PRE=pre is None or len(pre) < 3 or pre[2],
            RN=round_normed,
            SCALE=scale is not None,
            NORM=norm,
            DT=tl_dtype(x.dtype),
            num_warps=8 if kp >= 4096 else (4 if kp >= 512 else 1),
        )
    return (out if write else None), ((q, s) if quant else None), (res if write_res else x)
