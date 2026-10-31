"""Gated depthwise 2D conv over image patch grids kept in merge-block order (LFM3-VL's vision
conv layers), straight on the packed token layout: no permute to raster order and back.

Token ``i`` of an image with a ``gh x gw`` patch grid sits in merge block ``i // m^2`` (blocks in
raster order, ``gw / m`` per row) at ``(i % m^2) // m, i % m`` inside it. Its output is
``c[i] * conv(b * x)[i]``: the 3x3 (``K x K``) neighbourhood of ``b * x`` (rounded to ``x``'s
16-bit dtype ``DT``, as the reference multiplies in it) with zero padding, the conv rounded to
``DT``, times ``c``. ``b``, ``c``, ``x`` may be column slices of one projection (any token
stride).
"""

import torch
import triton
import triton.language as tl

from tokenomos.core.floats import tl_dtype
from tokenomos.kernels.rounding import round_as


@triton.jit
def _grid_conv_kernel(
    b_ptr,
    c_ptr,
    x_ptr,
    w_ptr,
    out_ptr,
    tok_img,
    cu,
    gh_ptr,
    gw_ptr,
    T,
    DIM,
    stride_t,
    stride_o,
    M: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    DT: tl.constexpr,
):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    tm = t < T
    dm = d < DIM
    img = tl.load(tok_img + t, mask=tm, other=0)
    a = tl.load(cu + img, mask=tm, other=0)
    gh = tl.load(gh_ptr + img, mask=tm, other=1)
    gw = tl.load(gw_ptr + img, mask=tm, other=1)
    i = t - a
    blk = i // (M * M)
    inner = i % (M * M)
    bw = gw // M
    r = (blk // bw) * M + inner // M
    c = (blk % bw) * M + inner % M
    acc = tl.zeros((BT, BD), tl.float32)
    for tap in tl.static_range(K * K):
        r2 = r + (tap // K - K // 2)
        c2 = c + (tap % K - K // 2)
        ok = tm & (r2 >= 0) & (r2 < gh) & (c2 >= 0) & (c2 < gw)
        j = a + ((r2 // M) * bw + c2 // M) * (M * M) + (r2 % M) * M + c2 % M
        j = tl.where(ok, j, 0)
        m2 = ok[:, None] & dm[None, :]
        off = j[:, None] * stride_t + d[None, :]
        bv = tl.load(b_ptr + off, mask=m2, other=0.0).to(tl.float32)
        xv = tl.load(x_ptr + off, mask=m2, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + d * (K * K) + tap, mask=dm, other=0.0).to(tl.float32)
        acc += round_as(bv * xv, DT) * wv[None, :]
    cv = tl.load(c_ptr + t[:, None] * stride_t + d[None, :], mask=tm[:, None] & dm[None, :])
    y = round_as(acc, DT) * cv.to(tl.float32)
    tl.store(
        out_ptr + t[:, None] * stride_o + d[None, :],
        y.to(DT),
        mask=tm[:, None] & dm[None, :],
    )


def grid_conv(b, c, x, weight, tok_img, cu, gh, gw, merge: int) -> torch.Tensor:
    """``b`` / ``c`` / ``x`` [T, dim] (shared token stride), ``weight`` [dim, K, K] (or
    [dim, 1, K, K]); ``tok_img`` [T] each token's image, ``cu`` [I + 1] image starts, ``gh`` /
    ``gw`` [I] patch grid sizes (multiples of ``merge``) -> ``c * conv(b * x)`` [T, dim] in
    ``x``'s 16-bit dtype."""
    t, dim = x.shape
    assert b.stride(0) == c.stride(0) == x.stride(0)
    k = weight.shape[-1]
    w = weight.reshape(dim, k * k).contiguous()
    out = torch.empty(t, dim, dtype=x.dtype, device=x.device)
    if t:
        bt, bd = 4, 256
        _grid_conv_kernel[(triton.cdiv(t, bt), triton.cdiv(dim, bd))](
            b,
            c,
            x,
            w,
            out,
            tok_img,
            cu,
            gh,
            gw,
            t,
            dim,
            x.stride(0),
            out.stride(0),
            M=merge,
            K=k,
            BT=bt,
            BD=bd,
            DT=tl_dtype(x.dtype),
            num_warps=4,
        )
    return out
