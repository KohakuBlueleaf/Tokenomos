"""Triton MXFP8 producers for sm120: a bf16 activation (or one these kernels make) to e4m3 values
plus ue8m0 block scales, 32 along K, the scales written straight into the 128x4 swizzled layout
cuBLAS (``SWIZZLE_32_4_4``) and FlashInfer's MXFP8 GEMMs read: row ``r`` of a 128-row tile at
``(r % 32) * 16 + (r // 32 % 4) * 4`` inside the tile's 512 bytes, tiles ordered row tile major.

Per block of 32: ``e = max(ceil(log2(amax / 448)), -127)``, ``q = e4m3(v / 2^e)``: the exponent
rounds up, so no value clips (the rule our MXFP8 checkpoints are quantized with). One program
per row of the padded row count (whole 128-row tiles); rows past ``M`` write zero scales, so the
scale buffer needs no zeroing.

``MODE`` selects the input: 0 ``a``; 1 ``a * b``; 2 ``silu(a) * b`` (``a`` / ``b`` the two halves
of one row); 3 ``a * sigmoid(b[col // GROUP])`` (a per-head gate); 4 ``gelu_tanh(a) * b``
(halves, as 2).
"""

import triton
import triton.language as tl

from tokenomos.kernels.activations import gelu_tanh

E4M3_MAX = tl.constexpr(448.0)
MX = tl.constexpr(32)


@triton.jit
def _quant(y, BLOCK: tl.constexpr):
    """fp32 ``[BLOCK]`` -> (e4m3 ``[BLOCK]``, biased ue8m0 exponents ``[BLOCK // 32]``)."""
    g = tl.reshape(y, (BLOCK // MX, MX))
    amax = tl.max(tl.abs(g), axis=1)
    e = tl.maximum(tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / E4M3_MAX)), -127.0)
    q = (g / tl.exp2(e)[:, None]).to(tl.float8e4nv)
    return tl.reshape(q, (BLOCK,)), (e + 127.0).to(tl.uint8)


@triton.jit
def _store_scales(s_ptr, row, k0, s, col_tiles, live, BLOCK: tl.constexpr, K):
    """Row ``row``'s scales for columns ``k0 .. min(k0 + BLOCK, K)`` into the swizzled layout."""
    groups: tl.constexpr = BLOCK // MX
    g = k0 // MX + tl.arange(0, groups)
    base = (row // 128) * col_tiles * 512 + (row % 32) * 16 + (row // 32 % 4) * 4
    tl.store(s_ptr + base + (g // 4) * 512 + g % 4, tl.where(live, s, 0), mask=g < K // MX)


@triton.jit
def _act_mx_kernel(
    a_ptr,
    b_ptr,
    q_ptr,
    s_ptr,
    M,
    K,
    col_tiles,
    stride_a,
    stride_b,
    stride_q,
    MODE: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    live = row < M
    for k0 in range(0, K, BLOCK):
        cols = k0 + tl.arange(0, BLOCK)
        a = tl.load(a_ptr + row * stride_a + cols, mask=live, other=0.0).to(tl.float32)
        if MODE == 1:
            b = tl.load(b_ptr + row * stride_b + cols, mask=live, other=0.0).to(tl.float32)
            y = a * b
        elif MODE == 2:
            b = tl.load(a_ptr + row * stride_a + K + cols, mask=live, other=0.0).to(tl.float32)
            y = a * tl.sigmoid(a) * b
        elif MODE == 3:
            b = tl.load(b_ptr + row * stride_b + cols // GROUP, mask=live, other=0.0)
            y = a * tl.sigmoid(b.to(tl.float32))
        elif MODE == 4:
            b = tl.load(a_ptr + row * stride_a + K + cols, mask=live, other=0.0).to(tl.float32)
            y = gelu_tanh(a) * b
        else:
            y = a
        q, s = _quant(y, BLOCK)
        tl.store(q_ptr + row * stride_q + cols, q, mask=live)
        _store_scales(s_ptr, row, k0, s, col_tiles, live, BLOCK, K)
