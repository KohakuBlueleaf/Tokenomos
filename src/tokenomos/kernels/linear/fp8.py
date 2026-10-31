"""FP8 (E4M3) linears.

Static: a per-tensor activation scale, the activation divided by it and rounded to E4M3
(saturating at 448), the product by cuBLAS's scaled GEMM, tensor-wise when the weight has one scale,
row-wise when it has one per output channel (fused parts with different scales).

Dynamic (compressed-tensors ``float-quantized``): weight scales per output row per ``GROUP``
columns, whatever the checkpoint's own granularity (``expand_scales``). ``dynamic_quant`` scales
each activation row per ``GROUP`` columns from its own amax, a finer grain than a token-wise scheme
asks for at no cost in the GEMM. ``dynamic_gemm`` (Triton): tensor-core FP8 dots over ``GROUP``
columns, each times both scales, float32 accumulation, one rounding to the output.
"""

import torch
import triton
import triton.language as tl

from tokenomos.core.floats import tl_dtype

E4M3_MAX = 448.0
GROUP = 128


def quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x`` [M, K] -> E4M3 with the 1-element float32 ``scale`` (``x ~ q * scale``)."""
    return (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)


def gemm(q: torch.Tensor, scale: torch.Tensor, w: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """``q`` [M, K] E4M3 with tensor scale ``scale``, weight ``w`` [N, K] E4M3 with ``ws`` (1 or
    ``N`` float32) -> bf16 [M, N]."""
    if ws.numel() == 1:
        return torch._scaled_mm(q, w.t(), scale, ws, out_dtype=torch.bfloat16)
    sa = scale.view(1, 1).expand(q.shape[0], 1).contiguous()
    return torch._scaled_mm(q, w.t(), sa, ws.view(1, -1), out_dtype=torch.bfloat16)


def expand_scales(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """A checkpoint's weight scale (one value, per channel, or per ``GROUP`` x ``GROUP`` block) ->
    float32 [N, ceil(K / GROUP)]."""
    kb = triton.cdiv(k, GROUP)
    s = scale.float()
    if s.numel() == 1 or s.numel() == n:
        return s.reshape(-1, 1).expand(n, kb).contiguous()
    if s.shape != (triton.cdiv(n, GROUP), kb):
        raise ValueError(f"FP8 weight scale {tuple(s.shape)} for a weight [{n}, {k}]")
    return s.repeat_interleave(GROUP, 0)[:n].contiguous()


@triton.jit
def _quant_kernel(x_ptr, q_ptr, s_ptr, K, KB, stride_x, GROUP: tl.constexpr):
    row = tl.program_id(0)
    for g in range(0, KB):
        cols = g * GROUP + tl.arange(0, GROUP)
        m = cols < K
        v = tl.load(x_ptr + row * stride_x + cols, mask=m, other=0.0).to(tl.float32)
        s = tl.maximum(tl.max(tl.abs(v)), 1e-30) / 448.0
        tl.store(q_ptr + row * K + cols, (v / s).to(tl.float8e4nv), mask=m)
        tl.store(s_ptr + row * KB + g, s)


def dynamic_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` [M, K] (row stride free) -> E4M3 [M, K] and scales [M, ceil(K / GROUP)] float32."""
    m, k = x.shape
    kb = triton.cdiv(k, GROUP)
    q = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(m, kb, dtype=torch.float32, device=x.device)
    if m:
        _quant_kernel[(m,)](x, q, s, k, kb, x.stride(0), GROUP=GROUP, num_warps=4)
    return q, s


@triton.jit
def _gemm_kernel(
    q_ptr,
    sq_ptr,
    w_ptr,
    sw_ptr,
    b_ptr,
    o_ptr,
    M,
    N,
    K,
    KB,
    stride_o,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GROUP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DT: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    rm, cm = rows < M, cols < N
    acc = tl.zeros((BM, BN), tl.float32)
    for g in range(0, KB):
        ks = g * GROUP + tl.arange(0, GROUP)
        km = ks < K
        ma, mb = rm[:, None] & km[None, :], km[:, None] & cm[None, :]
        a = tl.load(q_ptr + rows[:, None] * K + ks[None, :], mask=ma, other=0.0)
        b = tl.load(w_ptr + cols[None, :] * K + ks[:, None], mask=mb, other=0.0)
        sa = tl.load(sq_ptr + rows * KB + g, mask=rm, other=0.0)
        sb = tl.load(sw_ptr + cols * KB + g, mask=cm, other=0.0)
        acc += tl.dot(a, b) * sa[:, None] * sb[None, :]
    if HAS_BIAS:
        acc += tl.load(b_ptr + cols, mask=cm, other=0.0).to(tl.float32)[None, :]
    out = o_ptr + rows[:, None] * stride_o + cols[None, :]
    tl.store(out, acc.to(DT), mask=rm[:, None] & cm[None, :])


def dynamic_gemm(q, s, w, ws, bias=None, dtype=torch.bfloat16) -> torch.Tensor:
    """``dynamic_quant``'s output times ``w`` [N, K] E4M3 with ``expand_scales``' ``ws`` -> [M, N]
    in ``dtype``, plus ``bias``."""
    m, n, k = q.shape[0], w.shape[0], w.shape[1]
    out = torch.empty(m, n, dtype=dtype, device=q.device)
    bm, bn = (16 if m <= 64 else 64), 128
    _gemm_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        q,
        s,
        w,
        ws,
        bias if bias is not None else ws,
        out,
        m,
        n,
        k,
        s.shape[1],
        out.stride(0),
        BM=bm,
        BN=bn,
        GROUP=GROUP,
        HAS_BIAS=bias is not None,
        DT=tl_dtype(dtype),
        num_warps=4,
        num_stages=2,
    )
    return out
