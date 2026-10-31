"""A 16-bit float linear with a narrow output (``N <= 64``: per-head gates and the like), where
cuBLAS picks split-K kernels that cost far more than the weight read. ``BM`` rows per program,
the whole ``N`` at once, float32 accumulation over ``K`` chunks, one rounding to the output dtype
(as cuBLAS). An fp32 ``x`` takes the weight's dtype for the product and the output stays fp32.
"""

import torch
import triton
import triton.language as tl

MAX_N = 64


@triton.jit
def _skinny_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    M,
    N,
    K,
    stride_x,
    stride_o,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    rm, cm = rows < M, cols < N
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        km = ks < K
        x = tl.load(x_ptr + rows[:, None] * stride_x + ks[None, :], mask=rm[:, None] & km[None, :])
        w = tl.load(w_ptr + cols[:, None] * K + ks[None, :], mask=cm[:, None] & km[None, :])
        acc += tl.dot(x.to(w.dtype), tl.trans(w))
    if HAS_BIAS:
        acc += tl.load(b_ptr + cols, mask=cm, other=0.0).to(tl.float32)[None, :]
    out = o_ptr + rows[:, None] * stride_o + cols[None, :]
    tl.store(out, acc.to(o_ptr.dtype.element_ty), mask=rm[:, None] & cm[None, :])


def linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``x`` [M, K] (row stride free) times ``w`` [N, K] contiguous -> [M, N] in ``x``'s dtype."""
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    if m:
        bm = 16 if m <= 1024 or k > 512 else 64
        _skinny_kernel[(triton.cdiv(m, bm),)](
            x,
            w,
            bias if bias is not None else w,
            out,
            m,
            n,
            k,
            x.stride(0),
            out.stride(0),
            BM=bm,
            BN=max(16, triton.next_power_of_2(n)),
            BK=256,
            HAS_BIAS=bias is not None,
            num_warps=4,
        )
    return out
