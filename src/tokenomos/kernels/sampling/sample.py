"""Token sampling, one program per row, parameters read per slot from device arrays.

``temperature`` 0 is greedy (first max). Otherwise ``x = logit / T`` restricted to the allowed
set, then Gumbel-max: ``argmax(x + g)``, ``g = -log(-log(u))``, ``u`` from Philox keyed by
(seed, step, vocab index) only, so a request samples the same tokens alone or in any batch.

Allowed set: ``min_p`` keeps ``x >= max + log(min_p)``; ``top_k`` keeps the ``k`` largest (the
threshold found by bisection on ``x``); ``top_p`` keeps the smallest top set whose probability
mass reaches ``p`` of what the other filters left (bisection too). Each bisection costs ``ITERS``
passes over the row and runs only for rows that ask for it.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mass(logits, row_off, V, inv_t, top, thr, BLOCK: tl.constexpr):
    """(sum of exp(x - top) over x >= thr, count of x >= thr), ``x = logit * inv_t``."""
    acc = tl.zeros((BLOCK,), tl.float32)
    cnt = tl.zeros((BLOCK,), tl.int32)
    for off in range(0, V, BLOCK):
        i = off + tl.arange(0, BLOCK)
        x = tl.load(logits + row_off + i, mask=i < V, other=float("-inf")).to(tl.float32) * inv_t
        keep = x >= thr
        acc += tl.where(keep, tl.exp(x - top), 0.0)
        cnt += keep.to(tl.int32)
    return tl.sum(acc), tl.sum(cnt)


@triton.jit
def _sample_kernel(
    logits,
    out,
    slot_ptr,
    temp,
    top_p,
    top_k,
    min_p,
    seed,
    step,
    V,
    stride_l,
    BLOCK: tl.constexpr,
    ITERS: tl.constexpr,
):
    r = tl.program_id(0)
    s = tl.load(slot_ptr + r)
    row_off = r.to(tl.int64) * stride_l
    best = tl.full((BLOCK,), float("-inf"), tl.float32)
    best_i = tl.zeros((BLOCK,), tl.int32)
    for off in range(0, V, BLOCK):
        i = off + tl.arange(0, BLOCK)
        x = tl.load(logits + row_off + i, mask=i < V, other=float("-inf")).to(tl.float32)
        up = x > best
        best = tl.where(up, x, best)
        best_i = tl.where(up, i, best_i)
    mx = tl.max(best)
    greedy = tl.min(tl.where(best == mx, best_i, V))
    t = tl.load(temp + s)
    if t <= 0.0:
        tl.store(out + r, greedy)
    else:
        inv_t = 1.0 / t
        top = mx * inv_t
        thr = float("-inf")
        mp = tl.load(min_p + s)
        if mp > 0.0:
            thr = top + tl.log(mp)
        k = tl.load(top_k + s)
        if (k > 0) & (k < V):
            lo = thr
            if lo == float("-inf"):
                lo = top - 1e4
            hi = top
            for _ in range(ITERS):
                mid = 0.5 * (lo + hi)
                _, c = _mass(logits, row_off, V, inv_t, top, mid, BLOCK)
                if c >= k:
                    lo = mid
                else:
                    hi = mid
            thr = lo
        pp = tl.load(top_p + s)
        if pp < 1.0:
            z, _ = _mass(logits, row_off, V, inv_t, top, thr, BLOCK)
            lo = thr
            if lo == float("-inf"):
                lo = top - 1e4
            hi = top
            for _ in range(ITERS):
                mid = 0.5 * (lo + hi)
                m, _ = _mass(logits, row_off, V, inv_t, top, mid, BLOCK)
                if m >= pp * z:
                    lo = mid
                else:
                    hi = mid
            thr = lo
        key = (tl.load(seed + s) * 1000003 + tl.load(step + s) * 7919) & 0x7FFFFFFF
        best = tl.full((BLOCK,), float("-inf"), tl.float32)
        best_i = tl.zeros((BLOCK,), tl.int32)
        for off in range(0, V, BLOCK):
            i = off + tl.arange(0, BLOCK)
            x = tl.load(logits + row_off + i, mask=i < V, other=float("-inf")).to(tl.float32)
            x = x * inv_t
            u = tl.rand(key, i)
            g = -tl.log(-tl.log(tl.maximum(u, 1e-20)))
            y = tl.where(x >= thr, x + g, float("-inf"))
            up = y > best
            best = tl.where(up, y, best)
            best_i = tl.where(up, i, best_i)
        m2 = tl.max(best)
        tl.store(out + r, tl.min(tl.where(best == m2, best_i, V)))


def sample(
    logits: torch.Tensor,
    slot: torch.Tensor,
    temp: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    step: torch.Tensor,
    out: torch.Tensor | None = None,
    iters: int = 24,
) -> torch.Tensor:
    """``logits`` [N, V] (any float dtype) -> token ids [N] int32; the parameters are per-slot
    arrays indexed by ``slot`` [N]."""
    n, v = logits.shape
    out = torch.empty(n, dtype=torch.int32, device=logits.device) if out is None else out
    _sample_kernel[(n,)](
        logits,
        out,
        slot,
        temp,
        top_p,
        top_k,
        min_p,
        seed,
        step,
        v,
        logits.stride(0),
        BLOCK=2048,
        ITERS=iters,
        num_warps=8,
    )
    return out
