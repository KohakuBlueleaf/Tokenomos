"""``skinny``'s narrow linear on Volta tensor cores (TileLang): a program holds 64 rows, the padded
``N``, float32 accumulation over 64-column chunks, one rounding to the output dtype."""

from functools import cache

import tilelang
import tilelang.language as T
import torch

from tokenomos.kernels.linear.int8 import arch

BM, BK, THREADS = 64, 64, 128
NAMES = {torch.float16: "float16", torch.float32: "float32"}
PASSES = {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}


@cache
def _kernel(n: int, k: int, bn: int, xt: str, bt: str, has_bias: bool, target: str):
    m, lx, lw = T.dynamic("m"), T.dynamic("lx"), T.dynamic("lw")

    @T.prim_func
    def main(
        X: T.Tensor((lx,), xt),
        W: T.Tensor((lw,), "float16"),
        B: T.Tensor((n,), bt),
        Y: T.Tensor((m, n), xt),
        stride_x: T.int32,
    ):
        with T.Kernel(T.ceildiv(m, BM), threads=THREADS) as by:
            a_s = T.alloc_shared((BM, BK), "float16")
            b_s = T.alloc_shared((bn, BK), "float16")
            acc = T.alloc_fragment((BM, bn), "float32")
            T.clear(acc)
            for kb in T.serial(T.ceildiv(k, BK)):
                for i, j in T.Parallel(BM, BK):
                    r, c = by * BM + i, kb * BK + j
                    a_s[i, j] = T.if_then_else(T.And(r < m, c < k), X[r * stride_x + c], 0)
                for i, j in T.Parallel(bn, BK):
                    c = kb * BK + j
                    b_s[i, j] = T.if_then_else(T.And(i < n, c < k), W[i * k + c], 0)
                T.gemm(a_s, b_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(BM, bn):
                r = by * BM + i
                if r < m and j < n:
                    if has_bias:
                        Y[r, j] = (acc[i, j] + B[j].astype("float32")).astype(xt)
                    else:
                        Y[r, j] = acc[i, j].astype(xt)

    return tilelang.compile(main, target={"kind": "cuda", "arch": target}, pass_configs=PASSES)


def linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``x`` [M, K] (row stride free) times fp16 ``w`` [N, K] contiguous -> [M, N] in ``x``'s
    dtype."""
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    if m:
        flat = x.as_strided(((m - 1) * x.stride(0) + k,), (1,))
        names = NAMES[x.dtype], NAMES[w.dtype if bias is None else bias.dtype]
        kernel = _kernel(n, k, -(-n // 16) * 16, *names, bias is not None, arch(x.device))
        kernel(flat, w.view(-1), w[0, :n] if bias is None else bias, out, x.stride(0))
    return out
