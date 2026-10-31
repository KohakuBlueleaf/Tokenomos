"""FP16 linears accumulating in float32, the matmul of the fp32 work dtype.

``pack`` (Triton): an fp32 activation [M, K] to fp16 in one pass, padded with zeros to whole
tiles. ``gemm`` (TileLang): the float32 product [rows(M), pad_n(N)] of ``q`` and fp16 ``w``
[N, K] on tensor cores; one kernel per (N, K, tile), its row count read from ``q``. A tile is
``(BM, BN, BK, threads)``; a row costs its whole tile.
"""

from functools import cache

import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl

from tokenomos.kernels.linear.int8 import arch

# compute capability major -> {tile rows BM: (BN, BK, threads)}
TILES = {
    7: {32: (64, 64, 128), 64: (64, 64, 128), 128: (64, 64, 256)},
    8: {32: (64, 64, 128), 64: (64, 64, 128), 128: (64, 64, 256)},
}


def tiles(device) -> dict[int, tuple[int, int, int]]:
    return TILES[min(torch.cuda.get_device_capability(device)[0], 8)]


def block_m(m: int, device) -> int:
    """The tile rows for ``m`` rows: the shortest tile holding them, else the tallest."""
    heights = sorted(tiles(device))
    return next((b for b in heights if m <= b), heights[-1])


def rows(m: int, bm: int) -> int:
    return -(-m // bm) * bm


def pad_size(device) -> tuple[int, int]:
    """(N, K) multiples every tile of ``device`` fits."""
    table = tiles(device).values()
    return max(t[0] for t in table), max(t[1] for t in table)


@triton.jit
def _pack_kernel(
    x_ptr,
    q_ptr,
    M,
    stride_x,
    stride_q,
    K: tl.constexpr,
    KP: tl.constexpr,
    PADK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, KP)
    x = tl.load(x_ptr + row * stride_x + cols, mask=(row < M) & (cols < K), other=0.0)
    tl.store(q_ptr + row * stride_q + cols, x.to(tl.float16), mask=cols < PADK)


def pack(x: torch.Tensor, r: int, padk: int) -> torch.Tensor:
    """``x`` [M, K] fp32 (row stride free) -> fp16 [r, padk], zeros past ``x``."""
    m, k = x.shape
    q = torch.empty(r, padk, dtype=torch.float16, device=x.device)
    kp = triton.next_power_of_2(padk)
    _pack_kernel[(r,)](
        x,
        q,
        m,
        x.stride(0),
        q.stride(0),
        K=k,
        KP=kp,
        PADK=padk,
        num_warps=8 if kp >= 4096 else (4 if kp >= 512 else 1),
    )
    return q


@cache
def _gemm_kernel(n: int, k: int, bm: int, bn: int, bk: int, threads: int, target: str):
    m = T.dynamic("m")

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((n, k), "float16"),
        C: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
            a_s = T.alloc_shared((bm, bk), "float16")
            b_s = T.alloc_shared((bn, bk), "float16")
            acc = T.alloc_fragment((bm, bn), "float32")
            T.clear(acc)
            for kb in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.copy(A[by * bm : (by + 1) * bm, kb * bk : (kb + 1) * bk], a_s)
                T.copy(B[bx * bn : (bx + 1) * bn, kb * bk : (kb + 1) * bk], b_s)
                T.gemm(a_s, b_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(acc, C[by * bm : (by + 1) * bm, bx * bn : (bx + 1) * bn])

    return tilelang.compile(
        main,
        target={"kind": "cuda", "arch": target},
        pass_configs={"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True},
    )


def gemm(q: torch.Tensor, w: torch.Tensor, tile: tuple[int, int, int, int]) -> torch.Tensor:
    """``q`` [rows, K] and ``w`` [N, K], K and N whole tiles of ``tile`` -> [rows, N] float32."""
    bm, bn, bk, threads = tile
    out = torch.empty(q.shape[0], w.shape[0], dtype=torch.float32, device=q.device)
    _gemm_kernel(w.shape[0], w.shape[1], bm, bn, bk, threads, arch(q.device))(q, w, out)
    return out


def linear(x: torch.Tensor, w: torch.Tensor, n: int, bias: torch.Tensor | None) -> torch.Tensor:
    """``x`` [M, K] fp32 times ``w`` [>= N, >= K] fp16 padded to ``pad_size`` -> [M, N] fp32 (row
    stride free)."""
    m = x.shape[0]
    bm = block_m(m, x.device)
    bn, bk, threads = tiles(x.device)[bm]
    out = gemm(pack(x, rows(m, bm), w.shape[1]), w, (bm, bn, bk, threads))[:m, :n]
    return out if bias is None else out.add_(bias)
