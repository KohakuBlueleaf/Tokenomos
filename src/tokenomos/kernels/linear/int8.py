"""W8A8 linears on Ampere and Turing tensor cores.

``quant`` (Triton): one pass over an fp activation [M, K]: the regular Hadamard rotation in
registers, a scale per row from its amax, int8 values padded with zeros to whole tiles.
``gemm`` (TileLang): the int32 product [rows(M), N] of ``q`` and int8 ``w`` [N, K], exact; one
kernel per (N, K, tile), its row count read from ``q``, compiled once by nvcc and a C++ host
compiler. ``scale`` (Triton): the product to fp through both scales, plus the bias, in one pass.
"""

from functools import cache

import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl

BLOCK_N, BLOCK_K = 64, 64
SCALE_CHUNK = 1024


def block_m(m: int) -> int:
    """The tile rows for ``m`` rows: a tile costs its whole height."""
    return 32 if m <= 32 else 64 if m <= 64 else 128


def rows(m: int) -> int:
    bm = block_m(m)
    return -(-m // bm) * bm


def pad_k(k: int) -> int:
    return -(-k // BLOCK_K) * BLOCK_K


def pad_n(n: int) -> int:
    return -(-n // BLOCK_N) * BLOCK_N


@triton.jit
def _h4(x, cols, S: tl.constexpr):
    """Regular H4 over the four columns differing in bits ``S`` and ``S + 1`` of ``cols``: each
    entry is the group's sum less twice its mirror, over 2."""
    low = cols & ~(3 << S)
    a = tl.gather(x, low, axis=0)
    b = tl.gather(x, low + (1 << S), axis=0)
    c = tl.gather(x, low + (2 << S), axis=0)
    d = tl.gather(x, low + (3 << S), axis=0)
    mirror = tl.gather(x, cols ^ (3 << S), axis=0)
    return (a + b + c + d) * 0.5 - mirror


@triton.jit
def _quant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    M,
    stride_x,
    stride_q,
    K: tl.constexpr,
    KP: tl.constexpr,
    PADK: tl.constexpr,
    STAGES: tl.constexpr,
    GROUPED: tl.constexpr,
):
    row = tl.program_id(0)
    live = row < M
    cols = tl.arange(0, KP)
    m = live & (cols < K)
    x = tl.load(x_ptr + row * stride_x + cols, mask=m, other=0.0).to(tl.float32)
    if STAGES:
        y = x
        for stage in tl.static_range(STAGES):
            y = _h4(y, cols, 2 * stage)
        x = tl.where(cols < GROUPED, y, x)
    amax = tl.max(tl.abs(x))
    amax = tl.where(amax > 1e-12, amax, 1e-12) / 127.0
    q = tl.where(x >= 0, x / amax + 0.5, x / amax - 0.5).to(tl.int32)
    q = tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)
    tl.store(q_ptr + row * stride_q + cols, q, mask=cols < PADK)
    tl.store(s_ptr + row, amax, mask=live)


def quant(x: torch.Tensor, rotate: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` [M, K]; returns int8 [rows(M), pad_k(K)] and the scales [rows(M)] fp32."""
    m, k = x.shape
    r, padk = rows(m), pad_k(k)
    q = torch.empty(r, padk, dtype=torch.int8, device=x.device)
    s = torch.empty(r, dtype=torch.float32, device=x.device)
    kp = triton.next_power_of_2(max(k, padk))
    stages = {0: 0, 4: 1, 16: 2, 64: 3, 256: 4}[rotate]
    grouped = k // rotate * rotate if rotate else 0
    _quant_kernel[(r,)](
        x,
        q,
        s,
        m,
        x.stride(0),
        q.stride(0),
        K=k,
        KP=kp,
        PADK=padk,
        STAGES=stages,
        GROUPED=grouped,
        num_warps=8 if kp >= 4096 else (4 if kp >= 512 else 1),
    )
    return q, s


@cache
def _gemm_kernel(n: int, k: int, bm: int, arch: str):
    bn, bk = BLOCK_N, BLOCK_K
    threads = 256 if bm == 128 else 128
    m = T.dynamic("m")

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "int8"),
        B: T.Tensor((n, k), "int8"),
        C: T.Tensor((m, n), "int32"),
    ):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
            a_s = T.alloc_shared((bm, bk), "int8")
            b_s = T.alloc_shared((bn, bk), "int8")
            acc = T.alloc_fragment((bm, bn), "int32")
            T.clear(acc)
            for kb in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.copy(A[by * bm : (by + 1) * bm, kb * bk : (kb + 1) * bk], a_s)
                T.copy(B[bx * bn : (bx + 1) * bn, kb * bk : (kb + 1) * bk], b_s)
                T.gemm(a_s, b_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(acc, C[by * bm : (by + 1) * bm, bx * bn : (bx + 1) * bn])

    return tilelang.compile(
        main,
        target={"kind": "cuda", "arch": arch},
        pass_configs={"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True},
    )


def arch(device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def gemm(q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``q`` [rows(M), K] and ``w`` [N, K], K and N whole tiles; returns [rows(M), N] int32."""
    m = q.shape[0]
    out = torch.empty(m, w.shape[0], dtype=torch.int32, device=q.device)
    _gemm_kernel(w.shape[0], w.shape[1], block_m(m), arch(q.device))(q, w, out)
    return out


@triton.jit
def _scale_kernel(
    p_ptr,
    a_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    stride_p,
    stride_out,
    HAS_BIAS: tl.constexpr,
    FP32: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * CHUNK + tl.arange(0, CHUNK)
    m = cols < N
    y = tl.load(p_ptr + row * stride_p + cols, mask=m, other=0).to(tl.float32)
    y = y * tl.load(a_ptr + row) * tl.load(w_ptr + cols, mask=m, other=0.0)
    if HAS_BIAS:
        y += tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
    if FP32:
        tl.store(out_ptr + row * stride_out + cols, y, mask=m)
    else:
        tl.store(out_ptr + row * stride_out + cols, y.to(out_ptr.dtype.element_ty), mask=m)


def scale(
    p: torch.Tensor,
    m: int,
    n: int,
    amax: torch.Tensor,
    wmul: torch.Tensor,
    bias: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """The product [>= m, >= n] int32 as ``dtype`` [m, n]: ``p * amax * wmul + bias``."""
    out = torch.empty(m, n, dtype=dtype, device=p.device)
    _scale_kernel[(m, -(-n // SCALE_CHUNK))](
        p,
        amax,
        wmul,
        bias if bias is not None else p,
        out,
        n,
        p.stride(0),
        out.stride(0),
        HAS_BIAS=bias is not None,
        FP32=dtype == torch.float32,
        CHUNK=SCALE_CHUNK,
        num_warps=4,
    )
    return out
