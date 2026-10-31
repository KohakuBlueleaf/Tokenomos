"""MXFP8 producers and GEMM as plain functions.

Producers return the e4m3 values ``[M, K]`` and the swizzled scales, flat uint8, rows rounded up
to 128 (``producers``). ``gemm`` multiplies them by a weight in the same form (``[N, K]`` e4m3,
scales swizzled over ``N`` by ``swizzle``) on the backend the GPU's table names for its
``(N, K)`` and token count: ``cublas`` (``F.scaled_mm``, block-32 scales, ``SWIZZLE_32_4_4``) or
FlashInfer's ``b12x`` (sm120 warp-level MMA with small-M tiles).
"""

import functools
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from tokenomos.kernels.mxfp8 import producers

try:
    from flashinfer.gemm import mm_mxfp8
except ImportError:  # the b12x backend is then off; cuBLAS takes every shape
    mm_mxfp8 = None

BLOCK = torch._C._ScalingType.BlockWise1x32
SWIZZLE = torch._C._SwizzleType.SWIZZLE_32_4_4
TABLES = Path(__file__).parent / "tables"
# (N, K) -> [(largest M, backend), ...] in rising M, past the last: cuBLAS. Used when no measured
# table exists for the GPU (``bench/profile/tune_gemm.py`` writes ``tables/<gpu>.json``).
GEMM: dict[tuple[int, int], list[tuple[int, str]]] = {
    (2048, 2048): [(512, "b12x")],
    (3072, 2048): [(256, "b12x")],
    (2048, 7680): [(512, "b12x")],
}


def gpu_slug() -> str:
    return re.sub(r"[^a-z0-9]+", "-", torch.cuda.get_device_name().lower()).strip("-")


@functools.cache
def table() -> dict[tuple[int, int], list[tuple[int, str]]]:
    """The measured table of this GPU, else ``GEMM``."""
    path = TABLES / f"{gpu_slug()}.json"
    if not path.exists():
        return GEMM
    shapes = json.loads(path.read_text())["shapes"]
    return {
        tuple(int(v) for v in key.split("x")): [(top, name) for top, name in ranges]
        for key, ranges in shapes.items()
    }


def scale_numel(m: int, k: int) -> int:
    return -(-m // 128) * 128 * (k // 128) * 4


def swizzle(scales: torch.Tensor) -> torch.Tensor:
    """Row-major ue8m0 scales ``[N, K / 32]`` -> the flat 128x4 swizzled layout, rows padded to
    128 with zero scales (``K`` a multiple of 128)."""
    n, kb = scales.shape
    rows = -(-n // 128) * 128
    padded = scales.new_zeros(rows, kb)
    padded[:n] = scales
    tiles = padded.view(rows // 128, 4, 32, kb // 4, 4).permute(0, 3, 2, 1, 4)
    return tiles.contiguous().view(-1)


def _block(k: int) -> int:
    """The widest power of two up to 1024 dividing ``k`` (a column chunk of the act kernel)."""
    b = 1024
    while k % b:
        b //= 2
    return b


def _act(a: torch.Tensor, b: torch.Tensor | None, mode: int, group: int, k: int):
    m = a.shape[0]
    q = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=a.device)
    s = torch.empty(scale_numel(m, k), dtype=torch.uint8, device=a.device)
    if m:
        producers._act_mx_kernel[(-(-m // 128) * 128,)](
            a,
            b if b is not None else a,
            q,
            s,
            m,
            k,
            k // 128,
            a.stride(0),
            b.stride(0) if b is not None else 0,
            q.stride(0),
            MODE=mode,
            GROUP=group,
            BLOCK=_block(k),
            num_warps=4,
        )
    return q, s


def quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _act(x, None, 0, 1, x.shape[1])


def mul(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``a * b``, quantized; ``a`` may be a column slice (row stride free)."""
    return _act(a, b, 1, 1, a.shape[1])


def silu_mul(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``silu(x[:, :K]) * x[:, K:]``, quantized."""
    return _act(x, None, 2, 1, x.shape[1] // 2)


def gelu_mul(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``gelu_tanh(x[:, :K]) * x[:, K:]``, quantized."""
    return _act(x, None, 4, 1, x.shape[1] // 2)


def gate(x: torch.Tensor, g: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``x * sigmoid(g[:, col // group])``, quantized (a per-head output gate)."""
    return _act(x, g, 3, group, x.shape[1])


def backend(n: int, k: int, m: int) -> str:
    if mm_mxfp8 is None:
        return "cublas"
    for top, name in table().get((n, k), ()):
        if m <= top:
            return name
    return "cublas"


def gemm(q: torch.Tensor, s: torch.Tensor, w: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """``[M, K]`` values + scales times weight ``[N, K]`` + swizzled scales -> bf16 ``[M, N]``."""
    m, n = q.shape[0], w.shape[0]
    if backend(n, w.shape[1], m) == "b12x":
        return mm_mxfp8(q, w.t(), s, ws, out_dtype=torch.bfloat16, backend="b12x")
    return F.scaled_mm(
        q,
        w.t(),
        s.view(torch.float8_e8m0fnu),
        BLOCK,
        ws.view(torch.float8_e8m0fnu),
        BLOCK,
        SWIZZLE,
        SWIZZLE,
        output_dtype=torch.bfloat16,
    )
