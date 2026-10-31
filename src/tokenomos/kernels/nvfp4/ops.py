"""The NVFP4 producers as plain functions, and the GEMM. Each producer returns the packed codes
``[M, K / 2]`` uint8 and the swizzled E4M3 scales in an int32 buffer (four scale bytes per word,
rows rounded up to 128, scale columns to 4), the layout FlashInfer's FP4 GEMMs read; ``swizzle``
puts a checkpoint's row-major weight scales in the same layout.

``CVT`` picks the E2M1 conversion (``ptx`` the hardware instruction, ``arith`` comparisons); the
tiles are ``TILE`` when set, else ``quant_tile`` / ``norm_tile``.

``gemm`` runs one of FlashInfer's FP4 GEMM kernels, called directly: a CUTLASS tactic (its
tile and schedule, by index) or a b12x plan (sm120 warp-level MMA: tile, swap of A and B,
prefetch). ``BACKEND`` ``table`` takes the kernel the GPU's table names for the weight's
``(N, K)`` and the token count (``bench/profile/tune_nvfp4.py`` writes ``tables/<gpu>.json``;
without one, CUTLASS tactic 0); ``cutlass`` forces the CUTLASS tactic ``TACTIC``; ``b12x`` /
``cudnn`` go through FlashInfer's ``mm_fp4`` with its own pick.
"""

import functools
import json
from pathlib import Path

import torch
import triton

from tokenomos.kernels.mxfp8.ops import gpu_slug
from tokenomos.kernels.nvfp4 import producers

try:
    from flashinfer.gemm import gemm_base, mm_fp4
except ImportError:  # no NVFP4 GEMM; NVFP4 checkpoints then fail at their first linear
    gemm_base = mm_fp4 = None

CVT = {"name": "ptx"}
BACKEND = {"name": "table"}
TACTIC = {"id": 0}
TABLES = Path(__file__).parent / "tables"
# A kernel choice: ("cutlass", tactic index) or ("b12x", (tile_m, tile_n, swap_ab, prefetch))
DEFAULT = ("cutlass", 0)
# (BM rows per program, BK columns per chunk, warps); None: the tile functions pick
TILE: dict = {"value": None}
# The widest row the norm kernel holds in registers (one pass); wider rows take two passes
ONE_PASS_MAX = 8192


def quant_tile(m: int, k: int) -> tuple[int, int, int]:
    """The fastest tiles on the RTX 5090 (widths 2560-9728)."""
    if TILE["value"] is not None:
        return TILE["value"]
    return (2, 512, 4) if m <= 64 else (4, 512, 4)


def norm_tile(m: int, k: int) -> tuple[int, int, int]:
    if TILE["value"] is not None:
        return TILE["value"]
    bk = triton.next_power_of_2(k)
    if bk > ONE_PASS_MAX:
        return (2, 1024, 8)
    warps = 16 if bk >= 8192 or (bk >= 4096 and m >= 1024) else 8 if bk >= 4096 else 4
    return (1, bk, warps)


def scale_tensor(m: int, k: int, device) -> torch.Tensor:
    """The swizzled scale buffer of an ``[m, k]`` operand: int32 ``[ceil(m/128)*128, KBP/4]``,
    ``KBP`` the ``k / 16`` scale columns rounded up to 4."""
    rows = -(-m // 128) * 128
    cols = -(-(k // 16) // 4) * 4
    return torch.empty(rows, cols // 4, dtype=torch.int32, device=device)


def swizzle(scales: torch.Tensor) -> torch.Tensor:
    """Row-major E4M3 scales ``[N, K / 16]`` -> the swizzled int32 buffer of ``scale_tensor``
    (padded rows and columns zero)."""
    n, kb = scales.shape
    rows, kbp = -(-n // 128) * 128, -(-kb // 4) * 4
    padded = torch.zeros(rows, kbp, dtype=torch.uint8, device=scales.device)
    padded[:n, :kb] = scales.view(torch.uint8)
    tiles = padded.view(rows // 128, 4, 32, kbp // 4, 4).permute(0, 3, 2, 1, 4)
    return tiles.contiguous().view(rows, kbp).view(torch.int32)


@functools.cache
def _cutlass(device: torch.device):
    """FlashInfer's CUTLASS FP4 GEMM op for ``device`` and its workspace, called directly: the
    ``mm_fp4`` front end costs ~0.3 ms of host time a call (checks, logging, tactic lookup)."""
    major, minor = torch.cuda.get_device_capability(device)
    runner = gemm_base.get_cutlass_fp4_gemm_module(major, minor).cutlass_fp4_gemm_runner()
    workspace = torch.empty(gemm_base.DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=device)
    return runner._fp4_gemm_runner, workspace


def cutlass_tactics(device) -> int:
    major, minor = torch.cuda.get_device_capability(device)
    runner = gemm_base.get_cutlass_fp4_gemm_module(major, minor).cutlass_fp4_gemm_runner()
    return len(runner.get_valid_tactics(None, None))


@functools.cache
def _b12x(device: torch.device):
    """FlashInfer's b12x NVFP4 runner (sm120; PDL on)."""
    major, minor = torch.cuda.get_device_capability(device)
    return gemm_base._b12x_gemm_fp4_runner(major, minor, True, torch.bfloat16, True)


def b12x_plans(q, s, w, ws, alpha) -> list[tuple]:
    """The b12x plans valid for this shape, each ``(tile_m, tile_n, swap_ab, prefetch)``."""
    out = torch.empty(q.shape[0], w.shape[0], dtype=torch.bfloat16, device=q.device)
    inputs = [q, w.t(), s.view(torch.uint8), ws.view(torch.uint8).t(), alpha, None, out]
    plans = _b12x(q.device).get_valid_tactics(inputs + [None] * 3, None)
    return [(t[0][0], t[0][1], t[2], t[3]) for t in plans]


@functools.cache
def table() -> dict[tuple[int, int], list[tuple[int, tuple]]]:
    """``(N, K)`` -> ``[(largest M, choice), ...]`` in rising M (past the last: the last
    choice), the measured table of this GPU; empty without one."""
    path = TABLES / f"{gpu_slug()}.json"
    if not path.exists():
        return {}
    shapes = json.loads(path.read_text())["shapes"]
    return {
        tuple(int(v) for v in key.split("x")): [(top, _choice(c)) for top, c in ranges]
        for key, ranges in shapes.items()
    }


def _choice(c: list) -> tuple:
    kind, arg = c
    return (kind, tuple(arg) if kind == "b12x" else int(arg))


def choose(n: int, k: int, m: int) -> tuple:
    ranges = table().get((n, k))
    if not ranges:
        return DEFAULT
    for top, c in ranges:
        if m <= top:
            return c
    return ranges[-1][1]


def run(choice: tuple, q, s, w, ws, alpha) -> torch.Tensor:
    """The GEMM on one kernel choice."""
    out = torch.empty(q.shape[0], w.shape[0], dtype=torch.bfloat16, device=q.device)
    kind, arg = choice
    if kind == "cutlass":
        op, workspace = _cutlass(q.device)
        op(q, w, s.view(torch.uint8), ws.view(torch.uint8), alpha, out, workspace, arg)
        return out
    tm, tn, swap, prefetch = arg
    inputs = [q, w.t(), s.view(torch.uint8), ws.view(torch.uint8).t(), alpha, None, out]
    tactic = ((tm, tn), (1, 1), swap, prefetch, "sm120", None)
    return _b12x(q.device).forward(inputs + [None] * 3, tactic=tactic)


def gemm(
    q: torch.Tensor, s: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """Codes ``[M, K / 2]`` + scales times weight codes ``[N, K / 2]`` + scales, times ``alpha``
    (the product of both tensor scales, a 1-element float32 tensor) -> bf16 ``[M, N]``."""
    name = BACKEND["name"]
    if name == "table":
        return run(choose(w.shape[0], q.shape[1] * 2, q.shape[0]), q, s, w, ws, alpha)
    if name == "cutlass":
        return run(("cutlass", TACTIC["id"]), q, s, w, ws, alpha)
    return mm_fp4(
        q,
        w.t(),
        s.view(torch.uint8),
        ws.view(torch.uint8).t(),
        alpha,
        torch.bfloat16,
        block_size=16,
        backend=BACKEND["name"],
    )


def _alloc(m: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.empty(m, k // 2, dtype=torch.uint8, device=device)
    return q, scale_tensor(m, k, device)


def _grid(m: int, bm: int) -> tuple[int]:
    return (triton.cdiv(-(-m // 128) * 128, bm),)


def nvfp4_quant(x: torch.Tensor, gs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` [..., K] -> codes and scales with the divisor-style tensor scale ``gs``
    (``448 * 6 / amax``, a 1-element float32 tensor)."""
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    q, s = _alloc(m, k, x.device)
    if m:
        bm, bk, warps = quant_tile(m, k)
        grid = (_grid(m, bm)[0], triton.cdiv(k, bk))
        args = (x2, q, s.view(torch.uint8), gs, m, k, x2.stride(0))
        meta = {"BM": bm, "BK": bk, "PTX": CVT["name"] == "ptx"}
        producers.quant[grid](*args, **meta, num_warps=warps)
    return q, s


ACT_MODES = {"plain": 0, "mul": 1, "silu_mul": 2, "gate": 3, "gelu_mul": 4}


def act(
    a: torch.Tensor,
    b: torch.Tensor | None,
    mode: str,
    gs: torch.Tensor,
    group: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``producers.act``: ``mode`` ``plain`` / ``mul`` (``a * b``) / ``silu_mul`` (``silu`` of
    ``a``'s first half times its second) / ``gate`` (``a * sigmoid(b[:, col // group])``) /
    ``gelu_mul`` (as ``silu_mul`` with tanh GELU), quantized with tensor scale ``gs``. Row
    strides free."""
    k = a.shape[1] // 2 if mode in ("silu_mul", "gelu_mul") else a.shape[1]
    m = a.shape[0]
    q, s = _alloc(m, k, a.device)
    if m:
        bm, bk, warps = quant_tile(m, k)
        grid = (_grid(m, bm)[0], triton.cdiv(k, bk))
        bb = b if b is not None else a
        producers.act[grid](
            a,
            bb,
            q,
            s.view(torch.uint8),
            gs,
            m,
            k,
            a.stride(0),
            bb.stride(0),
            MODE=ACT_MODES[mode],
            GROUP=group,
            BM=bm,
            BK=bk,
            PTX=CVT["name"] == "ptx",
            num_warps=warps,
        )
    return q, s


def rmsnorm_nvfp4(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    gs0: torch.Tensor,
    gs1: torch.Tensor | None,
    round_sum: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``rmsnorm(x + residual) * weight`` (no add when ``residual`` is None), quantized with
    ``gs0`` and, when given, ``gs1``: ``(q0, s0, q1, s1, residual out)``, the unused ones
    empty. ``round_sum``: statistics of the sum rounded to bf16, as the Hugging Face models
    take them (False: the float32 sum, vLLM's ``fused_add_rms_norm``)."""
    k = x.shape[-1]
    x2 = x.reshape(-1, k).contiguous()
    r2 = residual.reshape(-1, k).contiguous() if residual is not None else x2
    m = x2.shape[0]
    res = torch.empty_like(x2) if residual is not None else x2.new_empty(0)
    q0, s0 = _alloc(m, k, x.device)
    q1, s1 = _alloc(m, k, x.device) if gs1 is not None else (q0.new_empty(0), s0.new_empty(0))
    if m:
        bm, bk, warps = norm_tile(m, k)
        args = (
            x2,
            r2,
            weight,
            res,
            q0,
            s0.view(torch.uint8),
            gs0,
            q1 if gs1 is not None else q0,
            (s1 if gs1 is not None else s0).view(torch.uint8),
            gs1 if gs1 is not None else gs0,
            m,
            k,
            x2.stride(0),
            r2.stride(0),
            eps,
        )
        meta = {
            "HAS_RES": residual is not None,
            "TWO": gs1 is not None,
            "ONE": bk >= k,
            "WBF16": weight.dtype != torch.float32,
            "RSUM": round_sum,
            "BM": bm,
            "BK": bk,
            "PTX": CVT["name"] == "ptx",
        }
        producers.add_rmsnorm_quant[_grid(m, bm)](*args, **meta, num_warps=warps)
    return q0, s0, q1, s1, res.view_as(x) if residual is not None else res
