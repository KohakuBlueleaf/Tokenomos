"""GDN output norm: per head, ``rmsnorm(o) * w * silu(z)``, rounded as the Hugging Face
``RMSNormGated`` rounds: the normed value to the 16-bit dtype ``DT`` of ``o``, times the 16-bit
weight to ``DT``, times ``silu(z)`` in float32, to ``DT``. ``norm_gate_nvfp4`` quantizes the
result for an NVFP4 consumer in the same pass (the rows padded to 128, as the NVFP4 producers
write them)."""

import torch
import triton
import triton.language as tl

from tokenomos.core.floats import tl_dtype
from tokenomos.kernels.nvfp4 import ops as nv
from tokenomos.kernels.nvfp4.producers import _quant_store, _zero_scale_tail
from tokenomos.kernels.rounding import round_as


@triton.jit
def _gated(
    o_ptr, z_ptr, w_ptr, rows, mask, h, eps, stride_o, stride_z, V: tl.constexpr, DT: tl.constexpr
):
    cols = h * V + tl.arange(0, V)
    x = tl.load(o_ptr + rows[:, None] * stride_o + cols[None, :], mask=mask, other=0.0)
    x = x.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / V + eps)
    w = tl.load(w_ptr + tl.arange(0, V)).to(tl.float32)
    y = round_as(round_as(x * rstd[:, None], DT) * w[None, :], DT)
    z = tl.load(z_ptr + rows[:, None] * stride_z + cols[None, :], mask=mask, other=0.0)
    z = z.to(tl.float32)
    return round_as(y * z * tl.sigmoid(z), DT)


@triton.jit
def _norm_gate_kernel(
    o_ptr,
    z_ptr,
    w_ptr,
    y_ptr,
    M,
    eps,
    stride_o,
    stride_z,
    stride_y,
    V: tl.constexpr,
    DT: tl.constexpr,
):
    rows = tl.program_id(0) + tl.arange(0, 1)
    h = tl.program_id(1)
    mask = (rows < M)[:, None]
    y = _gated(o_ptr, z_ptr, w_ptr, rows, mask, h, eps, stride_o, stride_z, V, DT)
    cols = h * V + tl.arange(0, V)
    tl.store(y_ptr + rows[:, None] * stride_y + cols[None, :], y.to(DT), mask=mask)


@triton.jit
def _norm_gate_nvfp4_kernel(
    o_ptr,
    z_ptr,
    w_ptr,
    Q,
    S,
    GS,
    M,
    K,
    eps,
    stride_o,
    stride_z,
    V: tl.constexpr,
    BM: tl.constexpr,
    PTX: tl.constexpr,
    DT: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    h = tl.program_id(1)
    y = _gated(o_ptr, z_ptr, w_ptr, rows, (rows < M)[:, None], h, eps, stride_o, stride_z, V, DT)
    KB = K // 16
    KBP = (KB + 3) // 4 * 4
    _quant_store(y, rows, h * V, M, KB, KBP, Q, S, tl.load(GS), BM, V, PTX)
    if (KBP != KB) & (h == 0):
        _zero_scale_tail(rows, KB, KBP, S, BM)


def norm_gate(o: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``o`` / ``z`` [T, HV * V] (row strides free), ``weight`` [V] -> [T, HV * V] in ``o``'s
    16-bit dtype."""
    t, width = o.shape
    v = weight.shape[0]
    y = torch.empty(t, width, dtype=o.dtype, device=o.device)
    if t:
        _norm_gate_kernel[(t, width // v)](
            o,
            z,
            weight,
            y,
            t,
            eps,
            o.stride(0),
            z.stride(0),
            y.stride(0),
            V=v,
            DT=tl_dtype(o.dtype),
            num_warps=1,
        )
    return y


def norm_gate_nvfp4(o, z, weight, eps: float, gs: torch.Tensor):
    """As ``norm_gate``, quantized with tensor scale ``gs``: (codes, swizzled scales)."""
    t, width = o.shape
    v = weight.shape[0]
    q, s = nv._alloc(t, width, o.device)
    if t:
        bm = 4
        _norm_gate_nvfp4_kernel[(-(-t // 128) * 128 // bm, width // v)](
            o,
            z,
            weight,
            q,
            s.view(torch.uint8),
            gs,
            t,
            width,
            eps,
            o.stride(0),
            z.stride(0),
            V=v,
            BM=bm,
            PTX=nv.CVT["name"] == "ptx",
            DT=tl_dtype(o.dtype),
            num_warps=2,
        )
    return q, s
