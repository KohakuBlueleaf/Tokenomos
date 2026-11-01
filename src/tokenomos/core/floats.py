"""The work dtype: bf16 from sm80 up, fp16 below (PTX has no bf16 there), or as configured.

``fp32``: activations, state, embeddings and every kernel store in float32, no 16-bit rounding;
linear weights stay fp16 (``weights``) and a matmul runs fp32 -> fp16, float32 accumulation,
-> fp32 on the ``route`` a load picks: ``ROUTE``, ``cublas`` by default, 12-40% faster than
``tilelang`` on an RTX 3090 at 1.7-3.2x the logit gap. ``load`` sets the work dtype before it
builds a model. It decides at build time what weights and state pools hold; forward code and
kernels read their dtype from their tensors (``tl_dtype`` gives a kernel's ``tl.constexpr``). Only
``default`` touches CUDA: spawned prep workers import this module.
"""

from functools import cache

import torch
import triton.language as tl

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}
WORK = {"dtype": torch.bfloat16}
ROUTES = ("tilelang", "cublas")
ROUTE = "cublas"


def work() -> torch.dtype:
    return WORK["dtype"]


def weights() -> torch.dtype:
    """The dtype of a linear's weight."""
    return matmul(work())


def matmul(dtype: torch.dtype) -> torch.dtype:
    """The operand dtype of a tensor-core product between ``dtype`` tensors."""
    return torch.float16 if dtype == torch.float32 else dtype


def set_work(name: str) -> None:
    if name not in DTYPES:
        raise ValueError(f"work dtype {name!r}: one of {sorted(DTYPES)}")
    WORK["dtype"] = DTYPES[name]


def default(device: torch.device | str) -> str:
    """The work dtype's name for ``device``."""
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        return "fp16"
    return "bf16"


@cache
def is_volta(device: torch.device | str) -> bool:
    """Whether ``device`` is sm70: no MMA from Triton, its ``tl.dot`` runs on vector units."""
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (7, 0)


def tl_dtype(dtype: torch.dtype) -> tl.dtype:
    """The Triton dtype of the work dtype ``dtype``."""
    return TL[dtype]
