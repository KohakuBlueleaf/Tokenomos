"""Models: the framework (``contract``, ``blocks``, ``load``) and one declaration per family.

``ARCHS`` maps a checkpoint's Hugging Face ``architectures`` name to its declaration; ``register``
adds one (a new family is one ``Architecture`` built from the blocks), ``load`` builds a folder's
model.
"""

import torch

from tokenomos.core import floats
from tokenomos.kernels.linear import skinny
from tokenomos.models.blocks.linear import Linear
from tokenomos.models.contract import Architecture
from tokenomos.models.gemma4.arch import Gemma4
from tokenomos.models.lfm3.arch import Lfm3
from tokenomos.models.load.checkpoint import Checkpoint
from tokenomos.models.qwen35.arch import Qwen35

ARCHS: dict[str, Architecture] = {}


def register(arch: Architecture) -> None:
    for name in arch.names:
        ARCHS[name] = arch


def configure(
    path: str,
    int8: bool = False,
    rotate: int = 0,
    work: str | None = None,
    route: str | None = None,
    **kw,
) -> Architecture:
    """The declaration of folder ``path``'s architecture, built with ``kw`` (e.g. ``head``).
    ``int8``: the decoder's bf16 linears become W8A8 at load, ``rotate`` their rotation group.
    ``work``: the work dtype, ``bf16``, ``fp16`` or ``fp32`` (None: ``floats.default``).
    ``route``: fp32's matmul, ``tilelang`` or ``cublas`` (None: ``floats.ROUTE``)."""
    name = Checkpoint(path).architecture
    if name not in ARCHS:
        raise ValueError(f"no declaration for {name!r}; known: {sorted(ARCHS)}")
    arch = type(ARCHS[name])(**kw)
    arch.int8, arch.rotate, arch.work, arch.route = int8, rotate, work, route
    return arch


def _to_int8(model: torch.nn.Module, rotate: int) -> None:
    """The decoder's bf16 linears as W8A8; the vision tower, the head and the skinny ones stay."""
    for name, m in model.named_modules():
        tower = "vision" in name or name.startswith(("encoder", "patch", "merger"))
        small = isinstance(m, Linear) and m.n <= skinny.MAX_N
        head = "head" in name.rsplit(".", 1)[-1]
        if isinstance(m, Linear) and m.fmt == "bf16" and not (tower or small or head):
            m.to_int8(rotate)


def _set_route(model: torch.nn.Module, route: str) -> None:
    for m in model.modules():
        if isinstance(m, Linear) and m.fmt == "bf16":
            m.set_route(route)


def load(path: str, device: torch.device | str = "cuda", arch: Architecture | None = None):
    """The model in checkpoint folder ``path``, by its ``architectures`` name unless ``arch``
    (a configured declaration, e.g. ``Lfm3(fuse_quant=False)``) is given."""
    ckpt = Checkpoint(path)
    if arch is None:
        if ckpt.architecture not in ARCHS:
            raise ValueError(f"no declaration for {ckpt.architecture!r}; known: {sorted(ARCHS)}")
        arch = ARCHS[ckpt.architecture]
    floats.set_work(getattr(arch, "work", None) or floats.default(device))
    if floats.work() != torch.bfloat16 and getattr(arch, "head", "bf16") != "bf16":
        raise ValueError(f"head {arch.head}: quantized formats need the bf16 work dtype")
    with torch.no_grad():
        model = arch.build(ckpt, torch.device(device))
        if getattr(arch, "int8", False):
            _to_int8(model, arch.rotate)
        if floats.work() == torch.float32:
            _set_route(model, getattr(arch, "route", None) or floats.ROUTE)
    return model


register(Lfm3())
register(Qwen35())
register(Gemma4())
