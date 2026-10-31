"""Reference running without the engine: whole sequences packed into one ``Step``, fresh zero
states for the layers that keep one (row 0 the null row, sequences on rows 1..S), the ``dense``
attention (no KV cache). Tests and parity checks use it; the engine core builds its own steps.
"""

import importlib

import torch

from tokenomos.core.floats import work
from tokenomos.models import imageprep
from tokenomos.models.blocks.decoder import Decoder
from tokenomos.models.blocks.linear import Linear
from tokenomos.models.contract import ConvState, RecurrentState, StateKey, StateSpec, Step


def prefill_step(lengths: list[int], specs: dict[StateKey, StateSpec], device) -> Step:
    s = len(lengths)
    cu_host = [0]
    for n in lengths:
        cu_host.append(cu_host[-1] + n)
    lens = torch.tensor(lengths, device=device)
    states = {}
    for key, spec in specs.items():
        match spec:
            case ConvState(dim=dim, width=width):
                states[key] = torch.zeros(s + 1, dim, width - 1, dtype=work(), device=device)
            case RecurrentState(shape=shape, dtype=dtype):
                states[key] = torch.zeros(s + 1, *shape, dtype=dtype, device=device)
    return Step(
        positions=torch.cat([torch.arange(n, device=device) for n in lengths]),
        cu=torch.tensor(cu_host, dtype=torch.int32, device=device),
        tok_seq=torch.repeat_interleave(torch.arange(s, device=device), lens).to(torch.int32),
        n_tok=torch.tensor([cu_host[-1]], dtype=torch.int32, device=device),
        slot=torch.arange(1, s + 1, dtype=torch.int32, device=device),
        init=torch.zeros(s, dtype=torch.int32, device=device),
        states=states,
        meta={
            "dense_cu": cu_host,
            "cu_host": cu_host,
            "put": lambda v, dtype: torch.tensor(v, dtype=dtype, device=device),
        },
    )


def dequantize(model) -> None:
    """Every MXFP8 linear of ``model`` replaced by its 16-bit dequantization, fused producers off:
    the same weights without activation quantization (the reference for the MXFP8 path)."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, Linear) and child.is_mx:
                setattr(module, name, child.dequantized())
        if isinstance(module, Decoder):
            module.quant_mixer = module.quant_ffn = None


def mx_layers(model) -> int:
    return sum(isinstance(m, Linear) and m.is_mx for m in model.modules())


@torch.no_grad()
def multimodal_logits(
    model, ids: list[int], images: list[torch.Tensor], device, start: int = 0
) -> torch.Tensor:
    """Teacher-forced logits [len(ids) - start, vocab] (float32) of one sequence with the image
    features on its placeholder rows (and the processor's RoPE positions when it defines
    ``rope``)."""
    step = prefill_step([len(ids)], model.state_specs(), device)
    t = torch.tensor(ids, device=device)
    step.ids = t
    x = model.embed(t)
    if images:
        feats = model.encoders["image"]([im.to(device) for im in images])
        x[t == model.image_token] = feats.to(x.dtype)
        rope = getattr(importlib.import_module(model.processor), "rope", None)
        if rope is not None:
            step.rope_pos = rope(ids, images, model.image_token).to(device, torch.int64)
        if getattr(model, "image_runs", False):
            run = torch.full((len(ids),), -1, dtype=torch.int32, device=device)
            for i, (a, b) in enumerate(imageprep.runs(ids, model.image_token)):
                run[a:b] = i
            step.meta["blocks"] = (run, None)
    return model.logits(model(x, step), torch.arange(start, len(ids), device=device))


@torch.no_grad()
def forward_logits(model, sequences: list[list[int]], device) -> list[torch.Tensor]:
    """Teacher-forced logits [len, vocab] (float32) of each token sequence, all in one pass."""
    step = prefill_step([len(s) for s in sequences], model.state_specs(), device)
    ids = torch.tensor([t for s in sequences for t in s], device=device)
    step.ids = ids
    logits = model.logits(model(model.embed(ids), step))
    return list(torch.split(logits, [len(s) for s in sequences]))
