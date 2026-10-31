"""Decoder layer and the stack around it.

``Decoder``: ``mixer`` (attention, short conv, ...) and ``ffn``, each behind a pre-norm and, with
sandwich norms, followed by a post-norm; optionally an ``extra`` block on the stream after the
ffn (Gemma 4's per-layer input, its own post-norm) and a ``scalar`` multiplying the stream at
the layer's end. The residual stream is carried as ``(h, residual)``: a block's output, its
post-norm, the add into the stream, the layer scalar and the next pre-norm are one kernel
(``fused.add_norm``), which also writes the quantized activation when the consumer is MXFP8;
an NVFP4 consumer takes it from ``nvfp4.ops.rmsnorm_nvfp4`` (add + norm + quantize).

``Stack``: embedding (optionally scaled), an optional ``prelude`` filling per-step inputs the
layers read (``step.meta``), decoder layers, final norm (fused with the last add), head (a
``Linear``, or the tied 16-bit embedding; optional final logit softcap, optional suppressed
tokens).
"""

import torch
import torch.nn.functional as F
from torch import nn

from tokenomos.kernels.norm import fused
from tokenomos.kernels.nvfp4 import ops as nv
from tokenomos.models.blocks.linear import Linear
from tokenomos.models.blocks.norm import RMSNorm
from tokenomos.models.contract import StateKey, StateSpec, Step


def add_norm(
    norm: RMSNorm | None,
    h: torch.Tensor,
    residual: torch.Tensor | None,
    quant: Linear | None,
    write: bool,
    pre: RMSNorm | None = None,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, tuple | None, torch.Tensor]:
    """``(residual + pre(h)) * scale`` (``pre``: the previous block's post-norm, ``scale`` the
    previous layer's scalar) through ``norm`` (None: the stream only), one kernel: (normed 16-bit
    or None when ``quant`` and not ``write``, (values, scales) for the ``quant`` linear's format
    when given, the new residual)."""
    if quant is not None and quant.fmt == "nvfp4":
        q, s, _, _, res = nv.rmsnorm_nvfp4(h, residual, norm.weight, norm.eps, quant.igs, None)
        return None, (q, s), res if residual is not None else h
    return fused.add_norm(
        h,
        residual,
        norm.weight if norm is not None else None,
        norm.eps if norm is not None else 0.0,
        (pre.weight, pre.eps, pre.round_normed) if pre is not None else None,
        write or quant is None,
        quant is not None,
        norm.round_normed if norm is not None else True,
        scale,
    )


def _fused_input(block, norm: RMSNorm, pre: RMSNorm | None, fuse: bool) -> Linear | None:
    """The block's input linear when its quantized input comes out of the norm kernel: MXFP8
    (width a multiple of 128), or NVFP4 (no pending post-norm, the bf16 input unused, the
    norm's rounding the NVFP4 kernel's: a bf16 weight rounds the normed row first)."""
    linear = getattr(block, "input_linear", None)
    if not fuse or linear is None or not linear.quantizes:
        return None
    width = norm.weight.shape[0]
    if linear.is_mx:
        return linear if width % 128 == 0 else None
    ok = (
        pre is None
        and not getattr(block, "needs_bf16_input", False)
        and width % 16 == 0
        and norm.round_normed == (norm.weight.dtype != torch.float32)
    )
    return linear if ok else None


class Decoder(nn.Module):
    def __init__(
        self,
        mixer: nn.Module,
        ffn: nn.Module,
        pre_mixer: RMSNorm,
        pre_ffn: RMSNorm,
        post_mixer: RMSNorm | None = None,
        post_ffn: RMSNorm | None = None,
        fuse_quant: bool = True,
        extra: nn.Module | None = None,
        scalar: torch.Tensor | None = None,
    ):
        """``extra``: ``extra(stream, step, index)`` -> a block output added to the stream
        through ``extra.post_norm``; ``scalar``: a one-element tensor multiplying the stream."""
        super().__init__()
        self.mixer, self.ffn = mixer, ffn
        self.pre_mixer, self.pre_ffn = pre_mixer, pre_ffn
        self.post_mixer, self.post_ffn = post_mixer, post_ffn
        self.extra = extra
        self.scalar = nn.Parameter(scalar, requires_grad=False) if scalar is not None else None
        # the pending post-norm: the previous layer's post-ffn norm for the mixer (a sandwich
        # stack has one in every layer, so this layer's stands for it), this layer's post-mixer
        # norm for the ffn; a layer scalar keeps the norm unfused (the NVFP4 producer has none)
        pending = post_ffn if scalar is None else (post_ffn or pre_mixer)
        self.quant_mixer = _fused_input(mixer, pre_mixer, pending, fuse_quant)
        self.quant_ffn = _fused_input(ffn, pre_ffn, post_mixer, fuse_quant)

    def forward(
        self,
        h: torch.Tensor,
        residual: torch.Tensor | None,
        pending: tuple | None,
        step: Step,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """``pending``: (post-norm, scalar) of the previous layer, still to apply to ``h`` and
        the stream. Returns this layer's output, the residual, and its own pending pair."""
        pnorm, pscale = pending if pending is not None else (None, None)
        write = self.quant_mixer is None or self.mixer.needs_bf16_input
        x, xq, residual = add_norm(
            self.pre_mixer, h, residual, self.quant_mixer, write, pnorm, pscale
        )
        h = self.mixer(x, xq, step, index)
        x, xq, residual = add_norm(
            self.pre_ffn, h, residual, self.quant_ffn, False, self.post_mixer
        )
        h = self.ffn(x, xq)
        if self.extra is None:
            return h, residual, (self.post_ffn, self.scalar)
        _, _, residual = add_norm(None, h, residual, None, False, self.post_ffn)
        return self.extra(residual, step, index), residual, (self.extra.post_norm, self.scalar)


def _fp32(t: torch.Tensor) -> bool:
    return t.dtype == torch.float32


class Stack(nn.Module):
    def __init__(
        self,
        embedding: torch.Tensor,
        layers: list[Decoder],
        final_norm: RMSNorm,
        head: Linear | None = None,
        embed_scale: float | None = None,
        logit_softcap: float | None = None,
        prelude: nn.Module | None = None,
        suppress: list[int] | None = None,
    ):
        """``head`` None: the 16-bit ``embedding`` (tied). ``embed_scale``: multiplies the embedded
        rows (Gemma's ``sqrt(hidden)``); ``logit_softcap``: ``c * tanh(logits / c)``;
        ``prelude(h, step)``: run before the layers (fills ``step.meta``); ``suppress``: token
        ids whose logits are always ``-inf``."""
        super().__init__()
        self.embedding = nn.Parameter(embedding, requires_grad=False)
        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm
        # a tied fp32 embedding: the logits take the fp16 matmul, on a weight of their own
        self.head = Linear("bf16", embedding) if head is None and _fp32(embedding) else head
        self.prelude = prelude
        dev = embedding.device
        self.suppress = torch.tensor(suppress, device=dev) if suppress else None
        # modality -> encoder producing input rows for that modality's placeholder tokens
        self.encoders = nn.ModuleDict()
        self.image_token: int | None = None
        # module path of the image processor (``prepare(PIL) -> uint8 [H, W, 3]``), or None
        self.processor: str | None = None
        self.embed_scale = embed_scale
        self.logit_softcap = logit_softcap

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        h = F.embedding(ids, self.embedding)
        if self.embed_scale is not None:
            h = h * torch.tensor(self.embed_scale, dtype=h.dtype)
        return h

    def forward(self, h: torch.Tensor, step: Step) -> torch.Tensor:
        if self.prelude is not None:
            self.prelude(h, step)
        residual, pending = None, None
        for i, layer in enumerate(self.layers):
            h, residual, pending = layer(h, residual, pending, step, i)
        out, _, _ = add_norm(self.final_norm, h, residual, None, True, *pending)
        return out

    def logits(
        self, hidden: torch.Tensor, rows: torch.Tensor | None = None, fp32: bool = True
    ) -> torch.Tensor:
        """Logits of ``hidden`` (only ``rows`` of it when given), float32 or the head's 16-bit."""
        if rows is not None:
            hidden = hidden[rows]
        out = self.head(hidden) if self.head is not None else F.linear(hidden, self.embedding)
        if fp32:
            out = out.float()
        if self.logit_softcap is not None:
            out = self.logit_softcap * torch.tanh(out / self.logit_softcap)
        if self.suppress is not None:
            out = out.index_fill(-1, self.suppress, float("-inf"))
        return out

    def state_specs(self) -> dict[StateKey, StateSpec]:
        return {
            (i, name): spec
            for i, layer in enumerate(self.layers)
            for name, spec in layer.mixer.state_specs().items()
        }
