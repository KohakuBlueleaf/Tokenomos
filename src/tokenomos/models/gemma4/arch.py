"""Gemma 4 text stack (E4B, 12B, 31B).

Sandwich layers of attention and a GeGLU MLP, the stream times ``layer_scalar`` at each layer's
end. Sliding layers: heads of 256, RoPE over the whole head. Full layers: heads of 512, RoPE over
64 frequencies; with ``attention_k_eq_v`` V is the raw K projection. Scores unscaled; V takes a
weightless norm and no RoPE.

The last ``num_kv_shared_layers`` keep only Q and read the K / V of the last unshared layer of
their type. E4B adds a per-layer input built before the layers (``PerLayerInputs``) and gated
into each one (``PerLayerBlock``).

Image runs see themselves in both directions on ``image_runs`` layers: ``sliding`` (31B) or
``all`` (12B, as Hugging Face's forward does); ``auto`` picks per family.
"""

import json
import math

import torch
import torch.nn.functional as F
from torch import nn

from tokenomos.core.floats import work
from tokenomos.kernels.mxfp8 import ops as mx
from tokenomos.models.blocks.attention import Attention
from tokenomos.models.blocks.decoder import Decoder, Stack
from tokenomos.models.blocks.ffn import SwiGLU
from tokenomos.models.blocks.linear import Linear
from tokenomos.models.blocks.norm import RMSNorm
from tokenomos.models.blocks.rope import Rope
from tokenomos.models.contract import Step
from tokenomos.models.gemma4 import vision

ROOT = "model.language_model."
PLACEHOLDERS = ("image_token_id", "video_token_id", "audio_token_id")


class PerLayerInputs(nn.Module):
    """Fills ``step.meta["ple"]`` [T, layers, D]: ``(norm(project(h) * H^-0.5) + table[ids] *
    sqrt(D)) * 2^-0.5``, placeholders reading row 0 of the table."""

    def __init__(self, table, project: Linear, norm: RMSNorm, layers: int, placeholders):
        super().__init__()
        self.table = nn.Parameter(table, requires_grad=False)
        self.project, self.norm = project, norm
        self.layers, self.placeholders = layers, placeholders

    def forward(self, h: torch.Tensor, step: Step) -> None:
        n, width = h.shape
        dim = self.norm.weight.shape[0]
        ids = step.ids
        for t in self.placeholders:
            ids = torch.where(ids == t, 0, ids)
        table = F.embedding(ids, self.table) * dim**0.5
        p = self.project(h) * width**-0.5
        p = self.norm(p.view(n, self.layers, dim))
        step.meta["ple"] = (p + table.view(n, self.layers, dim)) * 2**-0.5


class PerLayerBlock(nn.Module):
    """``project(gelu_tanh(gate(stream)) * ple[:, index])``, added through ``post_norm``."""

    def __init__(self, gate: Linear, project: Linear, post_norm: RMSNorm):
        super().__init__()
        self.gate, self.project, self.post_norm = gate, project, post_norm

    def forward(self, stream: torch.Tensor, step: Step, index: int) -> torch.Tensor:
        g = F.gelu(self.gate(stream), approximate="tanh")
        return self.project(g * step.meta["ple"][:, index])


class Gemma4:
    names = (
        "Gemma4ForConditionalGeneration",
        "Gemma4ForCausalLM",
        "Gemma4UnifiedForConditionalGeneration",
    )

    def __init__(
        self,
        attention_backend: str = "dense",
        fuse_quant: bool = True,
        head: str = "bf16",
        vision: bool = True,
        image_runs: str = "auto",
    ):
        """``head``: ``bf16`` (tied), ``mxfp8`` or ``nvfp4`` (the embedding quantized at load).
        ``image_runs``: the layers where image runs are bidirectional, when the model has them
        (``auto``, ``sliding``, ``all``)."""
        if head not in ("bf16", "mxfp8", "nvfp4"):
            raise ValueError(f"head format {head!r}")
        if image_runs not in ("auto", "sliding", "all"):
            raise ValueError(f"image_runs {image_runs!r}")
        self.attention_backend = attention_backend
        self.fuse_quant = fuse_quant
        self.head = head
        self.vision = vision
        self.image_runs = image_runs

    def build(self, ckpt, device) -> Stack:
        cfg = ckpt.text_config
        eps = cfg["rms_norm_eps"]
        kinds = cfg["layer_types"]
        count = cfg["num_hidden_layers"]
        unified = ckpt.architecture.startswith("Gemma4Unified")
        k_eq_v = cfg.get("attention_k_eq_v", False)
        rp = cfg["rope_parameters"]
        hd, gd = cfg["head_dim"], cfg["global_head_dim"]
        partial = rp["full_attention"]["partial_rotary_factor"]
        rope_full = Rope(gd, int(partial * gd // 2), rp["full_attention"]["rope_theta"], device)
        rope_slide = Rope(hd, hd // 2, rp["sliding_attention"]["rope_theta"], device)
        rule = None
        if cfg.get("use_bidirectional_attention") == "vision":
            rule = self.image_runs
            if rule == "auto":
                rule = "all" if unified else "sliding"
        # layers from ``first`` on read the last unshared layer of their type
        first = count - cfg.get("num_kv_shared_layers", 0)
        source = {t: max(i for i in range(first) if kinds[i] == t) for t in set(kinds[first:])}
        last_reader = {t: max(i for i in range(first, count) if kinds[i] == t) for t in source}
        ple_dim = cfg.get("hidden_size_per_layer_input", 0)

        def norm(name: str) -> RMSNorm:
            return RMSNorm.load(ckpt, name, eps, device, fp32=True)

        def linear(*names: str):
            return Linear.load(ckpt, list(names), device)

        layers = []
        for i, kind in enumerate(kinds):
            p = f"{ROOT}layers.{i}."
            a, m = f"{p}self_attn.", f"{p}mlp."
            full = kind == "full_attention"
            shared = i >= first
            alt = k_eq_v and full
            if shared:
                qkv = linear(f"{a}q_proj")
            elif alt:
                qkv = linear(f"{a}q_proj", f"{a}k_proj")
            else:
                qkv = linear(f"{a}q_proj", f"{a}k_proj", f"{a}v_proj")
            mixer = Attention(
                qkv=qkv,
                o=linear(f"{a}o_proj"),
                heads=cfg["num_attention_heads"],
                kv_heads=cfg["num_global_key_value_heads"] if alt else cfg["num_key_value_heads"],
                head_dim=gd if full else hd,
                rope=rope_full if full else rope_slide,
                window=None if full else cfg["sliding_window"],
                q_norm=norm(f"{a}q_norm"),
                k_norm=None if shared else norm(f"{a}k_norm"),
                scale=1.0,
                k_eq_v=alt,
                backend=self.attention_backend,
                v_norm=eps,
                kv_source=source[kind] if shared else None,
                stash=not shared and source.get(kind) == i,
                write_shared=shared and last_reader[kind] == i,
                bidirectional=rule == "all" or (rule == "sliding" and not full),
            )
            ffn = SwiGLU(
                linear(f"{m}gate_proj", f"{m}up_proj"), linear(f"{m}down_proj"), "gelu_tanh"
            )
            extra = None
            if ple_dim:
                extra = PerLayerBlock(
                    linear(f"{p}per_layer_input_gate"),
                    linear(f"{p}per_layer_projection"),
                    norm(f"{p}post_per_layer_input_norm"),
                )
            layers.append(
                Decoder(
                    mixer,
                    ffn,
                    norm(f"{p}input_layernorm"),
                    norm(f"{p}pre_feedforward_layernorm"),
                    norm(f"{p}post_attention_layernorm"),
                    norm(f"{p}post_feedforward_layernorm"),
                    fuse_quant=self.fuse_quant,
                    extra=extra,
                    scalar=ckpt.get(f"{p}layer_scalar", device).to(work()),
                )
            )
        embedding = ckpt.get(f"{ROOT}embed_tokens.weight", device).to(work())
        head = None
        if self.head == "mxfp8":
            head = Linear("mxfp8", *mx.quant(embedding))
        elif self.head == "nvfp4":
            head = Linear.nvfp4_dynamic(embedding)
        prelude = None
        if ple_dim:
            prelude = PerLayerInputs(
                ckpt.get(f"{ROOT}embed_tokens_per_layer.weight", device).to(work()),
                linear(f"{ROOT}per_layer_model_projection"),
                norm(f"{ROOT}per_layer_projection_norm"),
                count,
                [ckpt.config[t] for t in PLACEHOLDERS if t in ckpt.config],
            )
        stack = Stack(
            embedding,
            layers,
            norm(f"{ROOT}norm"),
            head,
            embed_scale=math.sqrt(cfg["hidden_size"]),
            logit_softcap=cfg.get("final_logit_softcapping"),
            prelude=prelude,
            suppress=_suppressed(ckpt),
        )
        stack.image_runs = rule is not None
        if self.vision and "vision_config" in ckpt.config:
            build = vision.build_unified if unified else vision.build_tower
            stack.encoders = nn.ModuleDict({"image": build(ckpt, device)})
            stack.image_token = ckpt.config["image_token_id"]
            stack.processor = "tokenomos.models.gemma4.processor"
        return stack


def _suppressed(ckpt) -> list[int] | None:
    path = ckpt.path / "generation_config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8")).get("suppress_tokens") or None
