"""LFM3 (Liquid): the text stack of LFM3-VL / LFM3-Omni / LFM3 text checkpoints.

Layers by ``layer_types``: ``conv`` -> ``ShortConv`` (width ``conv_L_cache``);
``full_attention`` / ``sliding_attention`` -> ``Attention`` with QK RMSNorm, a sigmoid output
gate per head and zero-tail partial RoPE per layer type (``sliding_window`` on the sliding ones).
Every layer has a SwiGLU feed-forward (``w1`` gate, ``w3`` up, ``w2`` down) and, with
``use_sandwich_norm``, post-norms after the mixer and the feed-forward. Biases (``conv_bias``)
load when present. Final ``embedding_norm``; head tied to the embedding unless
``tie_word_embeddings`` is false (then ``lm_head``), optionally quantized to MXFP8.
"""

from torch import nn

from tokenomos.core.floats import work
from tokenomos.kernels.mxfp8 import ops as mx
from tokenomos.models.blocks.attention import Attention
from tokenomos.models.blocks.decoder import Decoder, Stack
from tokenomos.models.blocks.ffn import SwiGLU
from tokenomos.models.blocks.linear import Linear
from tokenomos.models.blocks.norm import RMSNorm
from tokenomos.models.blocks.rope import Rope
from tokenomos.models.blocks.shortconv import ShortConv
from tokenomos.models.lfm3 import vision


class Lfm3:
    names = (
        "Lfm3VlForConditionalGeneration",
        "Lfm3OmniForConditionalGeneration",
        "Lfm3ForCausalLM",
    )

    def __init__(
        self,
        attention_backend: str = "dense",
        fuse_quant: bool = True,
        head: str = "bf16",
        vision: bool = True,
    ):
        """``head``: ``bf16`` (the checkpoint's), ``mxfp8`` or ``nvfp4`` (quantized at load).
        ``vision``: build the image encoder when the checkpoint has one."""
        if head not in ("bf16", "mxfp8", "nvfp4"):
            raise ValueError(f"head format {head!r}")
        self.attention_backend = attention_backend
        self.fuse_quant = fuse_quant
        self.head = head
        self.vision = vision

    @staticmethod
    def prefix(ckpt) -> str:
        return (
            "model.language_model."
            if ckpt.has("model.language_model.embed_tokens.weight")
            else "model."
        )

    def build(self, ckpt, device) -> Stack:
        cfg = ckpt.text_config
        root = self.prefix(ckpt)
        eps = cfg["norm_eps"]
        head_dim = cfg["head_dim"]
        ropes = {}
        for kind, params in cfg["rope_parameters"].items():
            if params.get("rope_type", "default") != "default":
                raise ValueError(f"LFM3 {kind}: rope_type {params['rope_type']!r} (default only)")
            angles = int(params.get("partial_rotary_factor", 1.0) * head_dim) // 2
            ropes[kind] = Rope(head_dim, angles, params["rope_theta"], device)

        def norm(name: str) -> RMSNorm:
            return RMSNorm.load(ckpt, name, eps, device)

        layers = []
        for i, kind in enumerate(cfg["layer_types"]):
            p = f"{root}layers.{i}."
            if kind == "conv":
                conv = ckpt.get(f"{p}conv.conv.weight", device).to(work())
                bias = None
                if cfg.get("conv_bias", False):
                    bias = ckpt.get(f"{p}conv.conv.bias", device).to(work())
                mixer = ShortConv(
                    Linear.load(ckpt, [f"{p}conv.in_proj"], device),
                    Linear.load(ckpt, [f"{p}conv.out_proj"], device),
                    conv.view(conv.shape[0], conv.shape[-1]).contiguous(),
                    bias,
                )
            else:
                a = f"{p}self_attn."
                mixer = Attention(
                    qkv=Linear.load(ckpt, [f"{a}q_proj", f"{a}k_proj", f"{a}v_proj"], device),
                    o=Linear.load(ckpt, [f"{a}o_proj"], device),
                    heads=cfg["num_attention_heads"],
                    kv_heads=cfg["num_key_value_heads"],
                    head_dim=head_dim,
                    rope=ropes[kind],
                    window=cfg["sliding_window"] if kind == "sliding_attention" else None,
                    q_norm=norm(f"{a}q_norm"),
                    k_norm=norm(f"{a}k_norm"),
                    gate=Linear.load(ckpt, [f"{a}gate_proj"], device),
                    backend=self.attention_backend,
                )
            f = f"{p}feed_forward."
            ffn = SwiGLU(
                Linear.load(ckpt, [f"{f}w1", f"{f}w3"], device),
                Linear.load(ckpt, [f"{f}w2"], device),
            )
            sandwich = cfg.get("use_sandwich_norm", False)
            layers.append(
                Decoder(
                    mixer,
                    ffn,
                    norm(f"{p}operator_norm"),
                    norm(f"{p}ffn_norm"),
                    norm(f"{p}post_operator_norm") if sandwich else None,
                    norm(f"{p}post_ffn_norm") if sandwich else None,
                    fuse_quant=self.fuse_quant,
                )
            )
        embedding = ckpt.get(f"{root}embed_tokens.weight", device).to(work())
        weight = embedding if ckpt.tied else ckpt.get("lm_head.weight", device).to(work())
        if self.head == "mxfp8":
            head = Linear("mxfp8", *mx.quant(weight))
        elif self.head == "nvfp4":
            head = Linear.nvfp4_dynamic(weight)
        else:
            head = None if ckpt.tied else Linear("bf16", weight)
        stack = Stack(embedding, layers, norm(f"{root}embedding_norm"), head)
        if self.vision and "vision_config" in ckpt.config:
            stack.encoders = nn.ModuleDict({"image": vision.build(ckpt, device, self.fuse_quant)})
            stack.image_token = ckpt.config["image_token_id"]
            stack.processor = "tokenomos.models.lfm3.processor"
        return stack
