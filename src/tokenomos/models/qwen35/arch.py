"""Qwen3.5 / Qwen3.8 (``Qwen3_5ForConditionalGeneration``): the text stack.

Layers by ``layer_types`` (else every ``full_attention_interval``-th layer is attention):
``linear_attention`` -> ``GatedDeltaNet`` (``in_proj_qkv | in_proj_z | in_proj_b | in_proj_a``
fused, ``conv1d``, ``A_log``, ``dt_bias``, gated ``norm``, ``out_proj``); ``full_attention`` ->
``Attention`` with per-head (1 + w) QK norms, ``q_proj``'s per-head gate half moved behind
q / k / v (``fused_gate``), RoPE over the first ``partial_rotary_factor`` of a head with
interleaved multimodal sections. SwiGLU MLP, (1 + w) pre-norms, final ``norm``; the head tied
unless ``lm_head`` exists. The MTP layer is not loaded. With ``vision``, the image encoder
(``vision``) and the processor (``processor``, which also gives the prompt's RoPE positions).
"""

import torch
from torch import nn

from tokenomos.core.floats import work
from tokenomos.kernels.mxfp8 import ops as mx
from tokenomos.models.blocks.attention import Attention
from tokenomos.models.blocks.decoder import Decoder, Stack
from tokenomos.models.blocks.ffn import SwiGLU
from tokenomos.models.blocks.gdn import GatedDeltaNet
from tokenomos.models.blocks.linear import Linear
from tokenomos.models.blocks.norm import RMSNorm
from tokenomos.models.blocks.rope import Rope
from tokenomos.models.qwen35 import vision

ROOT = "model.language_model."


class Qwen35:
    names = ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")

    def __init__(
        self,
        attention_backend: str = "dense",
        fuse_quant: bool = True,
        head: str = "bf16",
        vision: bool = True,
        state_format: str = "fp32",
        gdn_prefill: str = "auto",
    ):
        """``head``: ``bf16`` (the checkpoint's, in its stored format), ``mxfp8`` or ``nvfp4`` (a
        bf16 head quantized at load; NVFP4 with a per-call activation scale).
        ``state_format``: the GDN recurrent state's (``fp32``, ``bf16``, ``fp8``, ``int8``).
        ``gdn_prefill``: the GDN chunked kernel (``fla``, ``flashinfer``, ``fused``, ``auto``)."""
        if head not in ("bf16", "mxfp8", "nvfp4"):
            raise ValueError(f"head format {head!r}")
        self.attention_backend = attention_backend
        self.fuse_quant = fuse_quant
        self.head = head
        self.vision = vision
        self.state_format = state_format
        self.gdn_prefill = gdn_prefill

    @staticmethod
    def layer_types(cfg: dict) -> list[str]:
        if "layer_types" in cfg:
            return cfg["layer_types"]
        every = cfg["full_attention_interval"]
        return [
            "full_attention" if (i + 1) % every == 0 else "linear_attention"
            for i in range(cfg["num_hidden_layers"])
        ]

    def build(self, ckpt, device) -> Stack:
        cfg = ckpt.text_config
        eps = cfg["rms_norm_eps"]
        hd = cfg["head_dim"]
        heads = cfg["num_attention_heads"]
        rp = cfg["rope_parameters"]
        if rp.get("rope_type", "default") != "default":
            raise ValueError(f"Qwen3.5: rope_type {rp['rope_type']!r} (default only)")
        rot = int(hd * rp.get("partial_rotary_factor", 1.0))
        sections = tuple(rp["mrope_section"]) if rp.get("mrope_section") else None
        if sections is not None and not rp.get("mrope_interleaved", False):
            raise ValueError("Qwen3.5: chunked multimodal RoPE sections (interleaved only)")
        rope = Rope(hd, rot // 2, rp["rope_theta"], device, dim=rot, sections=sections)
        # q_proj rows per head: [query D | gate D]
        per = torch.arange(heads * 2 * hd).view(heads, 2, hd)
        q_rows, gate_rows = per[:, 0].reshape(-1), per[:, 1].reshape(-1)

        def norm(name: str) -> RMSNorm:
            return RMSNorm.load(ckpt, name, eps, device, unit_offset=True)

        def f32(name: str) -> torch.Tensor:
            # the Hugging Face model holds these in bf16
            return ckpt.get(name, device).to(work()).float()

        layers = []
        for i, kind in enumerate(self.layer_types(cfg)):
            p = f"{ROOT}layers.{i}."
            if kind == "linear_attention":
                g = f"{p}linear_attn."
                conv = ckpt.get(f"{g}conv1d.weight", device).to(work())
                mixer = GatedDeltaNet(
                    Linear.load(
                        ckpt,
                        [f"{g}in_proj_qkv", f"{g}in_proj_z", f"{g}in_proj_b", f"{g}in_proj_a"],
                        device,
                    ),
                    Linear.load(ckpt, [f"{g}out_proj"], device),
                    conv.view(conv.shape[0], conv.shape[-1]).contiguous(),
                    f32(f"{g}A_log"),
                    f32(f"{g}dt_bias"),
                    ckpt.get(f"{g}norm.weight", device).to(work()),
                    eps,
                    cfg["linear_num_key_heads"],
                    cfg["linear_num_value_heads"],
                    cfg["linear_key_head_dim"],
                    cfg["linear_value_head_dim"],
                    self.state_format,
                    self.gdn_prefill,
                )
            elif kind == "full_attention":
                a = f"{p}self_attn."
                mixer = Attention(
                    qkv=Linear.load(
                        ckpt,
                        [
                            (f"{a}q_proj", q_rows),
                            f"{a}k_proj",
                            f"{a}v_proj",
                            (f"{a}q_proj", gate_rows),
                        ],
                        device,
                    ),
                    o=Linear.load(ckpt, [f"{a}o_proj"], device),
                    heads=heads,
                    kv_heads=cfg["num_key_value_heads"],
                    head_dim=hd,
                    rope=rope,
                    window=None,
                    q_norm=norm(f"{a}q_norm"),
                    k_norm=norm(f"{a}k_norm"),
                    backend=self.attention_backend,
                    fused_gate=True,
                )
            else:
                raise ValueError(f"Qwen3.5 layer {i}: type {kind!r}")
            m = f"{p}mlp."
            ffn = SwiGLU(
                Linear.load(ckpt, [f"{m}gate_proj", f"{m}up_proj"], device),
                Linear.load(ckpt, [f"{m}down_proj"], device),
            )
            layers.append(
                Decoder(
                    mixer,
                    ffn,
                    norm(f"{p}input_layernorm"),
                    norm(f"{p}post_attention_layernorm"),
                    fuse_quant=self.fuse_quant,
                )
            )
        embedding = ckpt.get(f"{ROOT}embed_tokens.weight", device).to(work())
        tied = not ckpt.has("lm_head.weight")
        stored = None if tied else Linear.load(ckpt, ["lm_head"], device)
        if self.head in ("mxfp8", "nvfp4"):
            if stored is not None and stored.fmt != "bf16":
                raise ValueError(f"head {self.head} over a {stored.fmt} checkpoint head")
            weight = embedding if tied else stored.weight
            if self.head == "mxfp8":
                head = Linear("mxfp8", *mx.quant(weight))
            else:
                head = Linear.nvfp4_dynamic(weight)
        else:
            head = stored
        stack = Stack(embedding, layers, norm(f"{ROOT}norm"), head)
        if self.vision and "vision_config" in ckpt.config:
            stack.encoders = nn.ModuleDict({"image": vision.build(ckpt, device)})
            stack.image_token = ckpt.config["image_token_id"]
            stack.processor = "tokenomos.models.qwen35.processor"
        return stack
