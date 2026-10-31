"""Gemma 4's image encoders, packed like every encoder here: images back to back, a resized uint8
image's patches in merge-block order (``lfm3.vision.pack``), one language token per 48 px block.

``Tower`` (E4B, 31B): 16 px patches mapped to [-1, 1], a linear, a learned table per axis; sandwich
blocks of bidirectional attention within an image (QK RMSNorm, weightless V norm, score scale 1,
2D RoPE: the first half of a head rotates by column, the second by row) and a GeGLU MLP, each
linear optionally clamped to bounds in the checkpoint (E4B); then each 3x3 block averaged, times
``sqrt(hidden)``, optionally standardized (31B), a weightless RMSNorm, a linear to the text width.

``Unified`` (12B): no tower: each 48 px block's pixels in [0, 1] through LayerNorm, a linear with
bias, LayerNorm, the per-axis table added, LayerNorm, a weightless RMSNorm, a linear.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from tokenomos.core.floats import work
from tokenomos.kernels.attention import paged as kattn
from tokenomos.kernels.attention import qk
from tokenomos.models.blocks.decoder import Decoder, add_norm
from tokenomos.models.blocks.linear import Linear
from tokenomos.models.blocks.norm import RMSNorm
from tokenomos.models.contract import Step
from tokenomos.models.lfm3.vision import Batch, direct, pack
from tokenomos.models.qwen35.vision import LayerNorm

PATCH, MERGE = 16, 3
EPS = 1e-6


def patches(images: list[torch.Tensor], patch: int, merge: int) -> torch.Tensor:
    """Resized uint8 images [H, W, 3] on the device -> float32 [N, patch^2 * 3] in [0, 1], merge
    blocks in raster order, a patch's values in (row, column, channel) order."""
    out = []
    for im in images:
        h, w, _ = im.shape
        gh, gw = h // patch // merge, w // patch // merge
        x = (im.float() / 255.0).reshape(gh, merge, patch, gw, merge, patch, 3)
        out.append(x.permute(0, 3, 1, 4, 2, 5, 6).reshape(gh * gw * merge**2, patch * patch * 3))
    return torch.cat(out)


def unit_rms(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm without a weight, rounded once."""
    y = x.float()
    return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype)


class Clipped(nn.Module):
    """A 16-bit linear with its input and output clamped; ``bounds`` (input min, max, output min,
    max) or None."""

    def __init__(self, linear: Linear, bounds: tuple | None):
        super().__init__()
        self.linear, self.bounds = linear, bounds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bounds is None:
            return self.linear(x)
        a, b, c, d = self.bounds
        return self.linear(x.clamp(a, b)).clamp(c, d)


class TowerAttention(nn.Module):
    needs_bf16_input = False
    input_linear = None

    def __init__(self, q, k, v, o, heads, head_dim, q_norm: RMSNorm, k_norm: RMSNorm):
        super().__init__()
        self.q, self.k, self.v, self.o = q, k, v, o
        self.heads, self.head_dim = heads, head_dim
        self.q_norm, self.k_norm = q_norm, k_norm

    def forward(self, x, xq, step: Step, index: int):
        n, shape = x.shape[0], (x.shape[0], self.heads, self.head_dim)
        cos, sin = step.meta["axial"]
        q, k, v = qk.norm_rope(
            self.q(x).view(shape),
            self.k(x).view(shape),
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_norm.eps,
            self.k_norm.eps,
            cos,
            sin,
            False,
            v=self.v(x).view(shape),
            eps_v=EPS,
            chunks=2,
        )
        seq, q0, zeros = step.meta["tiles"](kattn.prefill_block(self.head_dim)[0])
        o = kattn.prefill(q, k, v, k, v, step.cu, zeros, zeros, None, seq, q0, 1.0, causal=False)
        return self.o(o.view(n, self.heads * self.head_dim))


class TowerMLP(nn.Module):
    input_linear = None

    def __init__(self, gate, up, down):
        super().__init__()
        self.gate, self.up, self.down = gate, up, down

    def forward(self, x, xq) -> torch.Tensor:
        return self.down(F.gelu(self.gate(x), approximate="tanh") * self.up(x))


class Tower(nn.Module):
    """Images -> language rows: patch embedding, blocks, pooling, projection."""

    def __init__(self, proj, table, layers, inv_freq, embed, std: tuple | None):
        super().__init__()
        self.proj, self.embed = proj, embed
        self.table = nn.Parameter(table, requires_grad=False)
        self.layers = nn.ModuleList(layers)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.std = std
        self.patch, self.merge = PATCH, MERGE

    def tokens(self, grid: tuple[int, int]) -> int:
        return grid[0] * grid[1] // MERGE**2

    def forward(self, images: list[torch.Tensor], put=None) -> torch.Tensor:
        """``images``: resized uint8 [H, W, 3] on the device. ``put``: see ``lfm3.vision.pack``."""
        put = put or direct(images[0].device)
        grids = [(im.shape[0] // PATCH, im.shape[1] // PATCH) for im in images]
        batch: Batch = pack(grids, MERGE, put)
        dt = self.table.dtype
        x = (2.0 * (patches(images, PATCH, MERGE) - 0.5)).to(dt)
        h = self.proj(x) + self.table[0, batch.cols.long()] + self.table[1, batch.rows.long()]
        # the column angles twice, then the row angles twice: float32 tables rounded to 16 bits
        fx = batch.cols.float()[:, None] * self.inv_freq
        fy = batch.rows.float()[:, None] * self.inv_freq
        angles = torch.cat((fx, fx, fy, fy), dim=-1)
        cos, sin = angles.cos().to(dt).float(), angles.sin().to(dt).float()
        tiles: dict[int, tuple] = {}
        zeros = torch.zeros(len(grids), dtype=torch.int32, device=h.device)

        def tile(bm: int):
            if bm not in tiles:
                s, q0 = kattn.tiles(batch.cu_host, bm)
                tiles[bm] = (put(s, torch.int32), put(q0, torch.int32), zeros)
            return tiles[bm]

        step = Step(
            positions=None,
            cu=batch.cu,
            tok_seq=batch.tok_img,
            n_tok=None,
            slot=zeros,
            init=zeros,
            meta={"axial": (cos, sin), "tiles": tile},
        )
        residual, pending = None, None
        for i, layer in enumerate(self.layers):
            h, residual, pending = layer(h, residual, pending, step, i)
        _, _, h = add_norm(None, h, residual, None, False, pending[0])
        pooled = h.float().view(-1, MERGE**2, h.shape[-1]).mean(1).to(dt).float()
        pooled = pooled * math.sqrt(h.shape[-1])
        if self.std is not None:
            pooled = (pooled - self.std[0].float()) * self.std[1].float()
        return self.embed(unit_rms(pooled.to(dt)))


class Unified(nn.Module):
    """12B's patch embedder: images -> language rows."""

    def __init__(self, ln1, dense, ln2, table, ln3, embed):
        super().__init__()
        self.ln1, self.dense, self.ln2, self.ln3, self.embed = ln1, dense, ln2, ln3, embed
        self.table = nn.Parameter(table, requires_grad=False)
        self.patch, self.merge = PATCH * MERGE, 1

    def tokens(self, grid: tuple[int, int]) -> int:
        return grid[0] * grid[1]

    def forward(self, images: list[torch.Tensor], put=None) -> torch.Tensor:
        put = put or direct(images[0].device)
        grids = [(im.shape[0] // self.patch, im.shape[1] // self.patch) for im in images]
        batch = pack(grids, 1, put)
        h = self.ln2(self.dense(self.ln1(patches(images, self.patch, 1).to(self.table.dtype))))
        h = h + self.table[batch.cols.long(), 0] + self.table[batch.rows.long(), 1]
        return self.embed(unit_rms(self.ln3(h)))


def build_tower(ckpt, device) -> Tower:
    cfg = ckpt.config["vision_config"]
    root = "model.vision_tower."
    heads, head_dim = cfg["num_attention_heads"], cfg["head_dim"]
    clipped = cfg.get("use_clipped_linears", False)

    def clip(name: str) -> Clipped:
        bounds = None
        if clipped:
            bounds = tuple(
                ckpt.get(f"{name}.{b}", device).to(work()).item()
                for b in ("input_min", "input_max", "output_min", "output_max")
            )
        return Clipped(Linear.load(ckpt, [f"{name}.linear"], device), bounds)

    def norm(name: str) -> RMSNorm:
        return RMSNorm.load(ckpt, name, EPS, device, fp32=True)

    layers = []
    for i in range(cfg["num_hidden_layers"]):
        p = f"{root}encoder.layers.{i}."
        a, m = f"{p}self_attn.", f"{p}mlp."
        layers.append(
            Decoder(
                TowerAttention(
                    clip(f"{a}q_proj"),
                    clip(f"{a}k_proj"),
                    clip(f"{a}v_proj"),
                    clip(f"{a}o_proj"),
                    heads,
                    head_dim,
                    norm(f"{a}q_norm"),
                    norm(f"{a}k_norm"),
                ),
                TowerMLP(clip(f"{m}gate_proj"), clip(f"{m}up_proj"), clip(f"{m}down_proj")),
                norm(f"{p}input_layernorm"),
                norm(f"{p}pre_feedforward_layernorm"),
                norm(f"{p}post_attention_layernorm"),
                norm(f"{p}post_feedforward_layernorm"),
            )
        )
    spatial = head_dim // 2
    theta = cfg["rope_parameters"]["rope_theta"]
    inv = 1.0 / (theta ** (torch.arange(0, spatial, 2, dtype=torch.float32) / spatial))
    std = None
    if cfg.get("standardize", False):
        std = tuple(
            nn.Parameter(ckpt.get(f"{root}{n}", device).to(work()), requires_grad=False)
            for n in ("std_bias", "std_scale")
        )
    return Tower(
        Linear.load(ckpt, [f"{root}patch_embedder.input_proj"], device),
        ckpt.get(f"{root}patch_embedder.position_embedding_table", device).to(work()),
        layers,
        inv.to(device),
        Linear.load(ckpt, ["model.embed_vision.embedding_projection"], device),
        std,
    )


def build_unified(ckpt, device) -> Unified:
    root = "model.vision_embedder."

    def ln(name: str) -> LayerNorm:
        return LayerNorm(
            ckpt.get(f"{root}{name}.weight", device).to(work()),
            ckpt.get(f"{root}{name}.bias", device).to(work()),
            1e-5,
        )

    return Unified(
        ln("patch_ln1"),
        Linear.load(ckpt, [f"{root}patch_dense"], device),
        ln("patch_ln2"),
        ckpt.get(f"{root}pos_embedding", device).to(work()),
        ln("pos_norm"),
        Linear.load(ckpt, ["model.embed_vision.embedding_projection"], device),
    )
