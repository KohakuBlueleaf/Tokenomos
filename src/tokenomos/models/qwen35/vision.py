"""Qwen3.5's image encoder (``model.visual``): patch embedding (the Conv3d over
``(C, temporal, patch, patch)`` as a linear), the learned 48x48 position table resampled
bilinearly (align corners) to each image's patch grid, ``depth`` pre-LayerNorm blocks
(bidirectional attention within an image with axial 2D RoPE over the whole head: row angles then
column angles, repeated; GELU-tanh MLP), then the merger (LayerNorm per patch, each 2x2 merge
block's four patches concatenated, linear, GELU, linear) to one language token per block.

Images are packed back to back, each image's patches in merge-block order (``lfm3.vision.pack``
/ ``pixels``, the same layout and normalization).
"""

import torch
import torch.nn.functional as F
from torch import nn

from tokenomos.core.floats import work
from tokenomos.kernels.attention import paged as kattn
from tokenomos.kernels.attention import qk
from tokenomos.models.blocks.linear import Linear
from tokenomos.models.lfm3.vision import Batch, direct, pack, pixels


class LayerNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class Block(nn.Module):
    def __init__(self, norm1, norm2, qkv: Linear, proj: Linear, fc1: Linear, fc2: Linear, heads):
        super().__init__()
        self.norm1, self.norm2 = norm1, norm2
        self.qkv, self.proj, self.fc1, self.fc2 = qkv, proj, fc1, fc2
        self.heads = heads

    def forward(self, h: torch.Tensor, cos, sin, cu, tile) -> torch.Tensor:
        n, dim = h.shape
        hd = dim // self.heads
        qkv = self.qkv(self.norm1(h)).view(n, 3, self.heads, hd)
        q, k = qk.norm_rope(qkv[:, 0], qkv[:, 1], None, None, 0.0, 0.0, cos, sin)
        v = qkv[:, 2]
        seq, q0, zeros = tile(kattn.prefill_block(hd)[0])
        o = kattn.prefill(q, k, v, k, v, cu, zeros, zeros, None, seq, q0, hd**-0.5, causal=False)
        h = h + self.proj(o.view(n, dim))
        return h + self.fc2(F.gelu(self.fc1(self.norm2(h)), approximate="tanh"))


def interpolate(table: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, gh, gw):
    """The ``[side * side, D]`` position table resampled at patch ``(rows, cols)`` of grids
    ``(gh, gw)`` (per patch): bilinear, align corners, float32 weights; -> float32 [N, D]."""
    side = int(table.shape[0] ** 0.5)
    acc = None
    taps = []
    for idx, size in ((rows, gh), (cols, gw)):
        src = idx.float() * (side - 1) / torch.clamp(size - 1, min=1).float()
        lo = torch.floor(src)
        frac = src - lo
        lo = lo.long()
        taps.append(((lo.clamp(0, side - 1), 1 - frac), ((lo + 1).clamp(0, side - 1), frac)))
    for hi, hw in taps[0]:
        for wi, ww in taps[1]:
            term = table[hi * side + wi].float() * (hw * ww)[:, None]
            acc = term if acc is None else acc + term
    return acc


class Tower(nn.Module):
    def __init__(self, patch: Linear, pos: torch.Tensor, blocks: list[Block], inv_freq, merge):
        super().__init__()
        self.patch = patch
        self.pos = nn.Parameter(pos, requires_grad=False)
        self.blocks = nn.ModuleList(blocks)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.merge = merge

    def forward(self, patches: torch.Tensor, batch: Batch, put) -> torch.Tensor:
        dev = patches.device
        img = batch.tok_img.long()
        gh, gw = batch.gh[img], batch.gw[img]
        h = self.patch(patches.to(self.pos.dtype))
        h = h + interpolate(self.pos, batch.rows.long(), batch.cols.long(), gh, gw).to(h.dtype)
        freqs = torch.stack((batch.rows, batch.cols), -1)[..., None].float() * self.inv_freq
        hw = torch.cat((freqs[:, 0], freqs[:, 1]), dim=-1)
        angles = torch.cat((hw, hw), dim=-1)
        cos, sin = angles.cos(), angles.sin()
        tiles: dict[int, tuple] = {}
        zeros = torch.zeros(len(batch.grids), dtype=torch.int32, device=dev)

        def tile(bm: int):
            if bm not in tiles:
                s, q0 = kattn.tiles(batch.cu_host, bm)
                tiles[bm] = (put(s, torch.int32), put(q0, torch.int32), zeros)
            return tiles[bm]

        for block in self.blocks:
            h = block(h, cos, sin, batch.cu, tile)
        return h


class Merger(nn.Module):
    def __init__(self, norm: LayerNorm, fc1: Linear, fc2: Linear, merge: int):
        super().__init__()
        self.norm, self.fc1, self.fc2, self.merge = norm, fc1, fc2, merge

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).reshape(-1, x.shape[-1] * self.merge**2)
        return self.fc2(F.gelu(self.fc1(h)))


class Encoder(nn.Module):
    """Images -> language-model input rows (one per merge block), packed in image order."""

    def __init__(self, tower: Tower, merger: Merger, patch: int, temporal: int):
        super().__init__()
        self.tower, self.merger = tower, merger
        self.patch, self.temporal, self.merge = patch, temporal, tower.merge

    def tokens(self, grid: tuple[int, int]) -> int:
        return grid[0] * grid[1] // self.merge**2

    def forward(self, images: list[torch.Tensor], put=None) -> torch.Tensor:
        """``images``: resized uint8 [H, W, 3] on the device. ``put``: see ``lfm3.vision.pack``
        (default ``direct``)."""
        put = put or direct(images[0].device)
        grids = [(im.shape[0] // self.patch, im.shape[1] // self.patch) for im in images]
        batch = pack(grids, self.merge, put)
        patches = pixels(images, self.patch, self.merge, self.temporal)
        return self.merger(self.tower(patches, batch, put))


def build(ckpt, device) -> Encoder:
    cfg = ckpt.config["vision_config"]
    root = "model.visual."
    dim, heads, merge = cfg["hidden_size"], cfg["num_heads"], cfg["spatial_merge_size"]
    eps = 1e-6

    def ln(name):
        return LayerNorm(
            ckpt.get(f"{name}.weight", device).to(work()),
            ckpt.get(f"{name}.bias", device).to(work()),
            eps,
        )

    def linear(name):
        return Linear.load(ckpt, [name], device)

    blocks = []
    for i in range(cfg["depth"]):
        p = f"{root}blocks.{i}."
        blocks.append(
            Block(
                ln(f"{p}norm1"),
                ln(f"{p}norm2"),
                linear(f"{p}attn.qkv"),
                linear(f"{p}attn.proj"),
                linear(f"{p}mlp.linear_fc1"),
                linear(f"{p}mlp.linear_fc2"),
                heads,
            )
        )
    w = ckpt.get(f"{root}patch_embed.proj.weight", device).to(work())
    patch = Linear(
        "bf16",
        w.reshape(w.shape[0], -1).contiguous(),
        bias=ckpt.get(f"{root}patch_embed.proj.bias", device).to(work()),
    )
    rp = cfg.get("rope_parameters", {})
    theta = rp.get("rope_theta", 10000.0)
    half = dim // heads // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, 2, dtype=torch.float) / half))
    tower = Tower(
        patch,
        ckpt.get(f"{root}pos_embed.weight", device).to(work()),
        blocks,
        inv.to(device),
        merge,
    )
    m = f"{root}merger."
    merger = Merger(ln(f"{m}norm"), linear(f"{m}linear_fc1"), linear(f"{m}linear_fc2"), merge)
    return Encoder(tower, merger, cfg["patch_size"], cfg["temporal_patch_size"])
