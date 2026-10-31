"""LFM3-VL's image encoder: patchifier, ``embedding_norm``, conv / full-attention layers each with
a SwiGLU feed-forward (pre-norm residual, no final norm), then the projector (LayerNorm, each 2x2
merge block's four patches concatenated, linear, GELU, linear) to one language token per block.

Images are packed back to back, each image's patches in merge-block order (the processor's
layout). Conv layers run ``grid_conv`` on that layout; attention layers attend within an image
(bidirectional), with QK RMSNorm and axial 2D RoPE (row angles then column angles over half the
head, repeated). ``Batch`` holds the packing; ``pixels`` turns resized uint8 images into patches.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from tokenomos.core.floats import work
from tokenomos.kernels.attention import paged as kattn
from tokenomos.kernels.attention import qk
from tokenomos.kernels.conv.grid2d import grid_conv
from tokenomos.models.blocks.decoder import Decoder
from tokenomos.models.blocks.ffn import SwiGLU
from tokenomos.models.blocks.linear import Linear
from tokenomos.models.blocks.norm import RMSNorm
from tokenomos.models.contract import Step


@dataclass
class Batch:
    """A packed image batch: ``grids`` [(gh, gw)] patch grids, device tensors built by ``pack``."""

    grids: list[tuple[int, int]]
    cu_host: list[int]
    cu: torch.Tensor
    tok_img: torch.Tensor
    gh: torch.Tensor
    gw: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor


def direct(device):
    """A ``put`` that copies straight from pageable memory (the host waits for the stream)."""
    return lambda values, dtype: torch.as_tensor(values, dtype=dtype).to(device)


def pack(grids: list[tuple[int, int]], merge: int, put) -> Batch:
    """``put(values, dtype) -> device tensor`` moves the host-built index tensors (the engine
    passes its pinned staging)."""
    cu_host, img, rows, cols = [0], [], [], []
    for i, (gh, gw) in enumerate(grids):
        n = gh * gw
        cu_host.append(cu_host[-1] + n)
        img.append(torch.full((n,), i, dtype=torch.int32))
        hh, ww = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        blocks = (gh // merge, merge, gw // merge, merge)
        rows.append(hh.reshape(blocks).transpose(1, 2).flatten())
        cols.append(ww.reshape(blocks).transpose(1, 2).flatten())
    i32 = torch.int32
    return Batch(
        grids,
        cu_host,
        put(cu_host, i32),
        put(torch.cat(img), i32),
        put([g[0] for g in grids], i32),
        put([g[1] for g in grids], i32),
        put(torch.cat(rows), i32),
        put(torch.cat(cols), i32),
    )


def pixels(images: list[torch.Tensor], patch: int, merge: int, temporal: int) -> torch.Tensor:
    """Resized uint8 images [H, W, 3] (sides multiples of ``patch * merge``) on the device ->
    patches [sum(gh * gw), 3 * temporal * patch^2] float32, merge-block order, normalized as the
    processor's fused rescale + normalize does: ``(x - 127.5) / 127.5``."""
    out = []
    for im in images:
        h, w, _ = im.shape
        x = (im.permute(2, 0, 1).float() - 127.5) / 127.5
        gh, gw = h // patch, w // patch
        x = x.reshape(3, gh // merge, merge, patch, gw // merge, merge, patch)
        x = x.permute(1, 4, 2, 5, 0, 3, 6)
        x = x.unsqueeze(5).expand(-1, -1, -1, -1, -1, temporal, -1, -1)
        out.append(x.reshape(gh * gw, 3 * temporal * patch * patch))
    return torch.cat(out)


class GridConv(nn.Module):
    """``out_proj(c * conv2d(b * x))``, ``b, c, x = in_proj(h)``."""

    needs_bf16_input = False

    def __init__(self, in_proj: Linear, out_proj: Linear, weight: torch.Tensor, merge: int):
        super().__init__()
        self.in_proj, self.out_proj = in_proj, out_proj
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.merge = merge

    @property
    def input_linear(self) -> Linear:
        return self.in_proj

    def state_specs(self) -> dict:
        return {}

    def forward(self, x, xq, step: Step, index: int):
        bcx = self.in_proj.quantized(*xq) if xq is not None else self.in_proj(x)
        b, c, h = bcx.chunk(3, dim=-1)
        v: Batch = step.meta["vision"]
        return self.out_proj(
            grid_conv(b, c, h, self.weight, v.tok_img, v.cu, v.gh, v.gw, self.merge)
        )


class FullAttention(nn.Module):
    """Bidirectional GQA within each image, QK RMSNorm then axial RoPE (``step.meta`` tables)."""

    needs_bf16_input = False

    def __init__(self, qkv: Linear, out: Linear, heads, kv_heads, head_dim, q_norm, k_norm):
        super().__init__()
        self.qkv, self.out, self.q_norm, self.k_norm = qkv, out, q_norm, k_norm
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim

    @property
    def input_linear(self) -> Linear:
        return self.qkv

    def state_specs(self) -> dict:
        return {}

    def forward(self, x, xq, step: Step, index: int):
        n = x.shape[0] if x is not None else xq[0].shape[0]
        qkv = self.qkv.quantized(*xq) if xq is not None else self.qkv(x)
        hq, hk = self.heads * self.head_dim, self.kv_heads * self.head_dim
        q, k, v = qkv.split([hq, hk, hk], dim=-1)
        q = q.view(n, self.heads, self.head_dim)
        k = k.view(n, self.kv_heads, self.head_dim)
        v = v.view(n, self.kv_heads, self.head_dim)
        cos, sin = step.meta["axial"]
        q, k = qk.norm_rope(
            q, k, self.q_norm.weight, self.k_norm.weight, self.q_norm.eps, self.k_norm.eps, cos, sin
        )
        seq, q0, zeros = step.meta["tiles"](kattn.prefill_block(self.head_dim)[0])
        o = kattn.prefill(
            q, k, v, k, v, step.cu, zeros, zeros, None, seq, q0, self.head_dim**-0.5, causal=False
        )
        return self.out(o.view(n, hq))


class Tower(nn.Module):
    def __init__(self, patchifier: Linear, norm: RMSNorm, layers: list[Decoder], inv_freq, merge):
        super().__init__()
        self.patchifier, self.norm = patchifier, norm
        self.layers = nn.ModuleList(layers)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.merge = merge

    def forward(self, patches: torch.Tensor, batch: Batch, put) -> torch.Tensor:
        dev = patches.device
        freqs = torch.stack((batch.rows, batch.cols), -1)[..., None].float() * self.inv_freq.float()
        hw = torch.cat((freqs[:, 0], freqs[:, 1]), dim=-1)
        angles = torch.cat((hw, hw), dim=-1)
        tiles: dict[int, tuple] = {}
        zeros = torch.zeros(len(batch.grids), dtype=torch.int32, device=dev)

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
            meta={"vision": batch, "axial": (angles.cos(), angles.sin()), "tiles": tile},
        )
        h = self.norm(self.patchifier(patches.to(self.norm.weight.dtype)))
        residual, pending = None, None
        for i, layer in enumerate(self.layers):
            h, residual, pending = layer(h, residual, pending, step, i)
        return residual + h


class Projector(nn.Module):
    def __init__(self, norm_w, norm_b, eps, linear_1: Linear, linear_2: Linear, merge: int):
        super().__init__()
        self.norm_w = nn.Parameter(norm_w, requires_grad=False)
        self.norm_b = nn.Parameter(norm_b, requires_grad=False)
        self.eps, self.merge = eps, merge
        self.linear_1, self.linear_2 = linear_1, linear_2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.layer_norm(x, (x.shape[-1],), self.norm_w, self.norm_b, self.eps)
        h = h.reshape(-1, x.shape[-1] * self.merge**2)
        return self.linear_2(F.gelu(self.linear_1(h)))


class Encoder(nn.Module):
    """Images -> language-model input rows (one per merge block), packed in image order."""

    def __init__(self, tower: Tower, projector: Projector, patch: int, temporal: int):
        super().__init__()
        self.tower, self.projector = tower, projector
        self.patch, self.temporal, self.merge = patch, temporal, tower.merge

    def tokens(self, grid: tuple[int, int]) -> int:
        return grid[0] * grid[1] // self.merge**2

    def forward(self, images: list[torch.Tensor], put=None) -> torch.Tensor:
        """``images``: resized uint8 [H, W, 3] on the device. ``put``: see ``pack`` (default
        ``direct``)."""
        put = put or direct(images[0].device)
        grids = [(im.shape[0] // self.patch, im.shape[1] // self.patch) for im in images]
        batch = pack(grids, self.merge, put)
        patches = pixels(images, self.patch, self.merge, self.temporal)
        return self.projector(self.tower(patches, batch, put))


def build(ckpt, device, fuse_quant: bool = True) -> Encoder:
    cfg = ckpt.config["vision_config"]
    root = "model.vision_tower."
    eps = cfg["norm_eps"]
    merge = cfg["spatial_merge_size"]
    heads, kv, dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"]
    head_dim = dim // heads

    def norm(name):
        return RMSNorm.load(ckpt, name, eps, device)

    layers = []
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"{root}layers.{i}."
        if kind == "conv":
            w = ckpt.get(f"{p}conv.spatial_conv.weight", device).to(work())
            mixer = GridConv(
                Linear.load(ckpt, [f"{p}conv.in_proj"], device),
                Linear.load(ckpt, [f"{p}conv.out_proj"], device),
                w.reshape(dim, w.shape[-2], w.shape[-1]).contiguous(),
                merge,
            )
        else:
            a = f"{p}self_attn."
            mixer = FullAttention(
                Linear.load(ckpt, [f"{a}q_proj", f"{a}k_proj", f"{a}v_proj"], device),
                Linear.load(ckpt, [f"{a}out_proj"], device),
                heads,
                kv,
                head_dim,
                norm(f"{a}q_layernorm"),
                norm(f"{a}k_layernorm"),
            )
        f = f"{p}feed_forward."
        ffn = SwiGLU(
            Linear.load(ckpt, [f"{f}w1", f"{f}w3"], device),
            Linear.load(ckpt, [f"{f}w2"], device),
        )
        layers.append(
            Decoder(
                mixer, ffn, norm(f"{p}operator_norm"), norm(f"{p}ffn_norm"), fuse_quant=fuse_quant
            )
        )
    theta = cfg["rope_parameters"]["rope_theta"]
    if cfg["rope_parameters"]["rope_type"] != "axial":
        raise ValueError("LFM3-VL vision RoPE must be axial")
    half = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, 2, dtype=torch.float) / half))
    tower = Tower(
        Linear.load(ckpt, [f"{root}patchifier"], device),
        norm(f"{root}embedding_norm"),
        layers,
        inv.to(device),
        merge,
    )
    pr = "model.multi_modal_projector."
    projector = Projector(
        ckpt.get(f"{pr}layer_norm.weight", device).to(work()),
        ckpt.get(f"{pr}layer_norm.bias", device).to(work()),
        ckpt.config.get("projector_layer_norm_eps", 1e-6),
        Linear.load(ckpt, [f"{pr}linear_1"], device),
        Linear.load(ckpt, [f"{pr}linear_2"], device),
        merge,
    )
    return Encoder(
        projector=projector,
        tower=tower,
        patch=cfg["patch_size"],
        temporal=cfg["temporal_patch_size"],
    )
