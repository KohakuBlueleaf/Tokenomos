"""Qwen3.5's image preparation (Qwen2-VL fast processor: ``imageprep`` with factor
``patch * merge``) and its multimodal RoPE positions.

``MAX_PIXELS`` caps the serving size at 1024² px (the checkpoint allows 4096²): the prepared
image, its ring slot and its language tokens scale with it.

``rope``: text tokens take consecutive positions on all three axes; an image of merged grid
``(gh, gw)`` starting at position ``p`` takes ``(p, p + row, p + col)`` per token in raster
order, and the text after it starts at ``p + max(gh, gw)``.
"""

import torch
from PIL import Image

from tokenomos.models import imageprep

PATCH, MERGE = 16, 2
MIN_PIXELS, MAX_PIXELS = 65536, 1024 * 1024
# the largest prepared image (uint8 RGB at MAX_PIXELS)
MAX_BYTES = MAX_PIXELS * 3


def prepare(
    image: Image.Image, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> torch.Tensor:
    """PIL image -> resized uint8 [H, W, 3]."""
    return imageprep.prepare(image, PATCH * MERGE, min_pixels, max_pixels)


def grid(prepared: torch.Tensor) -> tuple[int, int]:
    """The merged (language-token) grid of a prepared image."""
    unit = PATCH * MERGE
    return prepared.shape[0] // unit, prepared.shape[1] // unit


def tokens(prepared: torch.Tensor) -> int:
    gh, gw = grid(prepared)
    return gh * gw


def rope(ids: list[int], images: list[torch.Tensor], image_token: int) -> torch.Tensor:
    """[3, len(ids)] int32 positions (T, H, W) of a prompt whose placeholders are expanded."""
    out = torch.empty(3, len(ids), dtype=torch.int32)
    grids = iter([grid(im) for im in images])
    i, pos, n = 0, 0, len(ids)
    while i < n:
        j = i
        while j < n and ids[j] != image_token:
            j += 1
        out[:, i:j] = torch.arange(pos, pos + j - i, dtype=torch.int32)
        pos += j - i
        if j == n:
            break
        gh, gw = next(grids)
        rows = torch.arange(gh, dtype=torch.int32).repeat_interleave(gw)
        cols = torch.arange(gw, dtype=torch.int32).repeat(gh)
        out[0, j : j + gh * gw] = pos
        out[1, j : j + gh * gw] = pos + rows
        out[2, j : j + gh * gw] = pos + cols
        pos += max(gh, gw)
        i = j + gh * gw
    return out
