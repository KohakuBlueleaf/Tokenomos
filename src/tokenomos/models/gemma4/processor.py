"""Gemma 4's image preparation: RGB, resized (down or up) to the largest multiples of ``UNIT`` =
48 px, aspect preserved, within a budget of ``soft_tokens`` ``UNIT`` blocks, torchvision bicubic
with antialias. One language token per block, placeholder expanded to ``BOI + image * n + EOI``.
"""

import math

import torch
from PIL import Image
from torchvision.transforms.v2 import functional as tvF

UNIT = 48
BOI, IMAGE, EOI = 255999, 258880, 258882
# the checkpoints' processor default (allowed: 70, 140, 280, 560, 1120)
SOFT_TOKENS = 280
MAX_BYTES = SOFT_TOKENS * UNIT * UNIT * 3


def target(height: int, width: int, soft_tokens: int = SOFT_TOKENS) -> tuple[int, int]:
    """The resized ``(height, width)``."""
    factor = math.sqrt(soft_tokens * UNIT * UNIT / (height * width))
    h = math.floor(height * factor / UNIT) * UNIT
    w = math.floor(width * factor / UNIT) * UNIT
    if h == 0:
        return UNIT, min(width // height * UNIT, soft_tokens * UNIT)
    if w == 0:
        return min(height // width * UNIT, soft_tokens * UNIT), UNIT
    return h, w


def prepare(image: Image.Image, soft_tokens: int = SOFT_TOKENS) -> torch.Tensor:
    """PIL image -> resized uint8 [H, W, 3]."""
    x = tvF.pil_to_tensor(image.convert("RGB"))
    h, w = target(x.shape[1], x.shape[2], soft_tokens)
    if (h, w) != (x.shape[1], x.shape[2]):
        x = tvF.resize(x, [h, w], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    return x.permute(1, 2, 0).contiguous()


def tokens(prepared: torch.Tensor) -> int:
    return prepared.shape[0] // UNIT * (prepared.shape[1] // UNIT)


def wrap(n: int) -> list[int]:
    """A placeholder's expansion for an image of ``n`` tokens."""
    return [BOI] + [IMAGE] * n + [EOI]
