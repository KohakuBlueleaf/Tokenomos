"""Image preparation shared by the Qwen2-VL-style processors (LFM3-VL, Qwen3.5): RGB, both sides
rounded to multiples of ``factor`` with the area kept within ``[min_pixels, max_pixels]`` (aspect
preserved), torchvision bicubic resize with antialias on the uint8 image. Runs on the CPU (the
prep workers); the result is a uint8 [H, W, 3] tensor the encoder normalizes and patchifies on
the GPU.
"""

import math

import torch
from PIL import Image
from torchvision.transforms.v2 import functional as tvF


def runs(ids: list[int], token: int) -> list[tuple[int, int]]:
    """The maximal runs ``(first, end)`` of ``token`` in ``ids``."""
    out, i, n = [], 0, len(ids)
    while i < n:
        if ids[i] == token:
            j = i
            while j < n and ids[j] == token:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio over 200")
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def prepare(image: Image.Image, factor: int, min_pixels: int, max_pixels: int) -> torch.Tensor:
    """PIL image -> resized uint8 [H, W, 3]."""
    x = tvF.pil_to_tensor(image.convert("RGB"))
    h, w = smart_resize(x.shape[1], x.shape[2], factor, min_pixels, max_pixels)
    x = tvF.resize(x, [h, w], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    return x.permute(1, 2, 0).contiguous()
