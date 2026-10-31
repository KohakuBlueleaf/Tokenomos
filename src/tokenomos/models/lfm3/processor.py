"""LFM3-VL's image preparation, as the checkpoint's processor does it (``imageprep`` with factor
``patch * merge``). ``tokens`` is the language-token count the image will take.
"""

import torch
from PIL import Image

from tokenomos.models import imageprep

PATCH, MERGE = 16, 2
MIN_PIXELS, MAX_PIXELS = 65536, 1024 * 1024
# the largest prepared image (uint8 RGB at MAX_PIXELS)
MAX_BYTES = MAX_PIXELS * 3
smart_resize = imageprep.smart_resize


def prepare(
    image: Image.Image, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> torch.Tensor:
    """PIL image -> resized uint8 [H, W, 3]."""
    return imageprep.prepare(image, PATCH * MERGE, min_pixels, max_pixels)


def tokens(prepared: torch.Tensor) -> int:
    return (prepared.shape[0] // PATCH) * (prepared.shape[1] // PATCH) // MERGE**2
