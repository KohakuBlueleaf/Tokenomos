"""RMSNorm rounded as the Hugging Face models round it: statistics in float32 on the 16-bit input,
then either the normed value rounded to the input's dtype times the 16-bit weight
(``round_normed``, LFM3) or the float32 normed value times a float32 weight, rounded once
(``fp32``: Gemma 4; Qwen3.5, whose stored weight is the offset from one: ``unit_offset`` loads
``1 + w`` in float32). On the GPU it runs the fused norm kernel (rows of the last dimension).
"""

import torch
from torch import nn

from tokenomos.core.floats import DTYPES, work
from tokenomos.kernels.norm import fused


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float, round_normed: bool = True):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.eps = eps
        self.round_normed = round_normed

    @classmethod
    def load(
        cls, ckpt, name: str, eps: float, device, unit_offset: bool = False, fp32: bool = False
    ) -> "RMSNorm":
        w = ckpt.get(f"{name}.weight", device).to(work())
        if unit_offset:
            return cls(1.0 + w.float(), eps, round_normed=False)
        if fp32:
            return cls(w.float(), eps, round_normed=False)
        return cls(w, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype in DTYPES.values():
            k = x.shape[-1]
            out, _, _ = fused.add_norm(
                x.reshape(-1, k), None, self.weight, self.eps, round_normed=self.round_normed
            )
            return out.view(x.shape)
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.round_normed:
            return self.weight * h.to(dtype)
        return (h * self.weight.float()).to(dtype)
