"""Rotary embedding (neox layout: rotate halves) computed from the positions each step.

``Rope(head_dim, angles, theta, dim=None, sections=None)``: the tables span the first ``dim``
dims of a head (``head_dim`` by default; the rest pass through), ``angles`` frequencies
following the progression ``theta ** (-2i / dim)``, the remaining ``dim // 2 - angles`` zero
(LFM3's zero-tail partial rotation; ``angles = dim // 2`` is plain RoPE, Qwen3.5 rotates
``dim = head_dim / 4`` fully). ``sections`` (T, H, W): interleaved multimodal RoPE, frequency
``j`` reading the H position when ``j % 3 == 1`` and ``j < 3 * sections[1]``, the W position
when ``j % 3 == 2`` and ``j < 3 * sections[2]``, else the T position; the positions come from
``step.rope_pos`` [3, T] (``step.positions`` on every axis when absent).

The frequencies are computed on the CPU as the Hugging Face models do; cos and sin are rounded
to the activation dtype and the rotation runs in float32. cos / sin are computed once per step
and kept in ``step.meta``.
"""

import torch
from torch import nn


class Rope(nn.Module):
    def __init__(
        self,
        head_dim: int,
        angles: int,
        theta: float,
        device,
        dim: int | None = None,
        sections: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        dim = dim or head_dim
        idx = torch.arange(0, 2 * angles, 2, dtype=torch.float32)
        inv = 1.0 / (theta ** (idx / dim))
        inv = torch.cat((inv, torch.zeros(dim // 2 - angles, dtype=torch.float32)))
        self.register_buffer("inv_freq", inv.to(device), persistent=False)
        self.sections = sections
        if sections is not None:
            axis = torch.zeros(dim // 2, dtype=torch.long)
            for a, offset in ((1, 1), (2, 2)):
                axis[offset : 3 * sections[a] : 3] = a
            self.register_buffer("axis", axis.to(device), persistent=False)

    def _freqs(self, step) -> torch.Tensor:
        pos = getattr(step, "rope_pos", None)
        if self.sections is None or pos is None:
            return step.positions.float()[:, None] * self.inv_freq[None, :]
        per_axis = pos.float()[:, :, None] * self.inv_freq[None, None, :]
        return per_axis.gather(0, self.axis[None, None, :].expand(1, pos.shape[1], -1))[0]

    def tables(self, step, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = ("rope", id(self), dtype)
        if key not in step.meta:
            freqs = self._freqs(step)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(dtype).float()[:, None, :]
            sin = emb.sin().to(dtype).float()[:, None, :]
            step.meta[key] = (cos, sin)
        return step.meta[key]

    def forward(self, step, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``q`` [T, H, D], ``k`` [T, Hkv, D]."""
        cos, sin = self.tables(step, q.dtype)
        return _rotate(q, cos, sin), _rotate(k, cos, sin)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    r = cos.shape[-1]
    xf = x.float()
    xr, rest = xf[..., :r], xf[..., r:]
    half = r // 2
    rot = torch.cat((-xr[..., half:], xr[..., :half]), dim=-1)
    return torch.cat((xr * cos + rot * sin, rest), dim=-1).to(x.dtype)
