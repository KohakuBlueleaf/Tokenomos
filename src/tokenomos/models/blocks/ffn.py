"""Gated feed-forward: ``down(act(gate(x)) * up(x))``, gate and up fused into one projection
(``gate_up``, gate first); ``act`` SiLU (SwiGLU) or tanh GELU (GeGLU, Gemma)."""

import torch
import torch.nn.functional as F
from torch import nn

from tokenomos.models.blocks.linear import Linear

ACTS = {"silu": "silu_mul", "gelu_tanh": "gelu_mul"}


class SwiGLU(nn.Module):
    def __init__(self, gate_up: Linear, down: Linear, act: str = "silu"):
        super().__init__()
        if act not in ACTS:
            raise ValueError(f"gated feed-forward activation {act!r}")
        self.gate_up, self.down = gate_up, down
        self.act = act

    @property
    def input_linear(self) -> Linear:
        return self.gate_up

    def forward(self, x: torch.Tensor | None, xq: tuple | None) -> torch.Tensor:
        h = self.gate_up.quantized(*xq) if xq is not None else self.gate_up(x)
        if self.down.quantizes:
            return self.down.quantized(*self.down.act(ACTS[self.act], h))
        g, u = h.chunk(2, dim=-1)
        if self.act == "silu":
            return self.down(F.silu(g) * u)
        return self.down(F.gelu(g, approximate="tanh") * u)
