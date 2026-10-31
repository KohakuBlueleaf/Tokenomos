"""Gated short-conv mixer (LFM2 / LFM3): ``b, c, x = in_proj(h)``, ``y = conv(b * x)`` causal
depthwise of width ``W`` (optional bias) per sequence, its last ``W - 1`` inputs kept in the
layer's ``conv`` state row, ``out_proj(c * y)``. ``b * x`` is formed inside the conv kernels.
"""

import torch
from torch import nn

from tokenomos.kernels.conv.varlen import update_state, varlen_conv
from tokenomos.models.blocks.linear import Linear
from tokenomos.models.contract import ConvState, Step


class ShortConv(nn.Module):
    def __init__(
        self,
        in_proj: Linear,
        out_proj: Linear,
        conv: torch.Tensor,
        conv_bias: torch.Tensor | None = None,
    ):
        """``conv`` [dim, W] depthwise weight."""
        super().__init__()
        self.in_proj, self.out_proj = in_proj, out_proj
        self.conv = nn.Parameter(conv, requires_grad=False)
        self.conv_bias = (
            nn.Parameter(conv_bias, requires_grad=False) if conv_bias is not None else None
        )
        self.dim, self.width = conv.shape
        self.needs_bf16_input = False

    @property
    def input_linear(self) -> Linear:
        return self.in_proj

    def state_specs(self) -> dict:
        return {"conv": ConvState(self.dim, self.width)}

    def forward(self, x: torch.Tensor | None, xq: tuple | None, step: Step, index: int):
        bcx = self.in_proj.quantized(*xq) if xq is not None else self.in_proj(x)
        b, c, h = bcx.chunk(3, dim=-1)
        state = step.states[(index, "conv")]
        y = varlen_conv(
            b,
            state,
            self.conv,
            self.conv_bias,
            step.tok_seq,
            step.cu,
            step.slot,
            step.init,
            step.n_tok,
            silu=False,
            xmul=h,
        )
        update_state(b, state, step.cu, step.slot, step.init, self.width, xmul=h)
        if self.out_proj.quantizes:
            return self.out_proj.quantized(*self.out_proj.act("mul", c, y))
        return self.out_proj(c * y)
