"""Gated DeltaNet mixer (Qwen3.5 / 3.8 ``linear_attention``).

``qkv | z | b | a = proj(x)`` (one fused projection); ``qkv`` through a causal depthwise conv of
width ``W`` with SiLU (its last ``W - 1`` inputs in the layer's ``conv`` row); ``q``, ``k``
[T, Hk, dk] and ``v`` [T, Hv, dv] from the conv output; per value head
``g = -exp(A_log) * softplus(a + dt_bias)``, ``beta = sigmoid(b)``; q / k L2-normed, q scaled
by ``dk ** -0.5``; the gated delta rule over the layer's ``rec`` state (value head ``i`` reads
key head ``i // (Hv / Hk)``); the per-head gated norm with ``z``; ``out``.

The state is ``[Hv, dv, dk]`` per slot in ``state_format`` (``kernels.gdn.state``). A decode step
(one token per sequence, ``step.meta["paged"]["decode"]``) runs the fused recurrent kernel; any
other step splits, L2-norms and gates in one kernel (``fused_post_conv_prep``), runs the chunked
kernel (``prefill``) from each sequence's state (zero for a fresh sequence) and writes the final
states back. A ``Stacked`` projection's parts are read without concatenating them.
An NVFP4 ``out`` takes the gated norm quantized in the same pass.
"""

import torch
from torch import nn

from tokenomos.core.floats import matmul, work
from tokenomos.kernels.conv.varlen import update_state, varlen_conv
from tokenomos.kernels.gdn import prefill as gp
from tokenomos.kernels.gdn import state as gstate
from tokenomos.kernels.gdn.fla import FLA_CHUNK_SIZE, chunk_gated_delta_rule, fused_post_conv_prep
from tokenomos.kernels.gdn.fla.index import chunk_indices_host
from tokenomos.kernels.gdn.norm import norm_gate, norm_gate_nvfp4
from tokenomos.models.blocks.linear import Linear, Stacked, columns
from tokenomos.models.contract import ConvState, RecurrentState, Step

try:
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_chunk_gated_delta_rule
except ImportError:  # the flashinfer prefill backend is then unavailable
    fi_chunk_gated_delta_rule = None

PREFILL = ("fla", "flashinfer", "fused")


def chunk_table(step: Step) -> torch.Tensor:
    """The FLA kernel's ``[sequence, chunk]`` table of this step, built once from the host
    offsets (``step.meta["cu_host"]``) and staged with ``step.meta["put"]``."""
    key = ("gdn", "chunks")
    if key not in step.meta:
        rows = chunk_indices_host(step.meta["cu_host"], FLA_CHUNK_SIZE)
        step.meta[key] = step.meta["put"](rows, torch.int32).view(-1, 2)
    return step.meta[key]


def fused_table(step: Step, bt: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused kernels' ``gp.table`` of this step, built once as ``chunk_table``."""
    key = ("gdn", "fused")
    if key not in step.meta:
        rows, starts = gp.table(step.meta["cu_host"], bt)
        put = step.meta["put"]
        step.meta[key] = (put(rows, torch.int32).view(-1, 2), put(starts, torch.int32))
    return step.meta[key]


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        proj: Linear,
        out: Linear,
        conv: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm: torch.Tensor,
        eps: float,
        heads_k: int,
        heads_v: int,
        dim_k: int,
        dim_v: int,
        state_format: str = "fp32",
        prefill: str = "auto",
    ):
        """``conv`` [conv_dim, W]; ``A_log`` / ``dt_bias`` [Hv] float32; ``norm`` [dv].
        ``prefill``: the chunked kernel, ``fla`` (vendored flash-linear-attention, Triton),
        ``flashinfer`` (FlashInfer's fused delta-rule prefill, 2.8x FLA's speed on the RTX 5090
        at the same error, sm90 and up) or ``fused`` (``kernels.gdn.prefill``, every card and
        work dtype); ``auto``: ``flashinfer`` where it runs, else ``fused``."""
        sixteen = work() != torch.float32
        if prefill == "auto":
            sm90 = conv.is_cuda and torch.cuda.get_device_capability(conv.device)[0] >= 9
            fi = sm90 and sixteen and fi_chunk_gated_delta_rule is not None
            prefill = "flashinfer" if fi else "fused"
        if prefill not in PREFILL:
            raise ValueError(f"GDN prefill {prefill!r}")
        if prefill == "flashinfer" and not sixteen:
            raise ValueError("GDN prefill flashinfer takes fp16 or bf16, not the fp32 work dtype")
        if state_format == "fp8" and work() != torch.bfloat16:
            raise ValueError(f"GDN state fp8 needs the bf16 work dtype, not {work()}")
        if prefill == "flashinfer" and fi_chunk_gated_delta_rule is None:
            raise ValueError("GDN prefill flashinfer: flashinfer.gdn_prefill not installed")
        super().__init__()
        self.prefill = prefill
        self.proj, self.out = proj, out
        self.conv = nn.Parameter(conv, requires_grad=False)
        self.A_log = nn.Parameter(A_log, requires_grad=False)
        self.dt_bias = nn.Parameter(dt_bias, requires_grad=False)
        self.norm = nn.Parameter(norm, requires_grad=False)
        self.eps = eps
        self.hk, self.hv, self.dk, self.dv = heads_k, heads_v, dim_k, dim_v
        self.key_dim, self.value_dim = heads_k * dim_k, heads_v * dim_v
        self.conv_dim, self.width = conv.shape
        self.state_format = state_format
        self.needs_bf16_input = False

    @property
    def input_linear(self) -> Linear:
        return self.proj

    def state_specs(self) -> dict:
        shape, dtype = gstate.shape(self.hv, self.dv, self.dk, self.state_format)
        return {
            "conv": ConvState(self.conv_dim, self.width),
            "rec": RecurrentState(shape, dtype),
        }

    def forward(self, x: torch.Tensor | None, xq: tuple | None, step: Step, index: int):
        if xq is not None:
            parts = [self.proj.quantized(*xq)]
        elif isinstance(self.proj, Stacked):
            parts = self.proj.parts(x)
        else:
            parts = [self.proj(x)]
        cd, vd, hv = self.conv_dim, self.value_dim, self.hv
        qkv, z = columns(parts, 0, cd), columns(parts, cd, cd + vd)
        b, a = columns(parts, cd + vd, cd + vd + hv), columns(parts, cd + vd + hv, cd + vd + 2 * hv)
        cstate, pool = step.states[(index, "conv")], step.states[(index, "rec")]
        mixed = varlen_conv(
            qkv, cstate, self.conv, None, step.tok_seq, step.cu, step.slot, step.init, step.n_tok
        )
        update_state(qkv, cstate, step.cu, step.slot, step.init, self.width)
        paged = step.meta.get("paged")
        if paged is not None and paged["decode"]:
            kd = self.key_dim
            q, k, v = mixed[:, :kd], mixed[:, kd : 2 * kd], mixed[:, 2 * kd :]
            o = torch.empty(mixed.shape[0], vd, dtype=mixed.dtype, device=mixed.device)
            gstate.decode(
                q, k, v, a, b, self.A_log, self.dt_bias, pool, step.slot, o,
                self.dk**-0.5, True, self.state_format,
            )  # fmt: skip
        else:
            o = self._chunked(mixed, a, b, pool, step)
        if self.out.fmt == "nvfp4":
            return self.out.quantized(*norm_gate_nvfp4(o, z, self.norm, self.eps, self.out.igs))
        return self.out(norm_gate(o, z, self.norm, self.eps))

    def _chunked(self, mixed, a, b, pool, step: Step) -> torch.Tensor:
        n = mixed.shape[0]
        mm = matmul(mixed.dtype)
        q, k, v, g, beta = fused_post_conv_prep(
            mixed, a, b, self.A_log, self.dt_bias, self.hk, self.dk, self.dv, dtype=mm
        )
        init = gstate.load(pool, step.slot, step.init, self.dk, self.state_format)
        if self.prefill == "flashinfer":
            o = torch.empty(n, self.hv, self.dv, dtype=mixed.dtype, device=mixed.device)
            final = torch.empty_like(init)
            fi_chunk_gated_delta_rule(
                q, k, v, torch.exp(g), beta, scale=self.dk**-0.5, initial_state=init,
                output_final_state=True, cu_seqlens=step.cu, output=o, output_state=final,
            )  # fmt: skip
            gstate.store(pool, step.slot, final, self.state_format)
            return o.view(n, self.value_dim)
        if self.prefill == "fused":
            tile = gp.config(mixed.device)
            o = torch.empty(n, self.hv, self.dv, dtype=mixed.dtype, device=mixed.device)
            chunks, starts = fused_table(step, tile[0])
            _, final = gp.chunk_gated_delta_rule(
                q, k, v, g, beta, self.dk**-0.5, init, step.cu, chunks, starts, o, tile
            )
            gstate.store(pool, step.slot, final, self.state_format)
            return o.view(n, self.value_dim)
        # the chunked kernel's products are all fp16 under fp32; its output is not
        wide = None if mm == mixed.dtype else mixed.new_empty(n * self.value_dim)
        o, final = chunk_gated_delta_rule(
            q[None],
            k[None],
            v[None],
            g[None],
            beta[None],
            scale=self.dk**-0.5,
            initial_state=init,
            output_final_state=True,
            cu_seqlens=step.cu,
            chunk_indices=chunk_table(step),
            core_attn_out=wide,
        )
        gstate.store(pool, step.slot, final, self.state_format)
        return o.view(n, self.value_dim)
