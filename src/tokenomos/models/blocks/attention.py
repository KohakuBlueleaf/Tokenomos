"""Attention mixer: fused qkv projection (or qk with ``k_eq_v``: V is the K projection), optional
per-head QK RMSNorm, optional weightless V norm (``v_norm``), RoPE, causal attention (full, or a
sliding window of ``window`` positions including the token itself) with an optional logit
softcap, an optional sigmoid output gate (per head from its own linear, LFM3's ``gate_proj``; or
per channel from ``H * D`` more columns of the fused projection, ``fused_gate``, Qwen3.5's
doubled ``q_proj``), output projection.

KV sharing (Gemma 4): a ``stash`` layer keeps this step's K / V in ``step.meta`` and leaves its
cache write to the last reader (``write_shared``); a reader (``kv_source``) has only Q and
attends over the source's K / V and cache. The write waits so a ring source is not overwritten
by a chunk before its readers see the old window. ``bidirectional``: a prefill's image runs
(``step.meta["blocks"]``) see each other in both directions.

The attention itself goes through ``backend``; ``dense`` is the reference: each sequence of the
packed batch alone (host offsets in ``step.meta["dense_cu"]``) with an explicit causal / window
mask, no cache. ``paged`` is the engine's: the layer's ``(index, "kv")`` state as the cache.
"""

import torch
import torch.nn.functional as F
from torch import nn

from tokenomos.kernels.attention import paged as kattn
from tokenomos.kernels.attention import qk
from tokenomos.models.blocks.linear import Linear, Stacked, columns
from tokenomos.models.blocks.norm import RMSNorm
from tokenomos.models.blocks.rope import Rope
from tokenomos.models.contract import PagedKV, RingKV, SharedKV, Step


def mask(n: int, window: int | None, device, run: torch.Tensor | None = None) -> torch.Tensor:
    """[n, n] bool: query i sees key j iff ``j <= i`` (or both in one run: ``run`` [n] ids, -1
    outside) and, with a window, ``i - j < window``."""
    i = torch.arange(n, device=device)
    m = i[None, :] <= i[:, None]
    if run is not None:
        m |= (run[:, None] == run[None, :]) & (run[:, None] >= 0)
    if window is not None:
        m &= i[:, None] - i[None, :] < window
    return m


def dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    step: Step,
    kv: torch.Tensor | None,
    window: int | None,
    scale: float,
    softcap: float | None,
    write: bool = True,
    bidirectional: bool = False,
) -> torch.Tensor:
    """``q`` [T, H, D], ``k`` / ``v`` [T, Hkv, D] -> [T, H, D]."""
    out = torch.zeros_like(q)
    cu = step.meta["dense_cu"]
    groups = q.shape[1] // k.shape[1]
    blocks = step.meta.get("blocks") if bidirectional else None
    for s in range(len(cu) - 1):
        a, b = cu[s], cu[s + 1]
        if b == a:
            continue
        qs, ks, vs = (t[a:b].transpose(0, 1) for t in (q, k, v))
        m = mask(b - a, window, q.device, blocks[0][a:b] if blocks is not None else None)
        if softcap is None:
            # float32 scores as the kernels compute them: unit-RMS heads score up to ``D``
            o = F.scaled_dot_product_attention(
                qs.float(), ks.float(), vs.float(), attn_mask=m, scale=scale, enable_gqa=True
            ).to(q.dtype)
        else:
            ks, vs = ks.repeat_interleave(groups, 0), vs.repeat_interleave(groups, 0)
            logits = (qs.float() @ ks.float().transpose(1, 2)) * scale
            logits = softcap * torch.tanh(logits / softcap)
            p = logits.masked_fill(~m, float("-inf")).softmax(-1).to(q.dtype)
            o = p @ vs
        out[a:b] = o.transpose(0, 1)
    return out


def paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    step: Step,
    kv: torch.Tensor,
    window: int | None,
    scale: float,
    softcap: float | None,
    write: bool = True,
    bidirectional: bool = False,
) -> torch.Tensor:
    """The engine's backend: ``kv`` [2, ...] the layer's cache (paged, or a ring per slot with a
    window); ``step.meta["paged"]``: ``decode`` (one token per sequence), ``ctx`` [S] positions
    already cached, ``table`` the page table, and for prefill ``tiles(bm)`` -> query tiles.
    ``write``: the new K / V into the cache after attention."""
    m = step.meta["paged"]
    ring = window is not None
    table = None if ring else m["table"]
    if m["decode"]:
        out = kattn.decode(
            q, k, v, kv[0], kv[1], step.slot, m["ctx"], table, scale, window, softcap, ring
        )
    else:
        seq, q0 = m["tiles"](kattn.prefill_block(q.shape[-1])[0])
        out = kattn.prefill(
            q,
            k,
            v,
            kv[0],
            kv[1],
            step.cu,
            step.slot,
            m["ctx"],
            table,
            seq,
            q0,
            scale,
            window,
            softcap,
            ring,
            blocks=step.meta.get("blocks") if bidirectional else None,
        )
    if write:
        kattn.write(
            k, v, kv[0], kv[1], step.tok_seq, step.cu, step.slot, m["ctx"], step.n_tok, table, ring
        )
    return out


BACKENDS = {"dense": dense, "paged": paged}


def use_backend(model: nn.Module, name: str) -> None:
    for m in model.modules():
        if isinstance(m, Attention):
            m.backend = name


class Attention(nn.Module):
    def __init__(
        self,
        qkv: Linear,
        o: Linear,
        heads: int,
        kv_heads: int,
        head_dim: int,
        rope: Rope,
        window: int | None,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
        gate: Linear | None = None,
        scale: float | None = None,
        softcap: float | None = None,
        k_eq_v: bool = False,
        backend: str = "dense",
        fused_gate: bool = False,
        v_norm: float | None = None,
        kv_source: int | None = None,
        stash: bool = False,
        write_shared: bool = False,
        bidirectional: bool = False,
    ):
        """``v_norm``: the weightless V norm's eps (None: no V norm). ``kv_source``: the layer
        whose K / V this one reads (``qkv`` then projects Q only); ``stash``: this layer's K / V
        are read by others; ``write_shared``: the last reader, which writes the source's cache."""
        super().__init__()
        self.qkv, self.o, self.rope = qkv, o, rope
        self.q_norm, self.k_norm, self.gate = q_norm, k_norm, gate
        self.fused_gate = fused_gate
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.window = window
        self.scale = head_dim**-0.5 if scale is None else scale
        self.softcap = softcap
        self.k_eq_v = k_eq_v
        self.backend = backend
        self.v_norm = v_norm
        self.kv_source, self.stash, self.write_shared = kv_source, stash, write_shared
        self.bidirectional = bidirectional

    @property
    def input_linear(self) -> Linear:
        return self.qkv

    @property
    def needs_bf16_input(self) -> bool:
        """A 16-bit gate reads the 16-bit input even when qkv takes it quantized."""
        return self.gate is not None and not self.gate.quantizes

    def state_specs(self) -> dict:
        if self.kv_source is not None:
            return {"kv": SharedKV(self.kv_source)}
        if self.window is None:
            return {"kv": PagedKV(self.kv_heads, self.head_dim, self.k_eq_v)}
        return {"kv": RingKV(self.kv_heads, self.head_dim, self.window, self.k_eq_v)}

    def _qkv(self, parts: list[torch.Tensor], step: Step, n: int):
        hq, hk = self.heads * self.head_dim, self.kv_heads * self.head_dim
        q, k, v = columns(parts, 0, hq), None, None
        if self.kv_source is None:
            k = columns(parts, hq, hq + hk)
            v = k if self.k_eq_v else columns(parts, hq + hk, hq + 2 * hk)
        q = q.reshape(n, self.heads, self.head_dim)
        cos, sin = self.rope.tables(step, q.dtype)
        norm = self.q_norm is not None
        wq = self.q_norm.weight if norm else None
        eq = self.q_norm.eps if norm else 0.0
        rn = self.q_norm.round_normed if norm else True
        if k is None:
            # Q alone: a zero-head K through the same kernel
            empty = q.new_empty(n, 0, self.head_dim)
            q, _ = qk.norm_rope(q, empty, wq, wq, eq, eq, cos[:, 0], sin[:, 0], rn)
            return q, None, None
        k = k.reshape(n, self.kv_heads, self.head_dim)
        v = v.reshape(n, self.kv_heads, self.head_dim)
        wk = self.k_norm.weight if norm else None
        ek = self.k_norm.eps if norm else 0.0
        if self.v_norm is not None:
            return qk.norm_rope(
                q, k, wq, wk, eq, ek, cos[:, 0], sin[:, 0], rn, v=v, eps_v=self.v_norm
            )
        q, k = qk.norm_rope(q, k, wq, wk, eq, ek, cos[:, 0], sin[:, 0], rn)
        return q, k, v

    def forward(self, x: torch.Tensor | None, xq: tuple | None, step: Step, index: int):
        """``x`` the normed input (16-bit), ``xq`` the same quantized (values, scales) or None."""
        n = step.positions.shape[0]
        if xq is not None:
            parts = [self.qkv.quantized(*xq)]
        elif isinstance(self.qkv, Stacked):
            parts = self.qkv.parts(x)
        else:
            parts = [self.qkv(x)]
        hq = self.heads * self.head_dim
        if self.fused_gate:
            g = columns(parts, self.qkv.n - hq, self.qkv.n)
        q, k, v = self._qkv(parts, step, n)
        if self.kv_source is not None:
            k, v = step.meta[("shared_kv", self.kv_source)]
        elif self.stash:
            step.meta[("shared_kv", index)] = (k, v)
        owner = index if self.kv_source is None else self.kv_source
        kv = step.states.get((owner, "kv"))
        write = self.write_shared if self.kv_source is not None else not self.stash
        out = BACKENDS[self.backend](
            q, k, v, step, kv, self.window, self.scale, self.softcap, write, self.bidirectional
        )
        out = out.reshape(n, hq)
        if self.fused_gate:
            if self.o.quantizes:
                return self.o.quantized(*self.o.act("gate", out, g))
            return self.o(out * torch.sigmoid(g))
        if self.gate is not None:
            quantized = xq is not None and self.gate.quantizes
            g = self.gate.quantized(*xq) if quantized else self.gate(x)
            if self.o.quantizes:
                return self.o.quantized(*self.o.act("gate", out, g, self.head_dim))
            out = (out.view(n, self.heads, self.head_dim) * torch.sigmoid(g)[..., None]).view(n, hq)
        return self.o(out)
