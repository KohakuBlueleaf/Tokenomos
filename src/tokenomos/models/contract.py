"""The model framework's contracts: what a model declares and what the engine hands it.

A model is assembled like a Togeari machine: a Python declaration (``Architecture.build``) wires
block instances (embedding, decoder layers of a mixer + a feed-forward, final norm, head) read
from a ``Checkpoint``. The engine core never knows which model runs: it reads every layer's
``StateSpec``s (what per-sequence memory the layer keeps, by name), allocates them, and passes a
``Step`` (the packed token layout plus the state tensors) to ``Model.forward``.

Row 0 of every per-sequence state is the null row: no live sequence owns it, padding sequences
point at it, and kernels may skip it.
"""

from dataclasses import dataclass, field
from typing import Protocol

import torch


@dataclass(frozen=True)
class PagedKV:
    """Full attention: K and V for every position, in pages (``k_eq_v``: one tensor for both)."""

    kv_heads: int
    head_dim: int
    k_eq_v: bool = False


@dataclass(frozen=True)
class RingKV:
    """Sliding attention: K and V for the last ``window`` positions."""

    kv_heads: int
    head_dim: int
    window: int
    k_eq_v: bool = False


@dataclass(frozen=True)
class SharedKV:
    """No memory of its own: reads layer ``source``'s KV (Gemma 4's KV-shared layers)."""

    source: int


@dataclass(frozen=True)
class ConvState:
    """A causal conv's last ``width - 1`` inputs: ``[dim, width - 1]`` per sequence."""

    dim: int
    width: int


@dataclass(frozen=True)
class RecurrentState:
    """A fixed-size recurrent state (GDN): ``shape`` per sequence."""

    shape: tuple[int, ...]
    dtype: torch.dtype


StateSpec = PagedKV | RingKV | SharedKV | ConvState | RecurrentState
# (layer index, state name within the layer)
StateKey = tuple[int, str]


@dataclass
class Step:
    """One forward's packed token layout: sequences back to back, ``T`` tokens (padded), ``S``
    sequences. Every tensor lives on the device (a CUDA graph may replay them).

    - ``positions`` [T] int64: each token's position in its sequence;
    - ``cu`` [S + 1] int32: sequence starts; ``tok_seq`` [T] int32: each token's sequence;
    - ``n_tok`` [1] int32: real tokens (the rest is padding);
    - ``slot`` [S] int32: each sequence's state row (``>= 1``); ``init`` [S] int32: 1 when the
      row holds the sequence's earlier state;
    - ``states``: (layer, name) -> that state's tensor (from the layer's ``StateSpec``s);
    - ``rope_pos`` [3, T] int64 or None: multimodal RoPE positions (T, H, W) when they differ
      from ``positions``;
    - ``ids`` [T] int32 or None: the token ids (placeholders included) the rows were embedded
      from, for models whose layers read them again (Gemma 4's per-layer inputs);
    - ``meta``: per-backend metadata (attention page tables), the host offsets ``cu_host`` of a
      prefill, ``put(values, dtype)`` (a host list onto the device without a sync), ``blocks``
      (prefill with bidirectional image runs: per token its run id, -1 outside a run, and the
      run's last position, both [T] int32), and per-step caches the blocks fill (RoPE tables,
      per-layer inputs, shared K / V), keyed by their owner.
    """

    positions: torch.Tensor
    cu: torch.Tensor
    tok_seq: torch.Tensor
    n_tok: torch.Tensor
    slot: torch.Tensor
    init: torch.Tensor
    states: dict[StateKey, torch.Tensor] = field(default_factory=dict)
    rope_pos: torch.Tensor | None = None
    ids: torch.Tensor | None = None
    meta: dict = field(default_factory=dict)


class Mixer(Protocol):
    """A decoder layer's token mixer (attention, short conv, GDN, ...)."""

    # the linear reading the normed input (a fused norm quantizes for it when it is MXFP8)
    input_linear: object
    # whether the mixer also reads the normed input in bf16 when it gets it quantized
    needs_bf16_input: bool

    def state_specs(self) -> dict[str, StateSpec]: ...

    def forward(self, x, xq, step: Step, index: int) -> torch.Tensor: ...


class FeedForward(Protocol):
    # None: no fused quantized input (e.g. a routed MoE)
    input_linear: object | None

    def forward(self, x, xq) -> torch.Tensor: ...


class Model(Protocol):
    def embed(self, ids: torch.Tensor) -> torch.Tensor: ...

    def forward(self, h: torch.Tensor, step: Step) -> torch.Tensor: ...

    def logits(
        self, hidden: torch.Tensor, rows: torch.Tensor | None = None, fp32: bool = True
    ) -> torch.Tensor: ...

    def state_specs(self) -> dict[StateKey, StateSpec]: ...


class Architecture(Protocol):
    """One model family: the Hugging Face ``architectures`` names it serves and its declaration."""

    names: tuple[str, ...]

    def build(self, ckpt, device: torch.device) -> Model: ...
