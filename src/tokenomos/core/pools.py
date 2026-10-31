"""Per-layer state tensors built from the model's ``StateSpec``s.

- ``PagedKV`` -> ``[2, pages, page, Hkv, D]`` (K, V), shared page table;
- ``RingKV`` -> ``[2, slots + 1, window, Hkv, D]``, one ring per slot;
- ``ConvState`` -> ``[slots + 1, dim, width - 1]`` stored dim-contiguous (a transposed view, so
  a token's ``dim`` loads coalesce); ``RecurrentState`` -> ``[slots + 1, *shape]``.

``k_eq_v`` layers keep K and V apart (V is the unrotated projection). ``SharedKV`` layers own
nothing. Everything is in the work dtype except a recurrent state's own dtype.
"""

from dataclasses import replace

import torch

from tokenomos.core.floats import work
from tokenomos.models.contract import ConvState, PagedKV, RecurrentState, RingKV, SharedKV


def bounded(specs: dict, max_len: int) -> dict:
    """``specs`` with every ring at most ``max_len`` rows: a shorter sequence never wraps it."""
    return {
        key: replace(s, window=max_len) if isinstance(s, RingKV) and s.window > max_len else s
        for key, s in specs.items()
    }


def page_bytes(specs: dict, page: int) -> int:
    """Bytes one page costs across every paged layer."""
    per = [2 * page * s.kv_heads * s.head_dim for s in specs.values() if isinstance(s, PagedKV)]
    return sum(per) * work().itemsize


def slot_bytes(specs: dict) -> int:
    """Bytes one slot costs across every per-slot state (rings, conv rows, recurrent rows)."""
    total = 0
    size = work().itemsize
    for s in specs.values():
        match s:
            case RingKV(kv_heads=h, head_dim=d, window=w):
                total += 2 * w * h * d * size
            case ConvState(dim=dim, width=width):
                total += dim * (width - 1) * size
            case RecurrentState(shape=shape, dtype=dtype):
                total += torch.Size(shape).numel() * dtype.itemsize
    return total


def fit(
    slots: int | None,
    entries: int | None,
    per_slot: int,
    pages_per_slot: int,
    budget: int,
    top: int = 256,
    top_entries: int = 64,
) -> tuple[int, int]:
    """``slots`` and prefix ``entries``, those left None sized to ``budget`` bytes: a slot costs its
    state rows, ``pages_per_slot`` bytes of pages and a quarter entry's state row. Under 16 slots
    the rows are worth more as slots: no entries."""
    if slots is None:
        cost = per_slot + pages_per_slot + (per_slot // 4 if entries is None else 0)
        slots = top if cost == 0 else max(1, min(top, budget // cost))
        if entries is None and slots < 16:
            slots = max(1, min(top, budget // (per_slot + pages_per_slot)))
    if entries is None:
        entries = min(top_entries, slots // 4) if slots >= 16 else 0
    return slots, entries


def slot_dim(spec) -> int | None:
    """The slot axis of a state built for ``spec`` (None: not per slot)."""
    match spec:
        case RingKV():
            return 1
        case ConvState() | RecurrentState():
            return 0
    return None


def copy_rows(specs: dict, dst: dict, dst_rows, src: dict, src_rows) -> None:
    """Every per-slot state's rows ``src_rows`` of ``src`` into rows ``dst_rows`` of ``dst``
    (index tensors on the device; rings, conv rows and recurrent rows alike)."""
    for key, spec in specs.items():
        d = slot_dim(spec)
        if d is not None:
            dst[key].index_copy_(d, dst_rows, src[key].index_select(d, src_rows))


def build(specs: dict, slots: int, pages: int, page: int, device) -> dict:
    states = {}
    dtype16 = work()
    for key, s in specs.items():
        match s:
            case PagedKV(kv_heads=h, head_dim=d):
                shape = (2, pages, page, h, d)
                states[key] = torch.zeros(shape, dtype=dtype16, device=device)
            case RingKV(kv_heads=h, head_dim=d, window=w):
                shape = (2, slots + 1, w, h, d)
                states[key] = torch.zeros(shape, dtype=dtype16, device=device)
            case ConvState(dim=dim, width=width):
                rows = torch.zeros(slots + 1, width - 1, dim, dtype=dtype16, device=device)
                states[key] = rows.transpose(1, 2)
            case RecurrentState(shape=shape, dtype=dtype):
                states[key] = torch.zeros(slots + 1, *shape, dtype=dtype, device=device)
            case SharedKV():
                pass
    return states
