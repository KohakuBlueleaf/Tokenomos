"""The prefix table: prompts that start alike skip prefilling their common start.

An entry is a page-aligned prompt start ``b`` tokens long, keyed by a hash of those tokens. It
keeps the start's ``b / page`` KV pages (shared, never freed with a slot) and a snapshot of
every per-slot state at position ``b`` (conv rows, sliding-window rings, recurrent rows) in a
row of its own pool. A hit copies the snapshot into the slot, points the slot's first pages at
the entry's and prefills from ``b``. A miss inserts: the request's prefill stops a chunk at
``b``, the slot's rows are snapshotted there, and its first pages become the entry's (their ids
come back to the host with a copy read one iteration later). Entries nobody uses are evicted
least recently used first when rows or pages run short; their pages go back on the free stack.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch

from tokenomos.core import pools


@dataclass
class Entry:
    key: bytes
    length: int
    row: int
    # page ids, None until the insert's copy lands
    pages: list[int] | None = None
    users: int = 0
    used: int = 0
    copy: tuple = field(default=(), repr=False)


def key_of(prompt: list[int], length: int) -> bytes:
    return hashlib.blake2b(
        np.asarray(prompt[:length], dtype=np.int32).tobytes(), digest_size=16
    ).digest()


class PrefixTable:
    def __init__(self, specs: dict, page: int, entries: int, device):
        """``entries``: snapshot rows (prompt starts kept at once)."""
        self.specs, self.page, self.device = specs, page, torch.device(device)
        self.snap = pools.build(
            {k: s for k, s in specs.items() if pools.slot_dim(s) is not None},
            entries,
            1,
            page,
            device,
        )
        self.free_rows = list(range(entries, 0, -1))
        self.entries: dict[bytes, Entry] = {}
        self.pending: list[Entry] = []
        self.clock = 0
        self.stats = {"hits": 0, "inserts": 0, "evictions": 0, "tokens_skipped": 0}

    def boundary(self, prompt: list[int], prefix: int) -> int:
        """The page-aligned start to share: within ``prefix`` and short of the whole prompt."""
        b = min(prefix, len(prompt) - 1) // self.page * self.page
        return b if b >= self.page else 0

    def lookup(self, prompt: list[int], b: int) -> tuple[Entry | None, bytes]:
        """(a ready entry for ``prompt[:b]``, its key)."""
        key = key_of(prompt, b)
        e = self.entries.get(key)
        return (e if e is not None and e.pages is not None else None), key

    def known(self, key: bytes) -> bool:
        return key in self.entries

    def take(self, e: Entry) -> None:
        self.clock += 1
        e.users += 1
        e.used = self.clock
        self.stats["hits"] += 1
        self.stats["tokens_skipped"] += e.length

    def drop(self, e: Entry) -> None:
        e.users -= 1

    def reserve(self, key: bytes, length: int) -> Entry | None:
        """A row for a new entry (None: every row is held)."""
        if not self.free_rows:
            return None
        self.clock += 1
        e = Entry(key, length, self.free_rows.pop(), users=1, used=self.clock)
        self.entries[key] = e
        self.stats["inserts"] += 1
        return e

    def victim(self) -> Entry | None:
        """The least recently used ready entry nobody holds."""
        idle = [e for e in self.entries.values() if e.users == 0 and e.pages is not None]
        return min(idle, key=lambda e: e.used) if idle else None

    def evict(self, e: Entry) -> list[int]:
        """Forget ``e``; returns its pages (the caller pushes them back)."""
        del self.entries[e.key]
        self.free_rows.append(e.row)
        self.stats["evictions"] += 1
        return e.pages or []

    def snapshot(self, slots: list[int], entries: list[Entry], table: torch.Tensor, put) -> None:
        """Queue: each slot's rows into its entry's row, and the slot's first pages to the host."""
        src = put(slots, torch.int64)
        pools.copy_rows(
            self.specs, self.snap, put([e.row for e in entries], torch.int64), self._live, src
        )
        for s, e in zip(slots, entries):
            host = table[s, : e.length // self.page].to("cpu", non_blocking=True)
            done = torch.cuda.Event()
            done.record()
            e.copy = (done, host)
            self.pending.append(e)

    def restore(self, slots: list[int], entries: list[Entry], put) -> None:
        """Queue: each entry's rows into its slot's rows."""
        rows = put([e.row for e in entries], torch.int64)
        pools.copy_rows(self.specs, self._live, put(slots, torch.int64), self.snap, rows)

    def settle(self) -> None:
        """Entries whose page ids arrived become ready."""
        keep = []
        for e in self.pending:
            done, host = e.copy
            if done.query():
                e.pages, e.copy = host.tolist(), ()
            else:
                keep.append(e)
        self.pending = keep

    def bind(self, live: dict) -> None:
        """The engine's per-slot states (the rows snapshots copy from and to)."""
        self._live = live
