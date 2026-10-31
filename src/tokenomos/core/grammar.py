"""The device copy of the grammar store: transition table, ``mask_of`` per state, mask bitsets.

The Rust ``Store`` is the source of truth: it interns the states the device may hold and
memoises one mask per clamped state. ``resolve`` turns the rows that missed an entry into new
entries (placed by the host, which mirrors the table's keys, so placement needs no device read)
and into each row's next state; ``flush`` queues the new entries, the new states' mask ids and
the new masks as host-to-device copies ahead of the next step.
"""

import threading
import time

import numpy as np
import torch

from tokenomos.kernels.grammar import device as kgram


class DeviceGrammar:
    def __init__(
        self,
        store,
        device,
        table_bits: int = 20,
        probes: int = 8,
        max_states: int = 1 << 20,
        max_masks: int = 8192,
    ):
        """``store``: a ``tokenomos._rs.grammar.Store``. ``table_bits``: log2 of the table's
        entries (keep it under half full); ``probes``: the longest probe the device walks;
        ``max_states`` / ``max_masks``: device capacities (masks: distinct bitsets)."""
        self.store, self.device, self.probes = store, torch.device(device), probes
        self.longest = store.longest
        # the store is used from the engine thread and from callers compiling schemas
        self.lock = threading.Lock()
        self.keys, self.vals = kgram.empty_table(table_bits, self.device)
        self.cap_mask = (1 << table_bits) - 1
        self.host_keys = np.full(1 << table_bits, -1, dtype=np.int64)
        self.mask_of = torch.zeros(max_states, dtype=torch.int32, device=self.device)
        self.shape_of = torch.full((max_states,), -1, dtype=torch.int32, device=self.device)
        self.body_of = torch.zeros(max_states, dtype=torch.int32, device=self.device)
        self.plain = torch.tensor(store.d_plain(), dtype=torch.int32, device=self.device)
        self.shape_host: list[int] = []
        # host seconds: store calls, entry placement, flush (copies queued)
        self.timing = {"store_s": 0.0, "place_s": 0.0, "flush_s": 0.0}
        # misses by kind: (state or shape, token) entries, (shape, counters) entries
        self.miss_kinds = {"token": 0, "counters": 0}
        self.masks = torch.zeros(max_masks, store.words, dtype=torch.int32, device=self.device)
        self.n_masks = self.n_states = self.entries = 0
        # store mask id -> device row; equal bitsets share a row
        self.row_of_mask: list[int] = []
        self.row_by_bits: dict[bytes, int] = {}
        self.pending: list[tuple[int, int, tuple]] = []
        self.compiled: dict[str, int] = {}

    def compile(self, schema: str) -> int:
        with self.lock:
            g = self.compiled.get(schema)
            if g is None:
                g = self.compiled[schema] = self.store.compile(schema)
            return g

    def initial(self, grammar: int) -> tuple[int, int, int]:
        """The ``(state, room, need)`` a reply of ``grammar`` starts as."""
        with self.lock:
            return self.store.d_initial(grammar)

    def _place(self, key: int, val: tuple) -> None:
        h = kgram.table_hash(key, self.cap_mask)
        for i in range(self.probes):
            idx = (h + i) & self.cap_mask
            if self.host_keys[idx] == key:
                return
            if self.host_keys[idx] == -1:
                self.host_keys[idx] = key
                self.pending.append((idx, key, val))
                self.entries += 1
                return
        raise RuntimeError(
            f"grammar table: no free slot in {self.probes} probes (raise table_bits)"
        )

    def _clamp(self, room: int, need: int) -> tuple[int, int]:
        r = self.longest if room < 0 else min(room, self.longest)
        return r, min(max(need, 0), self.longest)

    def resolve(self, rows: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int]]:
        """Rows ``(state, room, need, miss)`` from the event record -> each row's
        ``(state, room, need)``; every missing entry is added. ``miss >= 0``: the token whose
        ``(state, token)`` entry was missing; ``-2 - shape``: a missing ``(shape, counters)``
        entry (the counters already stepped)."""
        out: list[tuple[int, int, int] | None] = [None] * len(rows)
        tok = [i for i, r in enumerate(rows) if r[3] >= 0]
        cnt = [i for i, r in enumerate(rows) if r[3] < 0]
        self.miss_kinds["token"] += len(tok)
        self.miss_kinds["counters"] += len(cnt)
        tm = self.timing
        with self.lock:
            if tok:
                t0 = time.perf_counter()
                steps = self.store.d_step(*zip(*(rows[i] for i in tok)))
                t1 = time.perf_counter()
                for i, (entry, after) in zip(tok, steps):
                    s, _, _, t = rows[i]
                    key = kgram.key_token(s, self.shape_host[s], t)
                    self._place(key, (entry[0], int(entry[1]), *entry[2:]))
                    if entry[1]:
                        cr, cn = self._clamp(after[1], after[2])
                        self._place(kgram.key_counters(entry[0], cr, cn), (after[0], 0, 0, 0))
                    out[i] = after
                tm["store_s"] += t1 - t0
                tm["place_s"] += time.perf_counter() - t1
            if cnt:
                t0 = time.perf_counter()
                shapes = [-2 - rows[i][3] for i in cnt]
                rooms, needs = [rows[i][1] for i in cnt], [rows[i][2] for i in cnt]
                ids = self.store.d_counters(shapes, rooms, needs)
                t1 = time.perf_counter()
                for i, sh, r, n, s in zip(cnt, shapes, rooms, needs, ids):
                    cr, cn = self._clamp(r, n)
                    self._place(kgram.key_counters(sh, cr, cn), (s, 0, 0, 0))
                    out[i] = (s, r, n)
                tm["store_s"] += t1 - t0
                tm["place_s"] += time.perf_counter() - t1
        return out

    def flush(self, put) -> None:
        """Queue every new table entry, state mask id and mask (``put``: staging copies)."""
        t0 = time.perf_counter()
        with self.lock:
            self._flush(put)
        self.timing["flush_s"] += time.perf_counter() - t0

    def _flush(self, put) -> None:
        if self.pending:
            idx, key, val = zip(*self.pending)
            idx = put(idx, torch.int64)
            self.keys[idx] = put(key, torch.int64)
            self.vals[idx] = put(val, torch.int32)
            self.pending.clear()
        new = np.frombuffer(self.store.take_new(), dtype=np.int32)
        if new.size:
            fresh = []
            for row in new.reshape(-1, self.store.words):
                key = row.tobytes()
                at = self.row_by_bits.get(key)
                if at is None:
                    at = self.row_by_bits[key] = self.n_masks + len(fresh)
                    fresh.append(row)
                self.row_of_mask.append(at)
            if fresh:
                end = self.n_masks + len(fresh)
                if end > self.masks.shape[0]:
                    raise RuntimeError(f"grammar: {end} distinct masks over max_masks")
                src = torch.from_numpy(np.stack(fresh)).pin_memory()
                self.masks[self.n_masks : end].copy_(src, non_blocking=True)
                self.n_masks = end
        n = self.store.d_states
        if n > self.n_states:
            if n > self.mask_of.shape[0]:
                raise RuntimeError(f"grammar: {n} states over max_states")
            new_ids = list(range(self.n_states, n))
            ids = self.store.d_masks(new_ids)
            self.mask_of[self.n_states : n] = put([self.row_of_mask[m] for m in ids], torch.int32)
            shapes = self.store.d_shapes(new_ids)
            self.shape_of[self.n_states : n] = put(shapes, torch.int32)
            self.body_of[self.n_states : n] = put(self.store.d_bodies(new_ids), torch.int32)
            self.shape_host += shapes
            self.n_states = n
