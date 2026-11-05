"""The engine: device-resident slots, one CUDA graph per decode bucket, the host one step behind.

Per host iteration: admit queued requests into free slots (pages reserved for the prompt and
``reserve_new`` tokens), prefill them eagerly in chunks of ``prefill_budget`` tokens (the last
chunk samples each request's first token and appends its slot to the active list), replay the
decode graph of the smallest bucket holding the active list, then read the previous step's event
(alive bitmap, token counts, list length, stack depth) and gather finished slots' tokens
asynchronously. The host never waits on the step it just queued.

Decode graph: rows of the active list (dead ones on the null slot) -> page alloc -> embed ->
model (paged attention) -> logits -> sample -> post (finish test) -> release -> compact ->
event. This module is the Python reference of the host loop; its decisions are the ones a Rust
port must reproduce.
"""

import importlib
import queue
import threading
import time
from bisect import bisect_left
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch.cuda import nvtx

from tokenomos.core import pools
from tokenomos.core.grammar import DeviceGrammar
from tokenomos.core.prefix import Entry, PrefixTable
from tokenomos.core.staging import Staging
from tokenomos.core.state import NSTOP, SlotTable
from tokenomos.kernels.attention import paged as kattn
from tokenomos.kernels.engine import slots as kslots
from tokenomos.kernels.grammar import device as kgram
from tokenomos.kernels.mxfp8 import ops as mx
from tokenomos.kernels.nvfp4 import ops as nv
from tokenomos.kernels.sampling.sample import sample
from tokenomos.models import imageprep
from tokenomos.models.blocks.attention import use_backend
from tokenomos.models.contract import Step

BUCKETS = (8, 16, 32, 64, 96, 128, 192, 256, 384, 512)
# transient bytes per prefill token per hidden unit (peak allocated on 31B, 2148-8192 token chunks)
PREFILL_BYTES = 100
# ``Stats.stamps`` keeps between one and two times this many steps, the newest
STAMPS = 1 << 20


@dataclass
class Request:
    prompt: list[int]
    max_new: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int = 0
    stop: tuple[int, ...] = ()
    # prepared images (the model's processor output, CPU), in placeholder order
    images: list = field(default_factory=list)
    # called with the request from the engine's thread once ``result`` is set
    on_done: Callable | None = None
    # called from the engine's thread with each run of new token ids, ahead of ``on_done``
    on_tokens: Callable | None = None
    sent: int = 0
    # called (engine thread) once the device copies of ``images`` completed
    release: Callable | None = None
    # a grammar id of the engine's ``DeviceGrammar`` (``compile``): the reply follows it
    grammar: int | None = None
    # leading prompt tokens other requests may share (the prefix table; 0: none)
    prefix: int = 0
    # multimodal RoPE positions [3, len(prompt)] int32 (CPU); filled at submit for a model whose
    # processor defines ``rope`` and a request with images
    rope: torch.Tensor | None = None
    # image runs ``(first, end)`` of a model with bidirectional runs; filled at submit
    runs: list[tuple[int, int]] = field(default_factory=list)
    result: "Result | None" = None
    # wall-clock stamps (``time.time``) of the request's stages: ``chat`` / ``prep_start`` /
    # ``prepared`` / ``prep_end`` / ``ready`` (``LLM.submit_chat``), ``submit``, ``queued``,
    # ``inbox`` (taken off the inbox), ``admit``, ``prefilled`` (its last prefill chunk
    # launched), ``done``; and sizes: ``images``, ``pixels`` (prepared), ``cached`` prompt
    # tokens (a prefix hit)
    times: dict = field(default_factory=dict)


@dataclass
class Result:
    tokens: list[int]
    finish: str
    # grammar requests: the device state each token was sampled under
    states: list[int] | None = None


@dataclass
class _Run:
    req: Request
    pages: int
    since: int
    # the prefix entry this run reads (hit) or fills (insert), and the pages it hands that entry
    entry: Entry | None = None
    kept: int = 0
    # prompt tokens already in the slot at admission (a hit's prefix)
    start: int = 0
    # an insert: the prefill stops a chunk here and snapshots
    insert_at: int = 0


def _delta(r: Request) -> int:
    """Generated tokens' RoPE position minus their sequence position."""
    return 0 if r.rope is None else int(r.rope.max()) + 1 - len(r.prompt)


def _cut(runs: list[tuple[int, int]], start: int, take: int) -> int:
    """``take`` tokens from ``start``, moved so the chunk does not end inside a run: back to the
    run's start, or past its end when the run opens the chunk."""
    end = start + take
    for a, b in runs:
        if a < end < b:
            return a - start if a > start else b - start
    return take


@dataclass
class Stats:
    steps: int = 0
    pauses: int = 0
    prefill_tokens: int = 0
    prefill_forwards: int = 0
    prefill_s: float = 0.0
    host_s: float = 0.0
    wait_s: float = 0.0
    wall_s: float = 0.0
    buckets: dict = field(default_factory=dict)
    # (perf_counter, bucket, rows bound) at each decode step's enqueue
    stamps: list = field(default_factory=list)
    # GPU seconds (CUDA events) of the image encoder and of prefill forwards; host seconds of
    # admission (encode + prefill launches, syncs included)
    encode_gpu_s: float = 0.0
    prefill_gpu_s: float = 0.0
    admit_host_s: float = 0.0
    spans: list = field(default_factory=list)
    # grammar transitions resolved on the host (each paused its row one step) and their seconds
    grammar_misses: int = 0
    grammar_s: float = 0.0


class Engine:
    def __init__(
        self,
        model,
        slots: int | None = None,
        page: int = 32,
        max_len: int = 4096,
        new_cap: int = 1024,
        pages: int | None = None,
        kv_fraction: float = 0.25,
        kv_reserve: int | None = None,
        kv_leave: int = 0,
        prefill_budget: int | None = None,
        buckets: tuple[int, ...] = BUCKETS,
        graphs: bool = True,
        staging_bytes: int = 32 << 20,
        grammar: DeviceGrammar | None = None,
        prefix_entries: int | None = None,
        admit_batch: int | None = None,
        reserve_new: int = 256,
        stream_every: int = 4,
        device: str = "cuda",
    ):
        """``slots`` (None: sized to the memory, at most 256; a ring costs a slot ``min(window,
        max_len)`` rows). ``pages``: KV pages (default: the free memory after the slot states less
        ``kv_reserve`` bytes for vision, prefill and graph buffers, at least ``kv_fraction`` of it;
        ``kv_reserve`` None: a third of the free memory, at most 5 GiB). ``kv_leave``: bytes kept
        out of all sizing for other engines on the card. ``prefill_budget``:
        tokens in one prefill forward (None: what a tenth of the free memory covers, at most
        8192). ``graphs`` False runs the decode step eagerly (same kernels). ``grammar``: the device
        grammar requests with ``grammar`` set follow (None: no grammar kernels in the step).
        ``prefix_entries``: prompt starts the prefix table keeps (0: no prefix table; None: a
        quarter of the slots, at most 64, none under 16 slots). ``admit_batch``: free slots to
        gather while rows run and that many requests wait (None: an eighth of the slots).
        ``reserve_new``: new tokens a request reserves pages for; past that it draws on the pool,
        ``new_cap`` tokens of which stay back so the oldest row can always finish while the newest
        sit out (at ``new_cap`` or more: every request reserves all it can use)."""
        self.model, self.device = model, torch.device(device)
        self.grammar, self.g_running = grammar, 0
        proc = importlib.import_module(model.processor) if model.processor else None
        self.rope_fn = getattr(proc, "rope", None)
        use_backend(model, "paged")
        specs = pools.bounded(model.state_specs(), max_len)
        width = -(-max_len // page)
        per_page = pools.page_bytes(specs, page)
        # blocks the allocator caches from loading count as free
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(self.device)[0] - kv_leave
        transient = PREFILL_BYTES * model.embedding.shape[1]
        prefill_budget = prefill_budget or min(8192, free // (10 * transient))
        if kv_reserve is None:
            kv_reserve = min(5 << 30, free // 3)
        if slots is None or prefix_entries is None:
            budget = max(free - kv_reserve, int(free * kv_fraction))
            typical = min(max_len, 1024 + reserve_new)
            slots, prefix_entries = pools.fit(
                slots, prefix_entries, pools.slot_bytes(specs), per_page * typical // page, budget
            )
        if pages is None:
            if per_page == 0:
                pages = slots * width + 1
            else:
                # the slot states and the prefix table's snapshot rows come first
                rows = slots + prefix_entries
                free -= rows * pools.slot_bytes(specs)
                if free <= 0:
                    raise ValueError(
                        f"{rows} slot rows of {pools.slot_bytes(specs) >> 20} MiB exceed the free "
                        "memory: fewer slots or prefix entries, or a smaller state format"
                    )
                budget = max(free - kv_reserve, int(free * kv_fraction))
                pages = min(budget // per_page, slots * width + 1)
        self.slots, self.page, self.max_len, self.new_cap = slots, page, max_len, new_cap
        self.prefill_budget = prefill_budget
        self.admit_batch = admit_batch or max(1, slots // 8)
        self.st = SlotTable(slots, pages, page, max_len, new_cap, self.device)
        self.states = pools.build(specs, slots, pages, page, self.device)
        self.prefix = None
        if prefix_entries:
            self.prefix = PrefixTable(specs, page, prefix_entries, self.device)
            self.prefix.bind(self.states)
        self.buckets = sorted({b for b in buckets if b < slots} | {slots})
        self.staging = Staging(staging_bytes, self.device)
        s1 = slots + 1
        i32 = dict(dtype=torch.int32, device=self.device)
        self.ar = torch.arange(s1, **i32)
        self.ones = torch.ones(s1, **i32)
        self.n_tok = {b: torch.tensor([b], **i32) for b in self.buckets}
        # event record: alive, n_new, list length, free-stack top, then grammar state, miss, room
        # and need per slot
        self.ev_dev = torch.zeros(6 * s1 + 2, **i32)
        self.ev_host = [torch.zeros(6 * s1 + 2, dtype=torch.int32, pin_memory=True) for _ in "ab"]
        self.ev = [torch.cuda.Event() for _ in "ab"]
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        if graphs:
            self._capture()
        self.free_slots = list(range(slots, 0, -1))
        self.reserve_new = reserve_new
        self.headroom = 0 if reserve_new >= new_cap else -(-new_cap // page) + 1
        self.free_pages = pages - 1 - self.headroom
        # the device stack's depth as of step ``top_at``'s event, and the rows sitting out
        self.top_free, self.top_at = pages - 1, -1
        self.paused: set[int] = set()
        self.running: dict[int, _Run] = {}
        # slots whose requests take their tokens as they generate, read every ``stream_every`` steps
        self.streaming: set[int] = set()
        self.stream_every = stream_every
        # (event after an encode's image copies, its requests): released once the event is done
        self.copying: deque[tuple[torch.cuda.Event, list[Request]]] = deque()
        self.inbox: queue.Queue[Request] = queue.Queue()
        self.waiting: deque[Request] = deque()
        self.step_id = self.iteration = self.admitted = 0
        self.admitted_at: dict[int, int] = {}
        self.stats = Stats()
        # ``trace_on()``: one record per loop iteration (host seconds per phase, queue gauges)
        # and the (iteration, kind, start, end) CUDA event pairs of its encode / prefill / decode
        self.trace: list[dict] | None = None
        self.trace_spans: list[tuple] = []
        self._it: dict | None = None
        # the engine thread appends under it; ``trace_take`` swaps under it (any thread)
        self.trace_lock = threading.Lock()

    def trace_on(self) -> None:
        """Record every loop iteration from now on (``trace_take`` hands them over)."""
        with self.trace_lock:
            self.trace, self.trace_spans = [], []

    def trace_take(self) -> list[dict]:
        """The iteration records finished since the last take, with their GPU milliseconds per
        kind (waits for their events); spans of an iteration not finished yet stay for the next
        take. Any thread."""
        with self.trace_lock:
            if self.trace is None:
                return []
            recs, spans = self.trace, self.trace_spans
            self.trace, self.trace_spans = [], []
        by_it = {r["it"]: r for r in recs}
        later = []
        for span in spans:
            it, kind, a, b = span
            rec = by_it.get(it)
            if rec is None:
                later.append(span)
                continue
            b.synchronize()
            key = f"{kind}_gpu_ms"
            rec[key] = rec.get(key, 0.0) + a.elapsed_time(b)
        if later:
            with self.trace_lock:
                self.trace_spans[:0] = later
        return recs

    # ---- device work ----

    def _decode_body(self, b: int) -> None:
        st, model = self.st, self.model
        lst = st.lst[:b]
        # rows waiting on a grammar transition sit this step out on the null slot
        slot = torch.where((st.alive[lst] != 0) & (st.g_wait[lst] == 0), lst, 0)
        ctx = st.seq_len[slot]
        kslots.alloc(slot, ctx + 1, st.n, st, maxneed=1)
        step = Step(
            positions=(ctx + st.rope_delta[slot]).long(),
            cu=self.ar[: b + 1],
            tok_seq=self.ar[:b],
            n_tok=self.n_tok[b],
            slot=slot,
            init=self.ones[:b],
            states=self.states,
            ids=st.last_tok[slot],
            meta={"paged": {"decode": True, "ctx": ctx, "table": st.table}},
        )
        logits = model.logits(model(model.embed(step.ids), step), fp32=False)
        tok = self._sample(logits, slot)
        kslots.post(tok, slot, st.n, st, adv=1)
        kslots.release(st.lst, st.n, st)
        kslots.compact(st.lst, st.n, st.alive)
        s1 = self.slots + 1
        self.ev_dev[:s1].copy_(st.alive)
        self.ev_dev[s1 : 2 * s1].copy_(st.n_new)
        self.ev_dev[2 * s1 : 2 * s1 + 1].copy_(st.n)
        self.ev_dev[2 * s1 + 1 : 2 * s1 + 2].copy_(st.top)
        if self.grammar is not None:
            for i, t in enumerate((st.g_state, st.g_miss, st.g_room, st.g_need)):
                self.ev_dev[(2 + i) * s1 + 2 : (3 + i) * s1 + 2].copy_(t)
            st.g_miss.fill_(-1)

    def _sample(self, logits: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """Grammar mask, sample, grammar step (decode and the last prefill chunk alike)."""
        st, g = self.st, self.grammar
        if g is not None:
            kgram.apply(logits, slot, st.g_state, g.mask_of, g.masks, st.g_trace, st.n_new)
        tok = sample(logits, slot, st.temp, st.top_p, st.top_k, st.min_p, st.seed, st.n_new)
        if g is not None:
            kgram.advance(tok, slot, st, g)
        return tok

    def _capture(self) -> None:
        """Every bucket's graph, captured while no slot is alive (the warmup runs touch only the
        null slot)."""
        pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        for b in self.buckets:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self._decode_body(b)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            # The graph owns the cuBLAS workspace it records: PyTorch's per-stream workspace is
            # held only by its own map, which any later capture in the process may clear (and
            # free under this graph); cleared before, the workspace lands in this graph's pool,
            # cleared after, no later call reuses that block outside it
            torch._C._cuda_clearCublasWorkspaces()
            try:
                with torch.cuda.graph(g, pool=pool):
                    self._decode_body(b)
            finally:
                torch._C._cuda_clearCublasWorkspaces()
            self.graphs[b] = g
        torch.cuda.synchronize()

    def _admit(self, admits: list[tuple[int, Request]]) -> None:
        st, put = self.st, self.staging.put
        idx = put([s for s, _ in admits], torch.int32).long()
        ints = put(
            [[1, 0, 0, r.max_new, r.top_k, r.seed & 0x7FFFFFFF, _delta(r)] for _, r in admits],
            torch.int32,
        )
        floats = put([[r.temperature, r.top_p, r.min_p] for _, r in admits], torch.float32)
        stops = [list(r.stop) + [-1] * (NSTOP - len(r.stop)) for _, r in admits]
        cols = (st.alive, st.seq_len, st.n_new, st.max_new, st.top_k, st.seed, st.rope_delta)
        for col, t in enumerate(cols):
            t[idx] = ints[:, col]
        for col, t in enumerate((st.temp, st.top_p, st.min_p)):
            t[idx] = floats[:, col]
        st.stops[idx] = put(stops, torch.int32)
        if self.grammar is not None:
            rows = [
                (-1, -1, 0) if r.grammar is None else self.grammar.initial(r.grammar)
                for _, r in admits
            ]
            rows = put(rows, torch.int32)
            for col, t in enumerate((st.g_state, st.g_room, st.g_need)):
                t[idx] = rows[:, col]
            # index_fill_: a Python scalar in `t[idx] = v` is a pageable copy, which syncs
            st.g_wait.index_fill_(0, idx, 0)
            st.g_miss.index_fill_(0, idx, -1)
            self.grammar.flush(put)
            self.g_running += sum(r.grammar is not None for _, r in admits)
        runs = [self.running[s] for s, _ in admits]
        hits = [(s, run) for (s, _), run in zip(admits, runs) if run.start]
        if hits:
            self._restore(hits)
        mm = self._timed("encode", self._encode, admits)
        work = [[s, r.prompt, run.start, run.insert_at, r] for (s, r), run in zip(admits, runs)]
        while any(w[2] < len(w[1]) for w in work):
            chunk, budget, reached = [], self.prefill_budget, []
            for w in work:
                # an insert's chunk ends at its boundary, where its rows are snapshotted
                end = w[3] if w[2] < w[3] else len(w[1])
                take = _cut(w[4].runs, w[2], min(end - w[2], budget))
                if take > 0:
                    chunk.append((w[0], w[1], w[2], take))
                    budget -= take
                    w[2] += take
                    if w[2] == w[3]:
                        reached.append(w[0])
                if budget <= 0:
                    break
            self._timed("prefill", self._prefill, chunk, mm)
            if reached:
                self._snapshot(reached)
        now = time.time()
        for (_, r), run in zip(admits, runs):
            r.times["prefilled"] = now
            r.times["cached"] = run.start

    def _restore(self, hits: list[tuple[int, _Run]]) -> None:
        """Hit slots start from their entry: its pages first in the table, its rows restored."""
        st, put, p = self.st, self.staging.put, self.page
        flat, pages = [], []
        for s, run in hits:
            k = run.start // p
            flat += [s * st.table.stride(0) + j for j in range(k)]
            pages += run.entry.pages
        st.table.view(-1)[put(flat, torch.int64)] = put(pages, torch.int32)
        idx = put([s for s, _ in hits], torch.int64)
        k = put([run.start // p for _, run in hits], torch.int32)
        st.nblocks[idx] = k
        st.shared[idx] = k
        st.seq_len[idx] = put([run.start for _, run in hits], torch.int32)
        self.prefix.restore([s for s, _ in hits], [run.entry for _, run in hits], put)

    def _snapshot(self, slots: list[int]) -> None:
        """Insert slots reached their boundary: snapshot rows, keep their first pages."""
        put = self.staging.put
        runs = [self.running[s] for s in slots]
        self.prefix.snapshot(slots, [r.entry for r in runs], self.st.table, put)
        idx = put(slots, torch.int64)
        self.st.shared[idx] = put([r.kept for r in runs], torch.int32)

    def _timed(self, kind: str, fn, *args):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        nvtx.range_push(kind)
        h0 = time.perf_counter()
        a.record()
        out = fn(*args)
        b.record()
        nvtx.range_pop()
        self.stats.spans.append((kind, a, b))
        it = self._it
        if it is not None:
            with self.trace_lock:
                self.trace_spans.append((it["it"], kind, a, b))
            it[f"{kind}_host_s"] = it.get(f"{kind}_host_s", 0.0) + time.perf_counter() - h0
            it[f"{kind}_n"] = it.get(f"{kind}_n", 0) + 1
        return out

    def _encode(self, admits: list[tuple[int, Request]]):
        """Every admitted image through the encoder in one batch: (features, slot -> (placeholder
        positions in the prompt, first feature row)), or None without images."""
        imgs = [im for _, r in admits for im in r.images]
        if not imgs:
            return None
        enc = self.model.encoders["image"]
        it = self._it
        if it is not None:
            it["enc_images"] = it.get("enc_images", 0) + len(imgs)
            it["enc_pixels"] = it.get("enc_pixels", 0) + sum(
                im.shape[0] * im.shape[1] for im in imgs
            )
        feats = enc([im.to(self.device, non_blocking=True) for im in imgs], self.staging.put)
        copied = torch.cuda.Event()
        copied.record()
        self.copying.append((copied, [r for _, r in admits if r.images]))
        table, base = {}, 0
        for s, r in admits:
            n = self._image_tokens(r)
            if n:
                pos = [i for i, t in enumerate(r.prompt) if t == self.model.image_token]
                table[s] = (pos, base)
            base += n
        return feats, table

    def _image_tokens(self, r: Request) -> int:
        """The language tokens of ``r``'s images."""
        if not r.images:
            return 0
        enc = self.model.encoders["image"]
        return sum(
            enc.tokens((im.shape[0] // enc.patch, im.shape[1] // enc.patch)) for im in r.images
        )

    def _prefill(self, chunk: list[tuple[int, list[int], int, int]], mm=None) -> None:
        """One packed forward over ``(slot, prompt, done, take)``: tokens ``done .. done+take``;
        ``mm`` from ``_encode`` puts image features on the chunk's placeholder rows."""
        st, model, put = self.st, self.model, self.staging.put
        cu_host, ids, pos, seq = [0], [], [], []
        ropes = [self.running[c[0]].req.rope for c in chunk]
        for i, (_, prompt, done, take) in enumerate(chunk):
            cu_host.append(cu_host[-1] + take)
            ids += prompt[done : done + take]
            pos += range(done, done + take)
            seq += [i] * take
        rope_pos = None
        if any(r is not None for r in ropes):
            rope_pos = put(
                torch.cat(
                    [
                        (
                            r[:, done : done + take]
                            if r is not None
                            else torch.arange(done, done + take, dtype=torch.int32).expand(3, -1)
                        )
                        for r, (_, _, done, take) in zip(ropes, chunk)
                    ],
                    dim=1,
                ),
                torch.int64,
            )
        slot = put([c[0] for c in chunk], torch.int32)
        ctx = put([c[2] for c in chunk], torch.int32)
        cu = put(cu_host, torch.int32)
        rows = put([len(chunk)], torch.int32)
        kslots.alloc(slot, ctx + put([c[3] for c in chunk], torch.int32), rows, st)
        tiles: dict[int, tuple] = {}

        def tile(bm: int):
            if bm not in tiles:
                s, q0 = kattn.tiles(cu_host, bm)
                tiles[bm] = (put(s, torch.int32), put(q0, torch.int32))
            return tiles[bm]

        step = Step(
            positions=put(pos, torch.int64),
            cu=cu,
            tok_seq=put(seq, torch.int32),
            n_tok=put([cu_host[-1]], torch.int32),
            slot=slot,
            init=put([int(c[2] > 0) for c in chunk], torch.int32),
            states=self.states,
            rope_pos=rope_pos,
            ids=put(ids, torch.int32),
            meta={
                "paged": {"decode": False, "ctx": ctx, "table": st.table, "tiles": tile},
                "cu_host": cu_host,
                "put": put,
            },
        )
        if getattr(model, "image_runs", False):
            run, hi, rid = [-1] * len(pos), list(pos), 0
            for i, (s, _, done, take) in enumerate(chunk):
                for a, b in self.running[s].req.runs:
                    if a < done + take and done < b:
                        first = cu_host[i] + a - done
                        run[first : first + b - a] = [rid] * (b - a)
                        hi[first : first + b - a] = [b - 1] * (b - a)
                        rid += 1
            step.meta["blocks"] = (put(run, torch.int32), put(hi, torch.int32))
        x = model.embed(step.ids)
        if mm is not None:
            feats, table = mm
            rows, frows = [], []
            for i, (s, _, done, take) in enumerate(chunk):
                if s in table:
                    pos, base = table[s]
                    lo, hi = bisect_left(pos, done), bisect_left(pos, done + take)
                    rows += [cu_host[i] + p - done for p in pos[lo:hi]]
                    frows += range(base + lo, base + hi)
            if rows:
                x[put(rows, torch.int64)] = feats[put(frows, torch.int64)]
        h = model(x, step)
        st.seq_len[slot.long()] = ctx + (cu[1:] - cu[:-1])
        last = [i for i, (_, p, done, take) in enumerate(chunk) if done + take == len(p)]
        if last:
            fslot = put([chunk[i][0] for i in last], torch.int32)
            rows = put([cu_host[i + 1] - 1 for i in last], torch.int64)
            logits = model.logits(h, rows, fp32=False)
            tok = self._sample(logits, fslot)
            count = put([len(last)], torch.int32)
            kslots.post(tok, fslot, count, st, adv=0)
            kslots.append(st.lst, st.n, fslot, count)
        self.stats.prefill_tokens += cu_host[-1]
        self.stats.prefill_forwards += 1
        it = self._it
        if it is not None:
            it["pf_tokens"] = it.get("pf_tokens", 0) + cu_host[-1]
            it["pf_seqs"] = it.get("pf_seqs", 0) + len(chunk)

    # ---- host loop ----

    def submit(self, req: Request) -> None:
        """Queue ``req`` (any thread); the loop picks it up on its next iteration. A request the
        engine could not serve fails here: past this point an error ends the engine thread."""
        req.times["submit"] = time.time()
        req.times["images"] = len(req.images)
        req.times["pixels"] = sum(int(im.shape[0]) * int(im.shape[1]) for im in req.images)
        if req.max_new < 1:
            raise ValueError(f"max_new {req.max_new}: a reply has at least one token")
        if len(req.prompt) + req.max_new > self.max_len or req.max_new > self.new_cap:
            raise ValueError(f"request too long: {len(req.prompt)} + {req.max_new}")
        held = (self.st.pages - 1) * self.page
        if len(req.prompt) + req.max_new > held or self._need(req) * self.page > (
            held - self.headroom * self.page
        ):
            raise ValueError(
                f"request needs {len(req.prompt) + req.max_new} tokens of the {held} the page "
                f"pool holds, {self.headroom * self.page} of them kept back"
            )
        if not req.prompt or len(req.stop) > NSTOP:
            raise ValueError("empty prompt or too many stop tokens")
        if req.grammar is not None and self.grammar is None:
            raise ValueError("a grammar request on an engine built without grammar")
        if req.images and "image" not in self.model.encoders:
            raise ValueError("this model takes no images")
        n = self._image_tokens(req)
        if n != (req.prompt.count(self.model.image_token) if n else 0):
            raise ValueError(f"the prompt's image placeholders do not match {n} image tokens")
        if req.images and req.rope is None and self.rope_fn is not None:
            req.rope = self.rope_fn(req.prompt, req.images, self.model.image_token)
        if req.images and getattr(self.model, "image_runs", False):
            req.runs = imageprep.runs(req.prompt, self.model.image_token)
        # pinned on the caller's thread, so the loop's image copies never block it
        req.images = [im if im.is_pinned() else im.pin_memory() for im in req.images]
        req.times["queued"] = time.time()
        self.inbox.put(req)

    def warmup(self, lengths=None) -> None:
        """Compile every kernel variant serving will hit before the first request: prompts of
        ``lengths`` one forward each (both sides of the 16-divisibility specializations, every GEMM
        routing range up to a full multi-chunk prefill), a batch per row block, then images of a few
        grid sizes when the model has an encoder; two tokens each; stats reset."""
        top = self.prefill_budget + 100
        routes = {m for t in (nv.table(), mx.table()) for r in t.values() for m, _ in r}
        lengths = lengths or {1, 17, 100, 160, 600, 1000, 2100, 4200, top} | routes
        cap = min(self.max_len, (self.st.pages - 1 - self.headroom) * self.page) - 2
        for n in sorted({min(n, cap) for n in lengths}):
            self.generate([Request([1] * n, max_new=2)])
        # the slot kernels' variants, one per power of two of a forward's rows
        block = 16
        while block // 2 < self.slots:
            rows = min(block // 2 + 1, self.slots)
            n = min(self.page, self.prefill_budget // rows)
            self.generate([Request([1] * n, 2) for _ in range(rows)])
            block *= 2
        if "image" in self.model.encoders:
            enc = self.model.encoders["image"]
            unit = enc.patch * enc.merge
            for gh, gw in ((8, 8), (16, 24), (32, 32)):
                n = enc.tokens((gh * unit // enc.patch, gw * unit // enc.patch))
                if n + 20 >= cap:
                    continue
                im = torch.zeros(gh * unit, gw * unit, 3, dtype=torch.uint8)
                prompt = [1] * 9 + [self.model.image_token] * n + [1] * 9
                self.generate([Request(prompt, 2, images=[im])])
        torch.cuda.synchronize()
        self.stats = Stats()

    def generate(self, requests: list[Request]) -> list[Result]:
        """Submit ``requests`` and serve until the engine is idle."""
        for r in requests:
            self.submit(r)
        self.run()
        return [r.result for r in requests]

    def run(self, stop: threading.Event | None = None) -> None:
        """Serve the inbox: until nothing is queued or running (``stop`` None), or until ``stop``
        is set (a serving thread; it blocks briefly on the inbox while idle)."""
        waiting, outs = self.waiting, deque()
        bound, start = 0, time.perf_counter()
        while True:
            taken = 0
            while True:
                try:
                    r = self.inbox.get_nowait()
                except queue.Empty:
                    break
                r.times["inbox"] = time.time()
                waiting.append(r)
                taken += 1
            if not waiting and not self.running and not outs:
                self._release(block=True)
                if stop is None or stop.is_set():
                    break
                nvtx.range_push("idle")
                try:
                    r = self.inbox.get(timeout=0.02)
                    r.times["inbox"] = time.time()
                    waiting.append(r)
                    taken += 1
                except queue.Empty:
                    continue
                finally:
                    nvtx.range_pop()
            t0, wait = time.perf_counter(), 0.0
            it = None
            if self.trace is not None:
                it = self._it = {
                    "it": self.iteration,
                    "t": time.time(),
                    "inbox": taken,
                    "waiting": len(waiting),
                    "running": len(self.running),
                    "free_slots": len(self.free_slots),
                    "free_pages": self.free_pages,
                }
            self.staging.begin(self.iteration)
            if it is not None:
                it["begin_s"] = time.perf_counter() - t0
            admits, tokens = [], 0
            if self.prefix is not None:
                self.prefix.settle()
            # a one-row prefill forward costs nearly a full one: free slots gather into a batch
            few = len(self.free_slots) < min(self.admit_batch, len(waiting))
            hold = bool(self.running) and few
            while waiting and self.free_slots and not hold and not self.paused:
                r = waiting[0]
                run = self._plan(r)
                if self.prefix is not None and run.pages > self.free_pages:
                    self._evict(run.pages)
                left = len(r.prompt) - run.start
                short = run.pages > self.free_pages or (self.headroom and run.pages > self._room())
                if short or (admits and tokens + left > self.prefill_budget):
                    if run.entry is not None:
                        self._unplan(run)
                    break
                waiting.popleft()
                s = self.free_slots.pop()
                self.free_pages -= run.pages
                self.running[s] = run
                if r.on_tokens is not None:
                    self.streaming.add(s)
                admits.append((s, r))
                tokens += left
            if it is not None:
                it["plan_s"] = time.perf_counter() - t0 - it["begin_s"]
                it["admits"] = len(admits)
                it["admit_tokens"] = tokens
                it["held"] = hold
            if admits:
                a0 = time.perf_counter()
                stamp = time.time()
                for _, r in admits:
                    r.times["admit"] = stamp
                nvtx.range_push(f"admit {len(admits)}")
                self._admit(admits)
                nvtx.range_pop()
                self.stats.admit_host_s += time.perf_counter() - a0
                if it is not None:
                    it["admit_s"] = time.perf_counter() - a0
                self.admitted += len(admits)
                bound += len(admits)
            queued = bound > 0
            if queued:
                b = next(x for x in self.buckets if x >= bound)
                nvtx.range_push(f"decode {b}")
                d0 = time.perf_counter()
                if it is not None:
                    da = torch.cuda.Event(enable_timing=True)
                    da.record()
                if self.graphs:
                    self.graphs[b].replay()
                else:
                    self._decode_body(b)
                if it is not None:
                    db = torch.cuda.Event(enable_timing=True)
                    db.record()
                    with self.trace_lock:
                        self.trace_spans.append((it["it"], "decode", da, db))
                    it["decode_b"], it["decode_rows"] = b, bound
                    it["decode_s"] = time.perf_counter() - d0
                k = self.step_id % 2
                self.ev_host[k].copy_(self.ev_dev, non_blocking=True)
                self.ev[k].record()
                nvtx.range_pop()
                self.admitted_at[self.step_id] = self.admitted
                self.stats.buckets[b] = self.stats.buckets.get(b, 0) + 1
                self.stats.stamps.append((time.perf_counter(), b, bound))
                if len(self.stats.stamps) >= 2 * STAMPS:
                    del self.stats.stamps[:STAMPS]
                self.step_id += 1
                self.stats.steps += 1
            # the previous step's event, or this one's when nothing else is queued
            e = self.step_id - (2 if queued else 1)
            if e >= 0 and e in self.admitted_at:
                w0 = time.perf_counter()
                nvtx.range_push("wait")
                self.ev[e % 2].synchronize()
                nvtx.range_pop()
                wait += time.perf_counter() - w0
                r0 = time.perf_counter()
                bound = self._read(e, outs) + self.admitted - self.admitted_at.pop(e)
                for old in [x for x in self.admitted_at if x < e]:
                    del self.admitted_at[old]
                if it is not None:
                    it["read_s"] = time.perf_counter() - r0
            elif not queued and outs:
                # nothing left on the GPU but the token copies
                w0 = time.perf_counter()
                outs[0][0].synchronize()
                wait += time.perf_counter() - w0
            c0 = time.perf_counter()
            done = 0
            while outs and outs[0][0].query():
                done += len(outs[0][2]) if outs[0][4] else 0
                _collect(outs.popleft())
            c1 = time.perf_counter()
            self._release(block=False)
            self.collect_spans(block=False)
            self.iteration += 1
            self.stats.wait_s += wait
            self.stats.host_s += time.perf_counter() - t0 - wait
            if it is not None:
                it["wait_s"] = wait
                it["collect_s"], it["finished"] = c1 - c0, done
                it["tail_s"] = time.perf_counter() - c1
                it["host_s"] = time.perf_counter() - t0 - wait
                with self.trace_lock:
                    self.trace.append(it)
                self._it = None
        self.stats.wall_s += time.perf_counter() - start
        self.collect_spans()

    def _need(self, r: Request) -> int:
        """Pages ``r`` reserves at admission."""
        return -(-(len(r.prompt) + min(r.max_new, self.reserve_new)) // self.page)

    def _unseen(self) -> int:
        """Pages of the rows admitted after the last event, which its stack depth does not show."""
        return sum(run.pages for run in self.running.values() if run.since > self.top_at)

    def _room(self) -> int:
        """Pages an admission may take: the stack past the headroom and two steps of every row."""
        return self.top_free - self._unseen() - self.headroom - 2 * len(self.running)

    def _pressure(self, top: int) -> None:
        """The stack nears the headroom: only the oldest rows it can feed for two steps run (always
        one, whose finish the headroom guarantees), the newest sit out until pages return."""
        free = top - self._unseen() - self.headroom
        if not self.paused and free >= 2 * len(self.running):
            return
        rows = sorted(self.running, key=lambda s: self.running[s].since)
        want = set(rows[max(1, free // 2) :])
        on, off = want - self.paused, self.paused - want
        put = self.staging.put
        if on:
            self.st.g_wait.index_fill_(0, put(sorted(on), torch.int64), 2)
            self.stats.pauses += len(on)
        if off:
            self.st.g_wait.index_fill_(0, put(sorted(off), torch.int64), 0)
        self.paused = want

    def _plan(self, r: Request) -> _Run:
        """``r``'s run: its page reservation and, with a prefix, the entry it hits or inserts."""
        need = self._need(r)
        run = _Run(r, need, self.step_id)
        b = self.prefix.boundary(r.prompt, r.prefix) if self.prefix and r.prefix else 0
        if not b:
            return run
        hit, key = self.prefix.lookup(r.prompt, b)
        if hit is not None:
            self.prefix.take(hit)
            run.entry, run.start, run.pages = hit, b, need - b // self.page
        elif not self.prefix.known(key):
            if not self.prefix.free_rows:
                self._evict(0, rows=True)
            entry = self.prefix.reserve(key, b)
            if entry is not None:
                run.entry, run.kept, run.insert_at = entry, b // self.page, b
        return run

    def _unplan(self, run: _Run) -> None:
        """Undo ``_plan``'s hold on its entry (the request stays queued)."""
        stats = self.prefix.stats
        if run.start:
            self.prefix.drop(run.entry)
            stats["hits"] -= 1
            stats["tokens_skipped"] -= run.start
        else:
            self.prefix.evict(run.entry)
            stats["inserts"] -= 1
            stats["evictions"] -= 1

    def _evict(self, pages: int, rows: bool = False) -> None:
        """Evict idle entries until ``pages`` fit (or one row frees, ``rows``)."""
        freed = []
        while (rows and not self.prefix.free_rows) or (not rows and pages > self.free_pages):
            v = self.prefix.victim()
            if v is None:
                break
            got = self.prefix.evict(v)
            freed += got
            self.free_pages += len(got)
            self.top_free += len(got)
        if freed:
            put = self.staging.put
            kslots.push(put(freed, torch.int32), put([len(freed)], torch.int32), self.st)

    def _release(self, block: bool) -> None:
        while self.copying and (block or self.copying[0][0].query()):
            ev, reqs = self.copying.popleft()
            ev.synchronize()
            for r in reqs:
                r.images = []
                if r.release is not None:
                    r.release()

    def collect_spans(self, block: bool = True) -> None:
        """Fold the recorded encode / prefill event pairs into ``stats``: all of them (waiting), or
        with ``block`` False those already done, so a serving engine holds none for long."""
        spans, done = self.stats.spans, 0
        for kind, a, b in spans:
            if block:
                b.synchronize()
            elif not b.query():
                break
            name = f"{kind}_gpu_s"
            setattr(self.stats, name, getattr(self.stats, name) + a.elapsed_time(b) / 1e3)
            done += 1
        del spans[:done]

    def _read(self, e: int, outs) -> int:
        """Finished slots in step ``e``'s event: free them, queue their tokens' copy. Returns the
        active-list length after step ``e``."""
        ev = self.ev_host[e % 2]
        s1 = self.slots + 1
        alive = ev[:s1].tolist()
        if self.g_running:
            self._resolve(ev, alive)
        done = [s for s, run in self.running.items() if run.since <= e and not alive[s]]
        rows = []
        if self.streaming and e % self.stream_every == 0:
            rows = [s for s in self.streaming if alive[s] and self.running[s].since <= e]
        if rows:
            n_new = ev[s1 : 2 * s1].tolist()
            host = self.st.out_tok[self.staging.put(rows, torch.int64)].to("cpu", non_blocking=True)
            ready = torch.cuda.Event()
            ready.record()
            entries = [(self.running[s].req, n_new[s]) for s in rows]
            outs.append((ready, host, entries, None, False))
        if done:
            n_new = ev[s1 : 2 * s1].tolist()
            idx = self.staging.put(done, torch.int64)
            host = self.st.out_tok[idx].to("cpu", non_blocking=True)
            gram = any(self.running[s].req.grammar is not None for s in done)
            trace = self.st.g_trace[idx].to("cpu", non_blocking=True) if gram else None
            ready = torch.cuda.Event()
            ready.record()
            entries = [(self.running[s].req, n_new[s]) for s in done]
            outs.append((ready, host, entries, trace, True))
            for s in done:
                run = self.running.pop(s)
                self.streaming.discard(s)
                self.free_slots.append(s)
                self.free_pages += run.pages - run.kept
                self.g_running -= run.req.grammar is not None
                if run.entry is not None:
                    self.prefix.drop(run.entry)
        self.top_free, self.top_at = int(ev[2 * s1 + 1]), e
        if self.headroom:
            self._pressure(self.top_free)
        return int(ev[2 * s1])

    def _resolve(self, ev: torch.Tensor, alive: list[int]) -> None:
        """Rows that missed a grammar transition in this event: step them on the host, add the
        entries, and let the rows resume (from the step after the one already queued)."""
        s1 = self.slots + 1
        state, miss, room, need = (ev[(2 + i) * s1 + 2 : (3 + i) * s1 + 2] for i in range(4))
        slots = [s for s in (miss != -1).nonzero()[:, 0].tolist() if alive[s] and s in self.running]
        if not slots:
            return
        t0 = time.perf_counter()
        rows = [(int(state[s]), int(room[s]), int(need[s]), int(miss[s])) for s in slots]
        t1 = time.perf_counter()
        after = self.grammar.resolve(rows)
        t2 = time.perf_counter()
        put = self.staging.put
        idx = put(slots, torch.int64)
        after = put(after, torch.int32)
        for col, t in enumerate((self.st.g_state, self.st.g_room, self.st.g_need)):
            t[idx] = after[:, col]
        ready = [s for s in slots if s not in self.paused]
        if ready:
            self.st.g_wait.index_fill_(0, put(ready, torch.int64), 0)
        t3 = time.perf_counter()
        self.grammar.flush(put)
        tm = self.grammar.timing
        tm["rows_s"] = tm.get("rows_s", 0.0) + t1 - t0
        tm["resolve_s"] = tm.get("resolve_s", 0.0) + t2 - t1
        tm["write_s"] = tm.get("write_s", 0.0) + t3 - t2
        self.stats.grammar_misses += len(slots)
        self.stats.grammar_s += time.perf_counter() - t0


def _collect(item) -> None:
    _, host, entries, trace, final = item
    rows = host.tolist()
    traces = trace.tolist() if trace is not None else [None] * len(rows)
    for (req, n), row, tr in zip(entries, rows, traces):
        toks = row[:n]
        if req.on_tokens is not None and n > req.sent:
            req.on_tokens(toks[req.sent :])
            req.sent = n
        if not final:
            continue
        stop = bool(toks) and toks[-1] in req.stop
        states = tr[:n] if req.grammar is not None else None
        req.result = Result(toks, "stop" if stop else "length", states)
        req.times["done"] = time.time()
        if req.on_done is not None:
            req.on_done(req)
