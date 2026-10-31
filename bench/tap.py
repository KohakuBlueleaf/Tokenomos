"""Timing taps on a built engine, for a bench; the engine's source stays as it is. Every decode
graph replay lands between two CUDA timing events (GPU seconds per step, per bucket), encode and
prefill spans keep their sizes, and the host methods of the loop (``_admit``, ``_encode``,
``_prefill``, ``_read``, ``_release``, the staging arena's ``begin``, ``submit``) record wall and
thread-CPU seconds. Starved: time between decode steps with free slots while nothing waits and
prepared requests are still to come.

    tap = Tap(engine)
    tap.start(expected=len(rows))
    ...
    tap.stop()
    report = tap.report(engine.stats.stamps)
"""

import statistics
import threading
import time
from collections import defaultdict

import torch


def timing_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def seconds(a: torch.cuda.Event, b: torch.cuda.Event) -> float:
    return a.elapsed_time(b) / 1e3


class _Graph:
    """A decode graph whose replay is timed on the GPU and on the host."""

    def __init__(self, graph, bucket: int, tap: "Tap"):
        self.graph, self.bucket, self.tap = graph, bucket, tap

    def replay(self) -> None:
        tap = self.tap
        tap.on_step()
        a, b = timing_event(), timing_event()
        a.record()
        t0 = time.perf_counter()
        self.graph.replay()
        s = tap.host["replay"]
        s[0] += 1
        s[1] += time.perf_counter() - t0
        b.record()
        tap.steps.append((self.bucket, a, b))


class Tap:
    def __init__(self, engine, thread: int | None = None):
        """``thread``: ident of the thread running the engine loop (default the calling one)."""
        self.eng = engine
        # no per-thread clock on Windows: ``engine_thread_cpu_s`` reads 0 there
        self.clock = None
        if hasattr(time, "pthread_getcpuclockid"):
            self.clock = time.pthread_getcpuclockid(thread or threading.get_ident())
        self.reset()
        for name in ("_admit", "_encode", "_prefill", "_read", "_release"):
            self.wrap(engine, name)
        self.wrap(engine.staging, "begin", "staging_begin")
        submit = engine.submit

        def counted(req) -> None:
            self.entered.append(time.perf_counter())
            submit(req)

        engine.submit = counted
        self.wrap(engine, "submit")
        timed = engine._timed

        def sized(kind, fn, *args):
            out = timed(kind, fn, *args)
            work = args[0]
            n = len([0 for _, r in work for _ in r.images]) if kind == "encode" else 0
            n = n or sum(c[3] for c in work) if kind == "prefill" else n
            self.spans.append((kind, n, *engine.stats.spans[-1][1:]))
            return out

        engine._timed = sized
        engine.graphs = {b: _Graph(g, b, self) for b, g in engine.graphs.items()}

    def reset(self) -> None:
        self.host = defaultdict(lambda: [0, 0.0, 0.0])
        self.steps, self.spans, self.entered = [], [], []
        self.starved_s, self.starved_steps, self.expected = 0.0, 0, 0
        self.t0 = self.t1 = self.last = 0.0
        self.cpu0 = self.cpu1 = 0.0

    def wrap(self, owner, name: str, label: str | None = None) -> None:
        """``owner.name`` adds its calls, wall and thread-CPU seconds to ``host[label]``."""
        fn, label = getattr(owner, name), label or name.strip("_")

        def timed(*args, **kw):
            w0, c0 = time.perf_counter(), time.thread_time()
            try:
                return fn(*args, **kw)
            finally:
                s = self.host[label]
                s[0] += 1
                s[1] += time.perf_counter() - w0
                s[2] += time.thread_time() - c0

        setattr(owner, name, timed)

    def thread_cpu(self) -> float:
        return time.clock_gettime(self.clock) if self.clock is not None else 0.0

    def on_step(self) -> None:
        now, e = time.perf_counter(), self.eng
        dt, self.last = now - self.last, now
        pending = len(self.entered) < self.expected
        if pending and e.free_slots and not e.waiting and e.inbox.empty():
            self.starved_s += dt
            self.starved_steps += 1

    def start(self, expected: int = 0) -> None:
        self.reset()
        self.expected = expected
        self.t0 = self.last = time.perf_counter()
        self.cpu0 = self.thread_cpu()

    def stop(self) -> None:
        torch.cuda.synchronize()
        self.t1 = time.perf_counter()
        self.cpu1 = self.thread_cpu()

    def report(self, stamps=()) -> dict:
        """``stamps``: the engine's ``stats.stamps`` of the same window."""
        wall = self.t1 - self.t0
        buckets: dict = {}
        ms = [seconds(a, b) * 1e3 for _, a, b in self.steps]
        for (b, _, _), m in zip(self.steps, ms):
            row = buckets.setdefault(b, {"steps": 0, "gpu_s": 0.0, "full": []})
            row["steps"] += 1
            row["gpu_s"] += m / 1e3
        if len(stamps) == len(self.steps):
            for (_, b, bound), m in zip(stamps, ms):
                if bound == b:
                    buckets[b]["full"].append(m)
        for row in buckets.values():
            full = row.pop("full")
            row["gpu_s"] = round(row["gpu_s"], 4)
            row["ms_mean"] = round(row["gpu_s"] / row["steps"] * 1e3, 3)
            row["full_steps"] = len(full)
            row["full_ms_median"] = round(statistics.median(full), 3) if full else None
        encode = [(n, seconds(a, b)) for k, n, a, b in self.spans if k == "encode" and n]
        prefill = [(n, seconds(a, b)) for k, n, a, b in self.spans if k == "prefill"]
        enc_s, pre_s = sum(s for _, s in encode), sum(s for _, s in prefill)
        dec_s = sum(ms) / 1e3
        images, tokens = sum(n for n, _ in encode), sum(n for n, _ in prefill)
        return {
            "wall_s": round(wall, 3),
            "encode_gpu_s": round(enc_s, 3),
            "prefill_gpu_s": round(pre_s, 3),
            "decode_gpu_s": round(dec_s, 3),
            "gpu_uncovered_s": round(wall - enc_s - pre_s - dec_s, 3),
            "encode_calls": len(encode),
            "encode_images": images,
            "encode_batch_mean": round(images / max(len(encode), 1), 1),
            "encode_batch_max": max((n for n, _ in encode), default=0),
            "encode_img_s": round(images / enc_s, 1) if enc_s else None,
            "prefill_calls": len(prefill),
            "prefill_tokens": tokens,
            "prefill_tok_s": round(tokens / pre_s, 1) if pre_s else None,
            "buckets": {str(b): buckets[b] for b in sorted(buckets)},
            "engine_thread_cpu_s": round(self.cpu1 - self.cpu0, 3),
            "host": {
                k: {"calls": c, "wall_s": round(w, 4), "cpu_s": round(p, 4)}
                for k, (c, w, p) in sorted(self.host.items())
            },
            "starved_s": round(self.starved_s, 3),
            "starved_steps": self.starved_steps,
            "entered_last_s": round(self.entered[-1] - self.t0, 3) if self.entered else None,
        }
