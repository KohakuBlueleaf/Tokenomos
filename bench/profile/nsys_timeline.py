"""Whole-run GPU accounting from an nsys sqlite export, inside the NVTX range ``WINDOW`` (default
``bench``): GPU busy (union of kernel and memcpy intervals) against the window, and every idle
gap attributed to the engine thread's innermost NVTX range at the gap (``admit``, ``encode``,
``prefill``, ``decode``, ``wait``, ``idle``, or ``none``: plain Python between ranges). Also host
time per range and GPU time per range kind (kernels launched inside it, by correlation id).
Writes the tables as JSON next to the sqlite file.

    python bench/profile/nsys_timeline.py reports/nsys/caption.sqlite [WINDOW]
"""

import bisect
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path


def union(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def kind(text: str) -> str:
    return text.split(" ")[0]


def main():
    path = Path(sys.argv[1])
    window = sys.argv[2] if len(sys.argv) > 2 else "bench"
    db = sqlite3.connect(path)
    names = dict(db.execute("select id, value from StringIds"))
    ranges = [
        (s, e, t if t is not None else names.get(tid, "?"), gt)
        for s, e, t, tid, gt in db.execute(
            "select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null"
        )
    ]
    w0, w1 = next((s, e) for s, e, t, _ in ranges if t == window)
    engine = Counter(gt for s, e, t, gt in ranges if kind(t) in ("decode", "admit"))
    tid = engine.most_common(1)[0][0]
    mine = sorted((s, e, t) for s, e, t, gt in ranges if gt == tid and e > w0 and s < w1)
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    # whole graph launches (``--cuda-graph-trace=graph``) count as one interval each
    gpu = [
        (s, e, c)
        for table in (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
        )
        if table in tables
        for s, e, c in db.execute(f"select start, end, correlationId from {table}")
        if e > w0 and s < w1
    ]
    busy = union((max(s, w0), min(e, w1)) for s, e, _ in gpu)
    gaps, cur = [], w0
    for s, e in busy:
        if s > cur:
            gaps.append((cur, s))
        cur = max(cur, e)
    if cur < w1:
        gaps.append((cur, w1))
    starts = [r[0] for r in mine]
    longest = max(e - s for s, e, _ in mine)

    def innermost(t):
        # ranges nest on one thread: the latest-starting range still open at t is the innermost
        for i in range(bisect.bisect_right(starts, t) - 1, -1, -1):
            s, e, txt = mine[i]
            if e > t:
                return kind(txt)
            if s < t - longest:
                break
        return "none"

    idle_by, idle_n = Counter(), Counter()
    for a, b in gaps:
        k = innermost((a + b) // 2)
        idle_by[k] += b - a
        idle_n[k] += 1
    host_by, host_n = Counter(), Counter()
    for s, e, txt in mine:
        host_by[kind(txt)] += min(e, w1) - max(s, w0)
        host_n[kind(txt)] += 1
    # GPU time per launching range: runtime API calls on the engine thread carry the correlation
    calls = sorted(
        (s, c)
        for s, c, gt in db.execute(
            "select start, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"
        )
        if gt == tid and w0 <= s < w1
    )
    owner = {c: innermost(s) for s, c in calls}
    gpu_by = Counter()
    for s, e, c in gpu:
        gpu_by[owner.get(c, "other_thread")] += min(e, w1) - max(s, w0)
    span = w1 - w0
    total_busy = sum(e - s for s, e in busy)
    ms = lambda v: round(v / 1e6, 2)  # noqa: E731
    out = {
        "window_ms": ms(span),
        "gpu_busy_ms": ms(total_busy),
        "gpu_idle_ms": ms(span - total_busy),
        "busy_frac": round(total_busy / span, 4),
        "idle_by_host_range_ms": {k: ms(v) for k, v in idle_by.most_common()},
        "idle_gaps_by_host_range": dict(idle_n.most_common()),
        "host_range_ms": {k: ms(v) for k, v in host_by.most_common()},
        "host_range_count": dict(host_n.most_common()),
        "gpu_ms_by_launching_range": {k: ms(v) for k, v in gpu_by.most_common()},
    }
    path.with_suffix(".timeline.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
