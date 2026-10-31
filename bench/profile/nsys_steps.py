"""Per decode-step GPU accounting from an nsys sqlite export (``--cuda-graph-trace=node``).

Steps end at each kernel whose name contains ``END`` (default ``_sample_kernel``; vLLM's is
``_gumbel_sample_kernel``); over the last ``N`` steps (or the second half of a short capture):
median wall span, GPU busy (union of kernel intervals) and idle, then time per step by category
and by kernel (with launches per step). Writes the full table as JSON next to the sqlite file.

    python bench/profile/nsys_steps.py reports/nsys/decode256.sqlite [N [END]]
"""

import json
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

CATEGORIES = [
    ("mx_gemm", ("block_scaled", "blockscaled", "b12x")),
    ("head_gemm", ("cutlass_80_tensorop_bf16", "splitKreduce", "wmma_tensorop_bf16")),
    ("attention", ("_decode_kernel", "_prefill_kernel", "_write_kernel")),
    ("conv", ("_varlen_conv_kernel", "_update_state_kernel")),
    ("producers", ("_act_mx_kernel", "_rmsnorm_mx_kernel", "_mul_mx", "_gate_mx")),
    ("sampler", ("_sample_kernel",)),
    ("grammar", ("_mask_kernel", "_advance_kernel")),
    ("bookkeeping", ("_alloc_kernel", "_post_kernel", "_release_kernel", "_compact_kernel")),
]


def category(name: str) -> str:
    for cat, keys in CATEGORIES:
        if any(k in name for k in keys):
            return cat
    return "torch_glue"


def main():
    path = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    end = sys.argv[3] if len(sys.argv) > 3 else "_sample_kernel"
    db = sqlite3.connect(path)
    names = dict(db.execute("select id, value from StringIds"))
    ks = sorted(
        (s, e, names.get(d) or names.get(sn, "?"))
        for s, e, d, sn in db.execute(
            "select start, end, demangledName, shortName from CUPTI_ACTIVITY_KIND_KERNEL"
        )
    )
    ends = [i for i, (_, _, nm) in enumerate(ks) if end in nm]
    steps = []
    for a, b in zip(ends, ends[1:]):
        inside = ks[a + 1 : b + 1]
        busy, cur = 0, ks[a][1]
        for s, e, _ in inside:
            if e > cur:
                busy += e - max(s, cur)
                cur = e
        steps.append((ks[b][1] - ks[a][1], busy, inside))
    last = steps[-n - 4 : -4] if len(steps) > n + 8 else steps[len(steps) // 2 : -2]
    span = statistics.median(x[0] for x in last) / 1e6
    busy = statistics.median(x[1] for x in last) / 1e6
    by, cnt, cat = Counter(), Counter(), Counter()
    for _, _, inside in last:
        for s, e, nm in inside:
            by[nm[:120]] += e - s
            cnt[nm[:120]] += 1
            cat[category(nm)] += e - s
    per = len(last)
    out = {
        "steps": len(steps),
        "measured": per,
        "span_ms": round(span, 4),
        "busy_ms": round(busy, 4),
        "idle_ms": round(span - busy, 4),
        # every step of the capture, to set against a whole-run wall average
        "busy_ms_mean_all": round(statistics.mean(x[1] for x in steps) / 1e6, 4),
        "span_ms_mean_all": round(statistics.mean(x[0] for x in steps) / 1e6, 4),
        "categories_ms": {k: round(v / 1e6 / per, 4) for k, v in cat.most_common()},
        "kernels": [
            {"ms": round(d / 1e6 / per, 4), "launches": round(cnt[nm] / per, 2), "name": nm}
            for nm, d in by.most_common()
        ],
    }
    path.with_suffix(".steps.json").write_text(json.dumps(out, indent=1))
    print(f"span {span:.3f} ms  busy {busy:.3f}  idle {span - busy:.3f}  ({per} steps)")
    for k, v in out["categories_ms"].items():
        print(f"  {k:12s} {v:7.3f}")
    for row in out["kernels"][:25]:
        print(f"  {row['ms']:7.3f} ({row['launches']:5.1f})  {row['name']}")


if __name__ == "__main__":
    main()
