"""Measure the NVFP4 GEMM kernels on this GPU and write its routing table.

Every NVFP4 linear shape of the given models (head NVFP4 too) at every decode bucket and the
prefill sizes, on every CUTLASS tactic and every b12x plan: the per-call time of a graph of
back-to-back calls over weight copies past the L2. The fastest wins each token count;
consecutive token counts with the same winner merge into one range (a pick within ``TIE`` of the
fastest stays). Writes ``src/tokenomos/kernels/nvfp4/tables/<gpu>.json`` (the table ``ops``
routes by) and ``reports/nvfp4_tune_<gpu>.json`` (every time, the default's beside the winner's).

    python bench/profile/tune_nvfp4.py /models/qwen35-4b-nvfp4 ...

Across GPUs of one kind: ``--shard I/N`` on each (each writes its part report), then
``--merge N``.
"""

import argparse
import json
from pathlib import Path

import torch
import triton

from tokenomos import models
from tokenomos.core.engine import BUCKETS
from tokenomos.kernels.mxfp8.ops import gpu_slug
from tokenomos.kernels.nvfp4 import ops as nv
from tokenomos.models.blocks.linear import Linear

ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda"
MS = sorted(set(BUCKETS) | {1, 2, 4, 768, 1024, 2048, 4096, 8192})
REPS = 20
# the weight bytes one timed graph cycles through: past the RTX 5090's 96 MiB L2
COLD_BYTES = 256 << 20
TIE = 0.02


def timed(fn, weights: list) -> float:
    """Per-call time of ``fn(wq, ws)`` in a graph cycling through ``weights`` (copies enough to
    exceed the L2, as a model's layers stream their weights from DRAM)."""
    reps = max(REPS, len(weights))
    fn(*weights[0])
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fn(*weights[i % len(weights)])
    return triton.testing.do_bench(g.replay, warmup=10, rep=100, return_mode="median") / reps


def shapes(paths: list[str]) -> set[tuple[int, int]]:
    out = set()
    for p in paths:
        model = models.load(p, DEV, models.configure(p, head="nvfp4"))
        out |= {(m.n, m.k) for m in model.modules() if isinstance(m, Linear) and m.fmt == "nvfp4"}
        del model
        torch.cuda.empty_cache()
    return out


def key(choice: tuple) -> str:
    kind, arg = choice
    return f"{kind}:{arg}" if kind == "cutlass" else "b12x:" + "-".join(str(int(v)) for v in arg)


def measure(shape_list: list[tuple[int, int]]) -> tuple[dict, list]:
    gs = torch.ones(1, device=DEV)
    alpha = torch.ones(1, device=DEV)
    cutlass = [("cutlass", t) for t in range(nv.cutlass_tactics(DEV))]
    table, times = {}, []
    for n, k in shape_list:
        wq, ws = nv.nvfp4_quant(torch.randn(n, k, device=DEV, dtype=torch.bfloat16), gs)
        copies = -(-COLD_BYTES // (wq.numel() + ws.numel() * 4))
        weights = [(wq, ws)] + [(wq.clone(), ws.clone()) for _ in range(copies - 1)]
        winners = []
        for m in MS:
            q, s = nv.nvfp4_quant(torch.randn(m, k, device=DEV, dtype=torch.bfloat16), gs)
            b12x = [("b12x", p) for p in nv.b12x_plans(q, s, wq, ws, alpha)]
            row = {"n": n, "k": k, "m": m, "us": {}}
            ref = nv.run(nv.DEFAULT, q, s, wq, ws, alpha)
            for c in cutlass + b12x:
                try:
                    out = nv.run(c, q, s, wq, ws, alpha)
                    if not torch.equal(out, ref):
                        err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                        if err > 1e-2:
                            row["us"][key(c)] = f"wrong {err:.3g}"
                            continue
                    t = timed(lambda w, x, c=c, q=q, s=s: nv.run(c, q, s, w, x, alpha), weights)
                    row["us"][key(c)] = round(t * 1e3, 3)
                except Exception as e:  # noqa: BLE001 - a tactic the shape cannot take
                    row["us"][key(c)] = f"fail {type(e).__name__}"
            ok = {c: t for c in cutlass + b12x if isinstance(t := row["us"][key(c)], float)}
            best = min(ok, key=ok.get)
            # within TIE of the fastest, the previous token count's pick stays (fewer ranges)
            if winners and winners[-1][1] in ok and ok[winners[-1][1]] <= ok[best] * (1 + TIE):
                best = winners[-1][1]
            row["best"], row["best_us"] = key(best), ok[best]
            row["default_us"] = row["us"][key(nv.DEFAULT)]
            winners.append((m, best))
            times.append(row)
            print(json.dumps({x: row[x] for x in ("n", "k", "m", "best", "best_us", "default_us")}))
        ranges = []
        for m, best in winners:
            if ranges and ranges[-1][1] == best:
                ranges[-1][0] = m
            else:
                ranges.append([m, best])
        table[f"{n}x{k}"] = [
            [top, [c[0], list(c[1]) if c[0] == "b12x" else c[1]]] for top, c in ranges
        ]
    return table, times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--shard", default="0/1", help="I/N: measure every N-th shape from the I-th")
    ap.add_argument("--merge", type=int, default=0, help="merge N shard reports into the table")
    args = ap.parse_args()
    slug = gpu_slug()
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    if args.merge:
        parts = [
            json.loads((reports / f"nvfp4_tune_{slug}.part{i}.json").read_text())
            for i in range(args.merge)
        ]
        table = {key: r for p in parts for key, r in p["shapes"].items()}
        times = [row for p in parts for row in p["times"]]
    else:
        i, n = (int(v) for v in args.shard.split("/"))
        table, times = measure(sorted(shapes(args.models))[i::n])
        if n > 1:
            out = {"times": times, "shapes": table}
            (reports / f"nvfp4_tune_{slug}.part{i}.json").write_text(json.dumps(out, indent=1))
            return
    out = {"gpu": torch.cuda.get_device_name(), "shapes": dict(sorted(table.items()))}
    nv.TABLES.mkdir(exist_ok=True)
    (nv.TABLES / f"{slug}.json").write_text(json.dumps(out, indent=1))
    (reports / f"nvfp4_tune_{slug}.json").write_text(json.dumps({"times": times, **out}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
