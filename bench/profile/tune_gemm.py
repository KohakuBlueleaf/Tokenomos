"""Measure the MXFP8 GEMM backends on this GPU and write its routing table.

Every MXFP8 linear shape of the given models (head quantized too) at every decode bucket and
the prefill sizes, on each backend: the per-call time of a graph of ``REPS`` back-to-back calls.
The faster backend wins each token count; consecutive token counts with the same winner merge
into one range. Writes ``src/tokenomos/kernels/mxfp8/tables/<gpu>.json`` (the table ``ops``
routes by) and ``reports/gemm_tune_<gpu>.json`` (every time).

    python bench/profile/tune_gemm.py /models/lfm3-vl-2b-mxfp8
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from tokenomos import models
from tokenomos.core.engine import BUCKETS
from tokenomos.kernels.mxfp8 import ops as mx
from tokenomos.models.blocks.linear import Linear

ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda"
MS = sorted(set(BUCKETS) | {1, 2, 4, 768, 1024, 2048, 4096, 8192})
REPS = 20


def timed(fn) -> float:
    fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REPS):
            fn()
    return triton.testing.do_bench(g.replay, warmup=10, rep=100, return_mode="median") / REPS


def call(backend, q, s, wq, ws):
    if backend == "b12x":
        return lambda: mx.mm_mxfp8(q, wq.t(), s, ws, out_dtype=torch.bfloat16, backend="b12x")
    a, b = s.view(torch.float8_e8m0fnu), ws.view(torch.float8_e8m0fnu)
    return lambda: F.scaled_mm(
        q, wq.t(), a, mx.BLOCK, b, mx.BLOCK, mx.SWIZZLE, mx.SWIZZLE, output_dtype=torch.bfloat16
    )


def shapes(paths: list[str]) -> set[tuple[int, int]]:
    out = set()
    for p in paths:
        model = models.load(p, DEV, models.configure(p, head="mxfp8"))
        out |= {(m.n, m.k) for m in model.modules() if isinstance(m, Linear) and m.is_mx}
        del model
        torch.cuda.empty_cache()
    return out


def main() -> None:
    backends = ["cublas"] + (["b12x"] if mx.mm_mxfp8 is not None else [])
    table, times = {}, []
    for n, k in sorted(shapes(sys.argv[1:])):
        wq, ws = mx.quant(torch.randn(n, k, device=DEV, dtype=torch.bfloat16))
        winners = []
        for m in MS:
            q, s = mx.quant(torch.randn(m, k, device=DEV, dtype=torch.bfloat16))
            row = {"n": n, "k": k, "m": m}
            for b in backends:
                row[b] = round(timed(call(b, q, s, wq, ws)) * 1e3, 3)
            best = min(backends, key=lambda b: row[b])
            row["best"] = best
            winners.append((m, best))
            times.append(row)
            print(row)
        ranges = []
        for m, best in winners:
            if ranges and ranges[-1][1] == best:
                ranges[-1][0] = m
            else:
                ranges.append([m, best])
        if ranges[-1][1] == "cublas":
            ranges.pop()
        table[f"{n}x{k}"] = ranges
    slug = mx.gpu_slug()
    out = {"gpu": torch.cuda.get_device_name(), "shapes": table}
    mx.TABLES.mkdir(exist_ok=True)
    (mx.TABLES / f"{slug}.json").write_text(json.dumps(out, indent=1))
    rep = ROOT / "reports" / f"gemm_tune_{slug}.json"
    rep.parent.mkdir(exist_ok=True)
    rep.write_text(json.dumps({"us_per_call": times, **out}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
