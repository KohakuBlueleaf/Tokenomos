"""GDN chunked prefill: kernel microseconds and error of each backend on one card, same inputs.

Each backend against a float32 token recurrence, S sequences of T tokens with initial states. The
``fused`` kernels sweep ``(BT, BV, warps, stages)``; a tile that does not compile is a row with its
error. ``--write`` keeps the fastest tile within twice the sweep's best error on every shape and
dtype as this card's ``kernels/gdn/tables/<gpu>.json``. Writes ``reports/gdn_kernel_<tag>.json``.

    python bench/profile/gdn_kernel.py --tag 3090 --write
"""

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tokenomos.kernels.gdn import prefill as gp
from tokenomos.kernels.gdn.fla import FLA_CHUNK_SIZE
from tokenomos.kernels.gdn.fla import chunk_gated_delta_rule as fla_rule
from tokenomos.kernels.gdn.fla.index import chunk_indices_host
from tokenomos.kernels.mxfp8.ops import gpu_slug

try:
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_rule
except ImportError:
    fi_rule = None

ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda"
# Hk, Hv, K, V
SHAPES = {"4B": (16, 32, 128, 128), "27B": (16, 48, 128, 128)}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
ROWS, WARPS, STAGES = (32, 64), (2, 4), (2, 3)


def make(s, t, hk, hv, kd, vd, dtype):
    g = torch.Generator(DEV).manual_seed(0)
    n = s * t

    def r(*shape):
        return torch.randn(*shape, device=DEV, generator=g)

    q = F.normalize(r(n, hk, kd), dim=-1).to(dtype)
    k = F.normalize(r(n, hk, kd), dim=-1).to(dtype)
    v = r(n, hv, vd).to(dtype)
    return q, k, v, -F.softplus(r(n, hv) - 2.0), torch.sigmoid(r(n, hv)), r(s, hv, vd, kd) * 0.1


def reference(q, k, v, gate, beta, init, cu, kd):
    hk, hv = q.shape[1], v.shape[1]
    r = hv // hk
    o = torch.empty_like(v, dtype=torch.float32)
    finals = []
    for i in range(len(cu) - 1):
        h = init[i].clone()
        for t in range(cu[i], cu[i + 1]):
            ki = k[t].float().repeat_interleave(r, 0)
            qi = q[t].float().repeat_interleave(r, 0) * kd**-0.5
            h = h * gate[t].exp()[:, None, None]
            d = (v[t].float() - (h * ki[:, None]).sum(-1)) * beta[t][:, None]
            h = h + d[..., None] * ki[:, None]
            o[t] = (h * qi[:, None]).sum(-1)
        finals.append(h)
    return o, torch.stack(finals)


def err(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


def timed(fn, reps):
    """(median us, fastest us) of ``fn`` between syncs, past two warm calls."""
    fn()
    fn()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts), min(ts)


def stages(q, k, v, gate, beta, init, cu, chunks, starts, scale, tile):
    """Median us of ``gram``, ``invert`` and ``advance`` alone."""
    bt, *rest = tile
    gc, A, W = gp.gram(q, k, beta, gate, cu, chunks, bt)
    Ti = gp.invert(A, q.dtype)
    out = torch.empty_like(v)
    return {
        "gram_us": round(timed(lambda: gp.gram(q, k, beta, gate, cu, chunks, bt), 10)[0], 1),
        "invert_us": round(timed(lambda: gp.invert(A, q.dtype), 10)[0], 1),
        "advance_us": round(
            timed(
                lambda: gp.advance(q, k, v, gc, beta, Ti, W, out, init, cu, starts, scale, rest), 10
            )[0],
            1,
        ),
    }


def sweep(name, shape, dn, dtype, args, rows, best):
    """Every backend on one shape and dtype: appends to ``rows``, each tile's rows to ``best``."""
    hk, hv, kd, vd = shape
    q, k, v, gate, beta, init = make(args.seqs, args.tokens, hk, hv, kd, vd, dtype)
    n, scale = q.shape[0], kd**-0.5
    cu = [args.tokens * i for i in range(args.seqs + 1)]
    cu_t = torch.tensor(cu, dtype=torch.int32, device=DEV)
    ro, rf = reference(q, k, v, gate, beta, init, cu, kd)
    rows_fla = chunk_indices_host(cu, FLA_CHUNK_SIZE)
    chunks = torch.tensor(rows_fla, dtype=torch.int32, device=DEV).view(-1, 2)

    def fla():
        o, f = fla_rule(
            q[None],
            k[None],
            v[None],
            gate[None],
            beta[None],
            scale=scale,
            initial_state=init,
            output_final_state=True,
            cu_seqlens=cu_t,
            chunk_indices=chunks,
        )
        return o.view(n, hv, vd), f

    def flashinfer():
        o, f = torch.empty_like(v), torch.empty_like(init)
        fi_rule(
            q,
            k,
            v,
            gate.exp(),
            beta,
            scale=scale,
            initial_state=init,
            output_final_state=True,
            cu_seqlens=cu_t,
            output=o,
            output_state=f,
        )
        return o, f

    def fused(table, starts, tile):
        return lambda: gp.chunk_gated_delta_rule(
            q, k, v, gate, beta, scale, init, cu_t, table, starts, tile=tile
        )

    def record(backend, fn, tile=None):
        row = {"shape": name, "dtype": dn, "backend": backend, "tile": tile}
        try:
            o, f = fn()
            med, low = timed(fn, args.reps)
        except Exception as e:  # noqa: BLE001 - a tile that fails is a result
            row["error"] = str(e)[:200]
        else:
            row |= {"us": round(med, 1), "us_min": round(low, 1)}
            row["tok_s"] = round(n / med * 1e6, 1)
            row["out_err"] = round(err(o, ro), 6)
            row["state_err"] = round(err(f, rf), 6)
        rows.append(row)
        print(json.dumps(row), flush=True)
        return row

    record("fla", fla)
    if fi_rule is not None and torch.cuda.get_device_capability()[0] >= 9:
        record("flashinfer", flashinfer)
    for tile in itertools.product(gp.BT_SIZES, ROWS, WARPS, STAGES):
        rows_bt, starts_bt = gp.table(cu, tile[0])
        table = torch.tensor(rows_bt, dtype=torch.int32, device=DEV).view(-1, 2)
        starts = torch.tensor(starts_bt, dtype=torch.int32, device=DEV)
        row = record("fused", fused(table, starts, tile), tile)
        if "us" in row:
            row |= stages(q, k, v, gate, beta, init, cu_t, table, starts, scale, tile)
            best.setdefault(tile, []).append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--write", action="store_true", help="keep the best tile as this card's table")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    rows, best = [], {}
    with torch.no_grad():
        for (name, shape), (dn, dtype) in itertools.product(SHAPES.items(), DTYPES.items()):
            if dtype != torch.bfloat16 or torch.cuda.get_device_capability()[0] >= 8:
                sweep(name, shape, dn, dtype, args, rows, best)
    # a tile counts when it ran every shape and dtype
    most = max(len(rs) for rs in best.values())
    tiles = {t: rs for t, rs in best.items() if len(rs) == most}
    floor = {}
    for r in (r for rs in tiles.values() for r in rs):
        key = r["shape"], r["dtype"]
        floor[key] = min(floor.get(key, r["out_err"]), r["out_err"])
    sound = {
        t: rs
        for t, rs in tiles.items()
        if all(r["out_err"] <= 2 * floor[r["shape"], r["dtype"]] for r in rs)
    }
    pick = min(sound, key=lambda t: sum(r["us"] for r in sound[t]))
    report = {
        "gpu": gpu_slug(),
        "seqs": args.seqs,
        "tokens": args.tokens,
        "pick": pick,
        "rows": rows,
    }
    out = ROOT / "reports" / f"gdn_kernel_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("PICK", pick, flush=True)
    if args.write:
        path = gp.TABLES / f"{gpu_slug()}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"config": pick, "report": out.name}) + "\n")


if __name__ == "__main__":
    main()
