"""Decode throughput: ``N`` requests of ``P`` prompt tokens, ``G`` generated each (no stop).

Steady state is the stretch running the most rows at once (every request when the pages hold
them all; ``steady_rows``): the step time is the median gap
between consecutive decode-step enqueues there (the host runs one step ahead, so when the GPU is
the bottleneck the enqueue rate is the GPU's rate). Also: host seconds per step (the loop's own
work, waits excluded), the host's wait share, bucket use. Writes ``reports/decode_<tag>.json``.

    python bench/decode.py --model /models/lfm3-vl-2b-mxfp8 --n 256
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenomos._rs import grammar as rs_grammar

from tap import Tap
from tokenomos import models
from tokenomos.api.tokens import token_bytes
from tokenomos.core.engine import Engine, Request
from tokenomos.core.grammar import DeviceGrammar

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--page", type=int, default=32)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--no-warmup", action="store_true", help="skip the engine warmup")
    ap.add_argument("--arch-kw", default='{"head": "bf16"}', help="JSON kwargs of the declaration")
    ap.add_argument("--engine-kw", default="{}", help="JSON kwargs of the engine")
    ap.add_argument(
        "--profile",
        default="off",
        choices=["off", "decode", "prefill"],
        help="cudaProfilerStart / Stop around the timed decode or the timed prefill",
    )
    ap.add_argument("--prefill", type=int, default=2048, help="prefill phase prompt length; 0: off")
    ap.add_argument("--prefill-n", type=int, default=0, help="prefill phase requests; 0: --n")
    ap.add_argument(
        "--grammar",
        default="off",
        choices=["off", "idle", "on"],
        help="engine without grammar / with grammar but free-text rows / every row on --schema",
    )
    ap.add_argument("--schema", default="", help="JSON schema file for --grammar on")
    ap.add_argument("--tap", action="store_true", help="per-stage GPU and host seconds (tap.py)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(0)
    arch_kw = json.loads(args.arch_kw)
    model = models.load(args.model, "cuda", models.configure(args.model, **arch_kw))
    vocab = model.embedding.shape[0]
    dg = None
    if args.grammar != "off":
        path = Path(args.model)
        tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        gen = json.loads((path / "generation_config.json").read_text("utf-8"))
        eos = gen.get("eos_token_id", [])
        eos = eos if isinstance(eos, list) else [eos]
        dg = DeviceGrammar(rs_grammar.Store(token_bytes(tok, vocab), eos, vocab, 8), "cuda")
    t0 = time.perf_counter()
    eng = Engine(
        model,
        slots=args.slots or args.n,
        page=args.page,
        max_len=max(args.prompt + args.gen, args.prefill + 1) + 8,
        new_cap=args.gen,
        graphs=not args.eager,
        grammar=dg,
        **json.loads(args.engine_kw),
    )
    init_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    if not args.no_warmup:
        eng.warmup()
    warmup_s = time.perf_counter() - t0
    g = dg.compile(Path(args.schema).read_text("utf-8")) if args.grammar == "on" else None
    reqs = [
        Request(
            torch.randint(1000, vocab - 1000, (args.prompt,)).tolist(), max_new=args.gen, grammar=g
        )
        for _ in range(args.n)
    ]
    tap = Tap(eng) if args.tap else None
    eng.generate(reqs[:8])
    eng.stats = type(eng.stats)()
    if tap:
        tap.start()
    if args.profile == "decode":
        torch.cuda.cudart().cudaProfilerStart()
    w0 = time.perf_counter()
    results = eng.generate(reqs)
    torch.cuda.synchronize()
    decode_wall = time.perf_counter() - w0
    if args.profile == "decode":
        torch.cuda.cudart().cudaProfilerStop()
    st = eng.stats
    if tap:
        tap.stop()
    stamps = [t for t, _, _ in st.stamps]
    # steady: the steps running the most rows at once (all N when the pages hold them)
    rows = max(r for _, _, r in st.stamps)
    full = [i for i, (_, _, r) in enumerate(st.stamps) if r == rows]
    lo, hi = (full[0] + 4, full[-1] - 4) if len(full) > 16 else (1, len(stamps) - 1)
    gaps = [stamps[i + 1] - stamps[i] for i in range(lo, hi) if st.stamps[i][2] == rows]
    step_ms = statistics.median(gaps) * 1e3 if gaps else float("nan")
    out = {
        "model": args.model,
        "n": args.n,
        "prompt": args.prompt,
        "gen": args.gen,
        "graphs": not args.eager,
        "arch_kw": arch_kw,
        "pages": eng.st.pages,
        "init_s": round(init_s, 2),
        "warmup_s": round(warmup_s, 2),
        "steps": st.steps,
        "steady_steps": len(gaps),
        "step_ms_median": round(step_ms, 3),
        "step_ms_p90": round(sorted(gaps)[int(0.9 * len(gaps))] * 1e3, 3) if gaps else None,
        "steady_rows": rows,
        "decode_tok_s": round(rows / step_ms * 1e3, 1) if gaps else None,
        # every generated token over the whole call (prompt prefill and ramp included), as
        # vLLM's own number is taken
        "decode_tok_s_wall": round(sum(len(r.tokens) for r in results) / decode_wall, 1),
        "host_ms_per_step": round(st.host_s / max(st.steps, 1) * 1e3, 3),
        "wait_share": round(st.wait_s / max(st.wall_s, 1e-9), 3),
        "wall_s": round(st.wall_s, 3),
        "admit_host_s": round(st.admit_host_s, 3),
        "slowest_steps_ms": sorted(
            (round((b[0] - a[0]) * 1e3, 1), i)
            for i, (a, b) in enumerate(zip(st.stamps, st.stamps[1:]))
        )[:-13:-1],
        "gen_tokens": sum(len(r.tokens) for r in results),
        "prefill_tokens": st.prefill_tokens,
        "buckets": st.buckets,
        "grammar": args.grammar,
        "grammar_misses": st.grammar_misses,
    }
    if tap:
        out["tap"] = tap.report(st.stamps)
    if args.prefill:
        count = args.prefill_n or args.n
        long = [
            Request(torch.randint(1000, vocab - 1000, (args.prefill,)).tolist(), max_new=1)
            for _ in range(count)
        ]
        # the same per-request shapes once, so kernel compiles stay out of the timing
        eng.generate(long[: min(8, count)])
        eng.stats = type(eng.stats)()
        torch.cuda.synchronize()
        if tap:
            tap.start()
        if args.profile == "prefill":
            torch.cuda.cudart().cudaProfilerStart()
        t0 = time.perf_counter()
        eng.generate(long)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        if args.profile == "prefill":
            torch.cuda.cudart().cudaProfilerStop()
        out["prefill_len"] = args.prefill
        out["prefill_s"] = round(wall, 3)
        out["prefill_n"] = count
        out["prefill_tok_s"] = round(count * args.prefill / wall, 1)
        out["prefill_host_s"] = round(eng.stats.host_s, 3)
        if tap:
            tap.stop()
            out["tap_prefill"] = tap.report(eng.stats.stamps)
    tag = args.tag or Path(args.model).name + f"_n{args.n}" + ("_eager" * args.eager)
    tag += f"_grammar-{args.grammar}" * (args.grammar != "off")
    path = ROOT / "reports" / f"decode_{tag}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
