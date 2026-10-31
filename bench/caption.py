"""Root captions through ``tokenomos.LLM.chat``: the first ``N`` rows of a chat JSONL (``messages``
with the system prompt and the image as a data URL), greedy, thinking off. Images per second over
the whole call (image decode and resize in the prep pool, vision, prefill, decode), token counts,
and every output text. Writes ``reports/caption_<tag>.json``.

    python bench/caption.py --model M --rows calib_rows.jsonl --n 256
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.cuda import nvtx

from tap import Tap
from tokenomos import LLM, SamplingParams, models

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--slots", type=int, default=None, help="default: sized to the memory")
    ap.add_argument("--arch-kw", default='{"head": "mxfp8"}', help="JSON kwargs of the declaration")
    ap.add_argument("--engine-kw", default="{}", help="JSON kwargs of the engine")
    ap.add_argument("--prep-workers", type=int, default=16)
    ap.add_argument("--prefix-entries", type=int, default=None, help="0: no prefix table")
    ap.add_argument("--tap", action="store_true", help="per-stage GPU and host seconds (tap.py)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rows = []
    with open(args.rows, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["messages"])
            if len(rows) == args.n:
                break
    t0 = time.perf_counter()
    engine_kw = {"max_len": 8192, **json.loads(args.engine_kw)}
    llm = LLM(
        args.model,
        arch=models.configure(args.model, **json.loads(args.arch_kw)),
        prep_workers=args.prep_workers,
        slots=args.slots,
        new_cap=args.max_tokens,
        prefix_entries=args.prefix_entries,
        **engine_kw,
    )
    init_s = time.perf_counter() - t0
    params = SamplingParams(max_tokens=args.max_tokens)
    # a failed engine still closes the prep pool, so the process exits
    with llm:
        tap = None
        if args.tap:
            tap = Tap(llm.engine, llm._thread.ident)
            tap.wrap(llm, "request")
        llm.chat(rows[:4], params, enable_thinking=False)
        llm.engine.stats = type(llm.engine.stats)()
        if tap:
            tap.start(len(rows))
        t0 = time.perf_counter()
        nvtx.range_push("bench")
        outs = llm.chat(rows, params, enable_thinking=False)
        nvtx.range_pop()
        wall = time.perf_counter() - t0
        if tap:
            tap.stop()
        llm.engine.collect_spans()
        st = llm.engine.stats
        prefix = dict(llm.engine.prefix.stats) if llm.engine.prefix else None
    gen = sum(len(o.token_ids) for o in outs)
    report = {
        "model": args.model,
        "arch_kw": json.loads(args.arch_kw),
        "engine_kw": engine_kw,
        "n": len(rows),
        "slots": llm.engine.slots,
        "pages": llm.engine.st.pages,
        "init_s": round(init_s, 1),
        "wall_s": round(wall, 3),
        "images_per_s": round(len(rows) / wall, 2),
        "prompt_tokens_mean": round(sum(o.prompt_tokens for o in outs) / len(outs), 1),
        "gen_tokens_mean": round(gen / len(outs), 1),
        "gen_tok_s": round(gen / wall, 1),
        "finish": {k: sum(o.finish == k for o in outs) for k in ("stop", "length")},
        "prefill_tokens": st.prefill_tokens,
        "prefill_forwards": st.prefill_forwards,
        "prefix": prefix,
        "decode_steps": st.steps,
        "pauses": st.pauses,
        "buckets": st.buckets,
        "admit_host_s": round(st.admit_host_s, 3),
        "encode_gpu_s": round(st.encode_gpu_s, 3),
        "prefill_gpu_s": round(st.prefill_gpu_s, 3),
        "loop_host_s": round(st.host_s, 3),
        "wait_s": round(st.wait_s, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / (1 << 30), 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / (1 << 30), 2),
        "tap": tap.report(st.stamps) if tap else None,
        "texts": [o.text for o in outs],
        "token_ids": [o.token_ids for o in outs],
    }
    tag = args.tag or Path(args.model).name
    path = ROOT / "reports" / f"caption_{tag}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), "utf-8")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("texts", "token_ids")}, indent=1)
    )
    print(outs[0].text[:600])


if __name__ == "__main__":
    main()
