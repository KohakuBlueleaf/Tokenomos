"""Grammar on the device, verified and measured: the first ``N`` caption chats of a JSONL, each
reply constrained by one GBC schema (round-robin over ``--schemas``), greedy.

Checks: every reply parses and validates against its schema (``jsonschema``); zero mask
disagreements: each output token's device state (``Output.states``) must name the mask the
standalone Rust matcher gives after the same tokens, and no transition may be rejected. Measured:
images per second with and without grammar on the same rows, grammar misses (each paused its
row one step), host seconds resolving them, table entries, states, masks. Writes
``reports/grammar_<tag>.json``.

    python bench/grammar.py --model M --rows ROWS --schemas DIR --n 256
"""

import argparse
import json
import time
from pathlib import Path

import jsonschema

from tokenomos import LLM, SamplingParams
from tokenomos.models.lfm3.arch import Lfm3

ROOT = Path(__file__).resolve().parents[1]


def disagreements(store, out, grammar: int) -> tuple[int, int]:
    """(steps checked, steps whose device mask differs from the standalone matcher's)."""
    h = store.matcher(grammar)
    bad = 0
    masks = store.d_masks(out.states)
    for t, m in zip(out.token_ids, masks):
        want = store.step([h], [[]])[0]
        bad += int(want != m)
        store.step([h], [[t]])
    store.release(h)
    return len(masks), bad


def phase_report(st, g, before, n: int, wall: float) -> dict:
    """One run's numbers; ``before``: the store's stats and grammar timing at its start."""
    before, timing, kinds = before
    after = g.store.stats()
    return {
        **{k: round(v - timing.get(k, 0.0), 4) for k, v in g.timing.items()},
        **{f"misses_{k}": v - kinds[k] for k, v in g.miss_kinds.items()},
        "n": n,
        "images_per_s": round(n / wall, 2),
        "decode_steps": st.steps,
        "grammar_misses": st.grammar_misses,
        "grammar_host_s": round(st.grammar_s, 4),
        "masks_built": after[0] - before[0],
        "mask_build_s": round(after[1] - before[1], 4),
        "table_entries_total": g.entries,
        "states_total": g.n_states,
        "device_masks_total": g.n_masks,
    }


def run(llm, rows, schemas, max_tokens):
    convs, params = [], []
    for i, conv in enumerate(rows):
        name, schema = schemas[i % len(schemas)]
        convs.append(conv)
        params.append((name, SamplingParams(max_tokens=max_tokens, json_schema=schema)))
    t0 = time.perf_counter()
    futs = [llm.submit_chat(c, p, enable_thinking=False) for c, (_, p) in zip(convs, params)]
    outs = [llm.output(r) for r in llm.wait(futs)]
    return outs, [n for n, _ in params], time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--schemas", required=True, help="folder of JSON schemas")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--warm", type=int, default=512, help="rows run between cold and warm")
    ap.add_argument("--slots", type=int, default=256)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rows = []
    with open(args.rows, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["messages"])
            if len(rows) == 2 * args.n + args.warm:
                break
    cold_rows, warm_rows = rows[: args.n], rows[args.n + args.warm :]
    schemas = [
        (p.stem, json.loads(p.read_text("utf-8")))
        for p in sorted(Path(args.schemas).glob("*.json"))
    ]
    llm = LLM(
        args.model, arch=Lfm3(head="mxfp8"), slots=args.slots, max_len=8192, new_cap=args.max_tokens
    )
    eng, g = llm.engine, llm.engine.grammar
    plain = SamplingParams(max_tokens=args.max_tokens)
    llm.chat(rows[:4], plain, enable_thinking=False)
    t0 = time.perf_counter()
    llm.chat(warm_rows, plain, enable_thinking=False)
    free_s = time.perf_counter() - t0
    phases, outs, names = {}, [], []
    for phase, part in (("cold", cold_rows), ("between", rows[args.n : args.n + args.warm])):
        before = g.store.stats(), dict(g.timing), dict(g.miss_kinds)
        eng.stats = type(eng.stats)()
        o, nm, wall = run(llm, part, schemas, args.max_tokens)
        phases[phase] = phase_report(eng.stats, g, before, len(o), wall)
        outs, names = outs + o, names + nm
    before = g.store.stats(), dict(g.timing), dict(g.miss_kinds)
    eng.stats = type(eng.stats)()
    o, nm, wall = run(llm, warm_rows, schemas, args.max_tokens)
    phases["warm"] = phase_report(eng.stats, g, before, len(o), wall)
    outs, names = outs + o, names + nm
    llm.close()
    per, checked, bad, invalid, cut, examples = {}, 0, 0, 0, 0, []
    for out, name in zip(outs, names):
        schema = dict(schemas)[name]
        c, b = disagreements(g.store, out, g.compiled[json.dumps(schema)])
        checked, bad = checked + c, bad + b
        ok = True
        try:
            jsonschema.validate(json.loads(out.text), schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            ok = False
            if len(examples) < 4:
                examples.append({"schema": name, "error": str(e)[:200], "text": out.text[:300]})
        # a reply cut by max_tokens is unfinished JSON; only finished replies must validate
        invalid += not ok and out.finish == "stop"
        cut += out.finish == "length"
        p = per.setdefault(name, {"n": 0, "valid": 0, "finish_length": 0, "tokens": 0})
        p["n"] += 1
        p["valid"] += ok
        p["finish_length"] += out.finish == "length"
        p["tokens"] += len(out.token_ids)
    stats = g.store.stats()
    report = {
        "replies": len(outs),
        "images_per_s_no_grammar": round(len(warm_rows) / free_s, 2),
        "phases": phases,
        "mask_steps_checked": checked,
        "mask_disagreements": bad,
        "rejected_transitions": stats[5],
        "invalid_finished_replies": invalid,
        "cut_by_max_tokens": cut,
        "table_entries": g.entries,
        "states": g.n_states,
        "masks": g.n_masks,
        "per_schema": per,
        "invalid_examples": examples,
        "texts": [o.text for o in outs],
    }
    out = ROOT / "reports" / f"grammar_{args.tag or 'run'}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), "utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "texts"}, indent=1))


if __name__ == "__main__":
    main()
