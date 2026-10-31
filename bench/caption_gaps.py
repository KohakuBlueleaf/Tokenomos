"""Are greedy caption outputs faithful to the model? Each run's generated tokens (``token_ids`` of
caption reports: ours, vLLM's) teacher-forced through the dense single-sequence reference of
``--ref`` (default the served checkpoint; the bf16 one for a precision ceiling): per token, the
gap ``max log p - log p(chosen)``. A faithful greedy run picks the argmax or a near-tie
(``--tie``); a bug (wrong image rows, stale state, bad mask) shows as large gaps. Also, for each
pair of runs, the gap of both candidates at their first divergence. Writes
``reports/caption_gaps_<tag>.json``.

    python bench/caption_gaps.py --model M --rows ROWS --run ours=A.json --run vllm=B.json --n 64
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import torch

from tokenomos import models
from tokenomos.api.prep import worker
from tokenomos.models import reference

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda"


def logprobs(model, ids: list[int], images: list[torch.Tensor]) -> torch.Tensor:
    """Teacher-forced log-probs [len(ids), vocab] with the image features on placeholder rows."""
    return reference.multimodal_logits(model, ids, images, DEV).float().log_softmax(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", default="", help="reference checkpoint (default --model)")
    ap.add_argument("--ref-kw", default="{}", help="JSON kwargs of the reference declaration")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--run", action="append", required=True, help="NAME=caption report JSON")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tie", type=float, default=0.15)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ref_path = args.ref or args.model
    model = models.load(ref_path, DEV, models.configure(ref_path, **json.loads(args.ref_kw)))
    worker.init(args.model, model.processor, model.image_token, None, 8)
    runs = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        runs[name] = json.loads(Path(path).read_text("utf-8"))["token_ids"]
    rows = []
    with open(args.rows, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["messages"])
            if len(rows) == args.n:
                break
    gaps = {k: [] for k in runs}
    worst = {k: [] for k in runs}
    pair = {f"{a}|{b}": [] for a, b in combinations(runs, 2)}
    for i, conv in enumerate(rows):
        prompt, images, _ = worker.job((conv, {"enable_thinking": False}))
        for name, toks in runs.items():
            gen = toks[i]
            lp = logprobs(model, prompt + gen[:-1], images)[len(prompt) - 1 :]
            g = lp.max(-1).values - lp[torch.arange(len(gen)), torch.tensor(gen, device=DEV)]
            gaps[name].append(g.cpu())
            j = int(g.argmax())
            worst[name].append((round(float(g[j]), 3), i, j))
        for a, b in combinations(runs, 2):
            x, y = runs[a][i], runs[b][i]
            d = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), None)
            if d is None:
                continue
            lp = logprobs(model, prompt + x[:d], images)[-1]
            top = float(lp.max())
            pair[f"{a}|{b}"].append(
                {
                    "row": i,
                    "at": d,
                    "gap_a": round(top - float(lp[x[d]]), 4),
                    "gap_b": round(top - float(lp[y[d]]), 4),
                }
            )
    report = {"ref": ref_path, "n": len(rows), "tie": args.tie, "runs": {}, "divergence": {}}
    for name, gs in gaps.items():
        g = torch.cat(gs)
        report["runs"][name] = {
            "tokens": g.numel(),
            "argmax_agreement": round(float((g == 0).float().mean()), 5),
            "gap_mean": round(float(g.mean()), 5),
            "gap_max": round(float(g.max()), 4),
            "over_tie": int((g > args.tie).sum()),
            "over_1": int((g > 1.0).sum()),
            "worst": sorted(worst[name], reverse=True)[:8],
        }
    for key, ds in pair.items():
        both = [max(d["gap_a"], d["gap_b"]) for d in ds]
        report["divergence"][key] = {
            "diverged": len(ds),
            "max_gap_at_divergence_max": round(max(both), 4) if both else None,
            "within_tie": sum(v <= args.tie for v in both),
            "rows": ds,
        }
    out = ROOT / "reports" / f"caption_gaps_{args.tag or Path(ref_path).name}.json"
    out.write_text(json.dumps(report, indent=1))
    brief = {k: v for k, v in report["runs"].items()}
    div = {
        k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in report["divergence"].items()
    }
    print(json.dumps({"runs": brief, "divergence": div}, indent=1))


if __name__ == "__main__":
    main()
