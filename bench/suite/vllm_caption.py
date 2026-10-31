"""Root captions through vLLM's in-process ``LLM``, the arguments of the ``vllm serve`` runs: the
first ``N`` rows of a chat JSONL, greedy, thinking off. ``--seqs`` lists ``max_num_seqs`` to try in
turn: a card that holds too few Mamba cache blocks refuses the first. Writes ``--out``.

    python bench/suite/vllm_caption.py --model M --rows ROWS --out reports/vllm_caption_x.json
"""

import argparse
import json
import time

from vllm import LLM


def build(model: str, seqs: list[int]) -> tuple[LLM, int]:
    for i, n in enumerate(seqs):
        try:
            llm = LLM(
                model=model,
                max_model_len=4096,
                max_num_seqs=n,
                gpu_memory_utilization=0.90,
                mamba_ssm_cache_dtype="bfloat16",
                trust_remote_code=True,
            )
        except (RuntimeError, ValueError):
            if i == len(seqs) - 1:
                raise
            time.sleep(10)
            continue
        return llm, n
    raise ValueError("no --seqs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seqs", default="256,192,128")
    args = ap.parse_args()
    rows = []
    with open(args.rows, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["messages"])
            if len(rows) == args.n:
                break
    llm, seqs = build(args.model, [int(s) for s in args.seqs.split(",")])
    # the checkpoint's generation_config, as ``vllm serve`` applies it (LFM3: repetition_penalty)
    params = llm.get_default_sampling_params()
    params.temperature, params.max_tokens = 0.0, 1024
    kw = {"chat_template_kwargs": {"enable_thinking": False}}
    llm.chat(rows[:4], params, **kw)
    t0 = time.perf_counter()
    outs = llm.chat(rows, params, **kw)
    wall = time.perf_counter() - t0
    gen = sum(len(o.outputs[0].token_ids) for o in outs)
    report = {
        "model": args.model,
        "n": len(rows),
        "max_num_seqs": seqs,
        "wall_s": round(wall, 3),
        "images_per_s": round(len(rows) / wall, 2),
        "prompt_tokens_mean": round(sum(len(o.prompt_token_ids) for o in outs) / len(outs), 1),
        "gen_tokens_mean": round(gen / len(outs), 1),
        "gen_tok_s": round(gen / wall, 1),
        "finish": {
            k: sum(o.outputs[0].finish_reason == k for o in outs) for k in ("stop", "length")
        },
        "texts": [o.outputs[0].text for o in outs],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "texts"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
