"""Root captions through an OpenAI-compatible server (Tokenomos's or vLLM's): the first ``N`` rows
of a chat JSONL as concurrent ``/v1/chat/completions`` requests (greedy, thinking off), images per
second over the whole batch, token counts from ``usage``, every text. Writes
``reports/serve_caption_<tag>.json``.

    python bench/serve_caption.py --url http://127.0.0.1:8000 --rows ROWS --n 256
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import resource
except ImportError:  # Windows has no per-process open-file limit to raise
    resource = None

ROOT = Path(__file__).resolve().parents[1]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if resource is not None:
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    rows = []
    with open(args.rows, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["messages"])
            if len(rows) == args.n:
                break
    model = json.loads(urllib.request.urlopen(f"{args.url}/v1/models").read())["data"][0]["id"]

    def one(messages):
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        return post(f"{args.url}/v1/chat/completions", body)

    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(one, rows[:4]))
        t0 = time.perf_counter()
        outs = list(ex.map(one, rows))
        wall = time.perf_counter() - t0
    gen = sum(o["usage"]["completion_tokens"] for o in outs)
    report = {
        "url": args.url,
        "model": model,
        "n": len(rows),
        "wall_s": round(wall, 3),
        "images_per_s": round(len(rows) / wall, 2),
        "prompt_tokens_mean": round(sum(o["usage"]["prompt_tokens"] for o in outs) / len(outs), 1),
        "gen_tokens_mean": round(gen / len(outs), 1),
        "gen_tok_s": round(gen / wall, 1),
        "finish": {
            k: sum(o["choices"][0]["finish_reason"] == k for o in outs) for k in ("stop", "length")
        },
        "texts": [o["choices"][0]["message"]["content"] for o in outs],
    }
    path = ROOT / "reports" / f"serve_caption_{args.tag or 'run'}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), "utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "texts"}, indent=1))


if __name__ == "__main__":
    main()
