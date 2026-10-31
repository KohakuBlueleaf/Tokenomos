"""HTTP handling alone: the real ``api.server`` handler and server class over a stub ``LLM`` whose
requests finish at once with a real reply text, so the only work is the server's. Real caption
bodies from client processes, one connection per request as ``serve_caption.py``. Serially: JSON
body parse, ``params``, reply encode. For each concurrency: requests/s and the server process's
CPU per request (all its threads). No GPU. Writes ``reports/http_front_<tag>.json``.

    python bench/http_front.py --rows ROWS --replies reports/caption_cap_lfm3d.json
"""

import argparse
import itertools
import json
import multiprocessing
import statistics
import threading
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from tokenomos.api.llm import Output
from tokenomos.api.router import raise_open_files
from tokenomos.api.server import params, serve, usage

ROOT = Path(__file__).resolve().parents[1]


class Stub:
    """``LLM`` as ``Handler`` uses it, every request done on submit."""

    failure = None

    def __init__(self, outs: list[Output]):
        self.outs = itertools.cycle(outs)

    def submit_chat(self, messages, p, **kw) -> Future:
        f: Future = Future()
        f.set_result(next(self.outs))
        return f

    def wait(self, futs: list[Future]) -> list[Output]:
        return [f.result() for f in futs]

    def output(self, out: Output) -> Output:
        return out

    def close(self) -> None:
        pass


def body_of(messages: list[dict]) -> bytes:
    return json.dumps(
        {
            "model": "stub",
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()


def read_rows(path: str, n: int) -> list[list[dict]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["messages"])
            if len(rows) == n:
                break
    return rows


def post(url: str, body: bytes) -> float:
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    return time.perf_counter() - t0


def client(conn, url: str, rows: str, n: int, part: int, parts: int) -> None:
    raise_open_files()
    bodies = [body_of(m) for m in read_rows(rows, n)[part::parts]]
    conn.send("ready")
    while True:
        threads = conn.recv()
        if threads is None:
            return
        with ThreadPoolExecutor(threads) as ex:
            conn.send(list(ex.map(lambda b: post(url, b), bodies)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--replies", required=True, help="a caption report: its texts and token ids")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--concurrency", default="1,16,64,256,1024")
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--port", type=int, default=18931)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    raise_open_files()
    report_in = json.loads(Path(args.replies).read_text("utf-8"))
    outs = [
        Output(text, ids, 1030, "stop")
        for text, ids in zip(report_in["texts"], report_in["token_ids"])
    ]
    bodies = [body_of(m) for m in read_rows(args.rows, args.n)]
    t_parse, t_params, t_reply = [], [], []
    for body, out in zip(bodies, itertools.cycle(outs)):
        t0 = time.perf_counter()
        d = json.loads(body)
        t1 = time.perf_counter()
        params(d, 1024)
        t2 = time.perf_counter()
        msg = {"role": "assistant", "content": out.text}
        choice = {"index": 0, "message": msg, "finish_reason": "stop"}
        reply = {"choices": [choice], "usage": usage(1030, len(out.token_ids))}
        json.dumps(reply, ensure_ascii=False).encode()
        t3 = time.perf_counter()
        t_parse.append(t1 - t0)
        t_params.append(t2 - t1)
        t_reply.append(t3 - t2)
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    ctx = multiprocessing.get_context("spawn")
    links = []
    for part in range(args.clients):
        here, there = ctx.Pipe()
        a = (there, url, args.rows, args.n, part, args.clients)
        ctx.Process(target=client, args=a, daemon=True).start()
        links.append(here)
    for c in links:
        c.recv()
    threading.Thread(
        target=serve, args=(Stub(outs), "127.0.0.1", args.port, "stub"), daemon=True
    ).start()
    time.sleep(1.0)
    levels = {}
    for level in map(int, args.concurrency.split(",")):
        used = links[: min(level, args.clients)]
        c0, t0 = time.process_time(), time.perf_counter()
        for c in used:
            c.send(max(1, level // len(used)))
        lat = [x for c in used for x in c.recv()]
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        levels[level] = {
            "requests": len(lat),
            "requests_per_s": round(len(lat) / wall, 1),
            "server_cpu_us_per_request": round(cpu / len(lat) * 1e6, 1),
            "latency_ms_p50": round(statistics.median(lat) * 1e3, 3),
        }
        print(level, levels[level], flush=True)
    for c in links:
        c.send(None)
    report = {
        "n": len(bodies),
        "body_kb_mean": round(statistics.mean(len(b) for b in bodies) / 1e3, 1),
        "parse_us_mean": round(statistics.mean(t_parse) * 1e6, 1),
        "params_us_mean": round(statistics.mean(t_params) * 1e6, 1),
        "reply_us_mean": round(statistics.mean(t_reply) * 1e6, 1),
        "levels": levels,
    }
    out = ROOT / "reports" / f"http_front_{args.tag}.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
