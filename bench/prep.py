"""Host preparation cost per stage, no GPU, each timed serially per row on one thread: the task's
pickle round trip (the conversation to a worker), chat-template render, tokenize, base64 and image
decode (data URL -> PIL, loaded), resize (the model processor), placeholder expansion, the shared
prefix (the head re-rendered and re-tokenized), the ring write and the slot's return, the result's
pickle round trip, the RoPE table or image runs the serving process builds at submit, and
``worker.job`` whole as a check of the stages' sum (its ids and prefix must equal theirs). Then, for
each worker count, prep-pool throughput of whole conversations (``worker.job`` into an unpinned
``ImageRing``, the slots returned as results arrive) with the serving process's CPU per row, and
the same payloads through a pool whose job does nothing (IPC alone). Writes
``reports/prep_<tag>.json``.

    python bench/prep.py --model M --rows calib_rows.jsonl --n 1024
"""

import argparse
import base64
import importlib
import io
import json
import multiprocessing
import pickle
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from PIL import Image
from tokenizers import Tokenizer

from tokenomos import models
from tokenomos.api.prep import worker
from tokenomos.api.prep.ring import ImageRing
from tokenomos.api.template import ChatTemplate
from tokenomos.models import imageprep
from tokenomos.models.load.checkpoint import Checkpoint

ROOT = Path(__file__).resolve().parents[1]
KW = {"enable_thinking": False}


def ms(xs):
    return {
        "mean": round(statistics.mean(xs) * 1e3, 3),
        "p50": round(statistics.median(xs) * 1e3, 3),
        "max": round(max(xs) * 1e3, 3),
        "total": round(sum(xs) * 1e3, 1),
    }


def echo(task, reply):
    """A pool job without the work: what the real job's IPC carries, both ways."""
    return reply


def pool_rate(ctx, w, jobs, ring=None):
    """Rows/s through ``w`` workers and the serving process's CPU per row; ``jobs``: the pool
    calls, after every worker started."""
    init = (*ring[0], ring[1].args(), 4) if ring else None
    kw = {"initializer": worker.start, "initargs": init} if ring else {}
    with ProcessPoolExecutor(w, mp_context=ctx, **kw) as pool:
        list(pool.map(worker.ping, range(w * 4)))
        c0, t0 = time.process_time(), time.perf_counter()
        for f in as_completed([pool.submit(*j) for j in jobs]):
            if ring:
                ring[1].release(s for s, _ in f.result()[1])
            else:
                f.result()
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    return {
        "wall_s": round(wall, 3),
        "rows_per_s": round(len(jobs) / wall, 1),
        "parent_cpu_us_per_row": round(cpu / len(jobs) * 1e6, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--workers", default="8,16,32,64")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rows = []
    with open(args.rows, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["messages"])
            if len(rows) == args.n:
                break
    path = Path(args.model)
    ckpt = Checkpoint(str(path))
    family = type(models.ARCHS[ckpt.architecture]).__module__.rsplit(".", 1)[0]
    name, token = f"{family}.processor", ckpt.config["image_token_id"]
    runs = ckpt.text_config.get("use_bidirectional_attention") == "vision"
    tmpl, tok = ChatTemplate(path), Tokenizer.from_file(str(path / "tokenizer.json"))
    proc = importlib.import_module(name)
    rope, wrap = getattr(proc, "rope", None), getattr(proc, "wrap", None)
    torch.set_num_threads(1)
    ctx = multiprocessing.get_context("spawn")
    ring = ImageRing(8, proc.MAX_BYTES, ctx.Queue(), pin=False)
    worker.init(str(path), name, token, ring.args(), 4)
    keys = "task render tokenize b64 image resize expand prefix ring slot result submit job"
    t = {k: [] for k in keys.split()}
    srcs, sizes, out_bytes, formats, prompt = [], [], [], {}, []
    for conv in rows:
        task = (conv, KW)
        t0 = time.perf_counter()
        pickle.loads(pickle.dumps(task))
        t["task"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        text, s = tmpl.render(conv, **KW)
        t1 = time.perf_counter()
        ids = tok.encode(text, add_special_tokens=False).ids
        t2 = time.perf_counter()
        t["render"].append(t1 - t0)
        t["tokenize"].append(t2 - t1)
        images = []
        for src in s:
            srcs.append(src)
            t0 = time.perf_counter()
            raw = base64.b64decode(src.split(",", 1)[1])
            t1 = time.perf_counter()
            im = Image.open(io.BytesIO(raw))
            im.load()
            t2 = time.perf_counter()
            x = proc.prepare(im)
            t3 = time.perf_counter()
            t["b64"].append(t1 - t0)
            t["image"].append(t2 - t1)
            t["resize"].append(t3 - t2)
            sizes.append(im.size)
            out_bytes.append(x.numel())
            formats[im.format] = formats.get(im.format, 0) + 1
            images.append(x)
        t0 = time.perf_counter()
        ids = worker.expand(ids, [proc.tokens(im) for im in images], token, wrap)
        t1 = time.perf_counter()
        prefix = worker.shared_prefix(conv, ids, KW)
        t2 = time.perf_counter()
        slots = [ring.write(im) for im in images]
        t3 = time.perf_counter()
        ring.release(s for s, _ in slots)
        t4 = time.perf_counter()
        blob = pickle.dumps((ids, slots, prefix))
        pickle.loads(blob)
        t5 = time.perf_counter()
        if rope is not None:
            rope(ids, images, token)
        if runs:
            imageprep.runs(ids, token)
        t6 = time.perf_counter()
        t["expand"].append(t1 - t0)
        t["prefix"].append(t2 - t1)
        t["ring"].append(t3 - t2)
        t["slot"].append(t4 - t3)
        t["result"].append(t5 - t4)
        t["submit"].append(t6 - t5)
        t0 = time.perf_counter()
        got, got_slots, got_prefix, _ = worker.job(task)
        t["job"].append(time.perf_counter() - t0)
        if got != ids or got_prefix != prefix:
            raise RuntimeError("the timed stages diverge from worker.job")
        ring.release(s for s, _ in got_slots)
        prompt.append(len(ids))
    ring.close()
    pools, ipc = {}, {}
    init = (str(path), name, token)
    reply = (list(range(round(statistics.mean(prompt)))), [(0, (1024, 1024, 3))], 684)
    jobs = [(worker.job, (c, KW)) for c in rows]
    for w in map(int, args.workers.split(",")):
        ring = ImageRing(2 * w * 4, proc.MAX_BYTES, ctx.Queue(), pin=False)
        pools[w] = pool_rate(ctx, w, jobs, (init, ring))
        ring.close()
        ipc[w] = pool_rate(ctx, w, [(echo, (c, KW), reply) for c in rows])
        print(w, pools[w], ipc[w], flush=True)
    report = {
        "model": str(path),
        "processor": name,
        "n_rows": len(rows),
        "n_images": len(srcs),
        "formats": formats,
        "src_mb_mean": round(statistics.mean(len(s) for s in srcs) / 1e6, 4),
        "in_mpix_mean": round(statistics.mean(a * b for a, b in sizes) / 1e6, 3),
        "out_mb_mean": round(statistics.mean(out_bytes) / 1e6, 3),
        "prompt_tokens_mean": round(statistics.mean(prompt), 1),
        **{f"{k}_ms": ms(v) for k, v in t.items()},
        "pool": pools,
        "ipc": ipc,
    }
    out = ROOT / "reports" / f"prep_{args.tag or path.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
