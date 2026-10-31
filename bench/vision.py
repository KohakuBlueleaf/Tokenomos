"""The image encoder alone: the first ``N`` rows' images prepared as serving prepares them (pinned,
staged like the engine's), through ``model.encoders["image"]`` in batches of each size, every
batch between two CUDA timing events. GPU images/s (the events), wall images/s, host milliseconds
per image up to the enqueue, peak allocated memory. The first pass of each size compiles kernels
and is dropped. A batch size past the memory cap ends the sweep. Writes
``reports/vision_<tag>.json``.

    python bench/vision.py --model M --rows ROWS --arch-kw '{"head": "nvfp4"}'
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tokenomos import models
from tokenomos.api.prep import worker
from tokenomos.core.staging import Staging

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda"
GIB = 1 << 30


def one_pass(enc, images, batch, staging) -> tuple[float, float, float]:
    """(GPU s, wall s, host s) over ``images`` in batches of ``batch`` (a short tail dropped)."""
    gpu = host = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, at in enumerate(range(0, len(images) - batch + 1, batch)):
        staging.begin(i)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        h0 = time.perf_counter()
        a.record()
        imgs = [im.to(DEV, non_blocking=True) for im in images[at : at + batch]]
        feats = enc(imgs, staging.put)
        b.record()
        host += time.perf_counter() - h0
        b.synchronize()
        gpu += a.elapsed_time(b) / 1e3
        del feats, imgs
    return gpu, time.perf_counter() - t0, host


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batches", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--arch-kw", default='{"head": "mxfp8"}', help="JSON kwargs of the declaration")
    ap.add_argument("--memory-cap", type=float, default=0.9, help="share of the card allocated")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.memory_cap)
    conf = models.configure(args.model, **json.loads(args.arch_kw))
    model = models.load(args.model, DEV, conf)
    enc = model.encoders["image"]
    worker.init(args.model, model.processor, model.image_token, None, 4)
    images, tokens, shapes = [], [], {}
    with open(args.rows, encoding="utf-8") as f:
        for line in f:
            prompt, ims, _ = worker.job((json.loads(line)["messages"], {"enable_thinking": False}))
            images += [im.pin_memory() for im in ims]
            tokens.append(sum(t == model.image_token for t in prompt))
            for im in ims:
                shapes[str(tuple(im.shape))] = shapes.get(str(tuple(im.shape)), 0) + 1
            if len(images) >= args.n:
                break
    staging = Staging(256 << 20, torch.device(DEV))
    sizes, peak_base = {}, torch.cuda.memory_allocated() / GIB
    with torch.no_grad():
        for batch in map(int, args.batches.split(",")):
            torch.cuda.reset_peak_memory_stats()
            try:
                runs = [one_pass(enc, images, batch, staging) for _ in range(args.passes)]
            except torch.cuda.OutOfMemoryError:
                sizes[batch] = {"over_memory_cap": True}
                print(batch, sizes[batch], flush=True)
                break
            gpu, wall, host = runs[-1]
            done = len(images) // batch * batch
            sizes[batch] = {
                "images": done,
                "gpu_s": round(gpu, 3),
                "gpu_img_s": round(done / gpu, 1),
                "wall_img_s": round(done / wall, 1),
                "host_ms_per_image": round(host / done * 1e3, 3),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 2),
            }
            print(batch, sizes[batch], flush=True)
    done = {b: v for b, v in sizes.items() if "gpu_img_s" in v}
    best = max(done, key=lambda b: done[b]["gpu_img_s"])
    report = {
        "model": args.model,
        "arch_kw": json.loads(args.arch_kw),
        "n_images": len(images),
        "image_tokens_mean": round(statistics.mean(tokens), 1),
        "shapes_top": sorted(shapes.items(), key=lambda kv: -kv[1])[:8],
        "model_allocated_gib": round(peak_base, 2),
        "best_batch": best,
        "best_gpu_img_s": done[best]["gpu_img_s"],
        "batches": sizes,
    }
    tag = args.tag or Path(args.model).name
    out = ROOT / "reports" / f"vision_{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "batches"}, indent=1))


if __name__ == "__main__":
    main()
