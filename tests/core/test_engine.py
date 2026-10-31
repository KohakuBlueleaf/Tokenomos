"""The engine against the teacher-forced dense reference, on LFM3 (bf16 and MXFP8).

24 natural-text prompts (5 .. 700 tokens, past the 256-token window) through 8 slots, so slots,
pages and rings are reused while others decode. Every generated token is checked against the
dense reference run on prompt + generated tokens: in bf16 it must be the reference's argmax or
within a near-tie of it (MXFP8: a bounded rate of larger gaps). Graph replay must equal the
eager step bit for bit, a stop token must end the output at its first occurrence, and seeded
sampling must repeat. Numbers go to ``reports/engine.json``.
"""

import json
import threading
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from tokenomos import models as tk_models
from tokenomos.core.engine import Engine, Request
from tokenomos.kernels.engine import slots as kslots
from tokenomos.models import reference
from tokenomos.models.blocks.attention import use_backend

pytestmark = [pytest.mark.gpu, pytest.mark.model]
DEV = "cuda"
REPORT = Path(__file__).resolve().parents[2] / "reports" / "engine.json"
LENGTHS = [5, 700, 33, 260, 1, 128, 300, 17, 64, 511, 2, 90] * 2
TIE = 0.15


def write(key: str, result: dict) -> None:
    REPORT.parent.mkdir(exist_ok=True)
    data = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    data[key] = result
    REPORT.write_text(json.dumps(data, indent=1))
    print(key, result)


@pytest.fixture(scope="module")
def prompts(models):
    path = models / "lfm3-vl-2b"
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    text = "\n".join((path / n).read_text("utf-8") for n in ("README.md", "LICENSE"))
    ids = tok.encode(text, add_special_tokens=False)
    out, at = [], 0
    for n in LENGTHS:
        at = (at + 97) % (len(ids) - n)
        out.append(ids[at : at + n])
    return out


def requests(prompts, **kw):
    return [Request(p, max_new=24 + 7 * (i % 5), **kw) for i, p in enumerate(prompts)]


def check(model, reqs, results) -> dict:
    """Teacher-forced: each generated token vs the dense reference's distribution there."""
    use_backend(model, "dense")
    seqs = [r.prompt + res.tokens[:-1] for r, res in zip(reqs, results)]
    logits = reference.forward_logits(model, seqs, DEV)
    use_backend(model, "paged")
    agree, gaps = 0, []
    total = 0
    for r, res, lg in zip(reqs, results, logits):
        lp = lg[len(r.prompt) - 1 :].log_softmax(-1)
        toks = torch.tensor(res.tokens, device=DEV)
        gap = lp.max(-1).values - lp.gather(-1, toks[:, None])[:, 0]
        agree += int((gap == 0).sum())
        total += toks.numel()
        gaps.append(gap)
    gaps = torch.cat(gaps)
    return {
        "tokens": total,
        "argmax_agreement": round(agree / total, 5),
        "gap_mean": round(gaps.mean().item(), 5),
        "gap_max": round(gaps.max().item(), 5),
        "over_tie": int((gaps > TIE).sum()),
    }


@pytest.mark.parametrize(
    "name", ["lfm3-vl-2b", pytest.param("lfm3-vl-2b-mxfp8", marks=pytest.mark.sm100)]
)
@torch.no_grad()
def test_greedy_vs_reference(models, prompts, name):
    model = tk_models.load(str(models / name), DEV)
    reqs = requests(prompts)
    eng = Engine(model, slots=8, page=16, max_len=1024, new_cap=64, pages=600)
    results = eng.generate(reqs)
    for r, res in zip(reqs, results):
        assert len(res.tokens) == r.max_new and res.finish == "length"
    got = check(model, reqs, results)
    got["steps"] = eng.stats.steps
    write(f"greedy_{name}", got)
    # bf16: every token the argmax or a near-tie; MXFP8 activations amplify bf16-level
    # differences, so only a rate
    assert got["argmax_agreement"] >= 0.97
    if name.endswith("mxfp8"):
        assert got["over_tie"] <= 0.01 * got["tokens"] and got["gap_mean"] <= 0.03
    else:
        assert got["over_tie"] == 0
    eager = Engine(model, slots=8, page=16, max_len=1024, new_cap=64, pages=600, graphs=False)
    assert [x.tokens for x in eager.generate(reqs)] == [x.tokens for x in results]


@torch.no_grad()
def test_stop_and_sampling(models, prompts):
    model = tk_models.load(str(models / "lfm3-vl-2b"), DEV)
    eng = Engine(model, slots=8, page=16, max_len=1024, new_cap=64, pages=600)
    base = eng.generate(requests(prompts[:8]))
    stopped = []
    for p, res in zip(prompts[:8], base):
        t = res.tokens[5]
        stopped.append(Request(p, max_new=50, stop=(t,)))
    for req, res, b in zip(stopped, eng.generate(stopped), base):
        first = b.tokens.index(req.stop[0])
        assert res.tokens == b.tokens[: first + 1] and res.finish == "stop"
    hot = requests(prompts[:12], temperature=1.0, top_p=0.95, seed=7)
    one, two = eng.generate(hot), eng.generate(hot)
    assert [x.tokens for x in one] == [x.tokens for x in two]
    greedy = eng.generate(requests(prompts[:12]))
    assert sum(a.tokens != b.tokens for a, b in zip(one, greedy)) >= 10


@torch.no_grad()
def test_pages_run_out_without_prefix_table(models, prompts):
    """Requests past the pages wait for a finisher: the table's eviction is not there to run."""
    model = tk_models.load(str(models / "lfm3-vl-2b"), DEV)
    wide = Engine(model, slots=8, page=16, max_len=1024, new_cap=64, pages=600, prefix_entries=0)
    base = wide.generate(requests(prompts[:8]))
    need = max(-(-(len(p) + 52) // 16) for p in prompts[:8])
    tight = Engine(
        model, slots=8, page=16, max_len=1024, new_cap=64, pages=2 * need + 1, prefix_entries=0
    )
    assert [x.tokens for x in tight.generate(requests(prompts[:8]))] == [x.tokens for x in base]


def test_rows_past_their_reservation_sit_out(models, prompts):
    """Rows outgrowing a tight pool pause and resume: the tokens match a pool with room."""
    model = tk_models.load(str(models / "lfm3-vl-2b"), DEV)
    wide = Engine(model, slots=8, page=16, max_len=1024, new_cap=64, pages=600, prefix_entries=0)
    base = wide.generate(requests(prompts[:8]))
    held = [-(-(len(p) + 1) // 16) for p in prompts[:8]]
    pages = max(sum(held) // 2, 2 * max(held)) + 6
    tight = Engine(
        model,
        slots=8,
        page=16,
        max_len=1024,
        new_cap=64,
        pages=pages,
        prefix_entries=0,
        reserve_new=1,
    )
    got = tight.generate(requests(prompts[:8]))
    assert [x.tokens for x in got] == [x.tokens for x in base]
    assert tight.stats.pauses > 0 and not tight.paused
    assert int(tight.st.top) == pages - 1


def test_request_past_the_pool_is_refused(models):
    """A request the whole pool cannot hold fails at submit: no finisher would ever free it."""
    model = tk_models.load(str(models / "lfm3-vl-2b"), DEV)
    eng = Engine(model, slots=2, page=16, max_len=1024, new_cap=64, pages=9, prefix_entries=0)
    with pytest.raises(ValueError, match="page pool"):
        eng.submit(Request([1] * 200, max_new=2))
    eng.warmup()
    assert eng.generate([Request([1] * 100, max_new=2)])[0].tokens


def test_warmup_covers_the_slot_kernels_of_a_full_batch(models, monkeypatch):
    """A variant serving reaches first compiles mid-run, seconds of stall under every row."""
    model = tk_models.load(str(models / "lfm3-vl-2b"), DEV)
    eng = Engine(model, slots=64, page=32, max_len=1024, new_cap=64, pages=4096, prefix_entries=0)
    seen = set()
    alloc, append = kslots.alloc, kslots.append

    def spied_alloc(slot, target, n, st, maxneed=0):
        seen.add(("alloc", kslots._block(slot.shape[0]), maxneed))
        assert maxneed <= 1
        alloc(slot, target, n, st, maxneed)

    def spied_append(lst, n, staged, count):
        seen.add(("append", kslots._block(staged.shape[0])))
        append(lst, n, staged, count)

    monkeypatch.setattr(kslots, "alloc", spied_alloc)
    monkeypatch.setattr(kslots, "append", spied_append)
    eng.warmup()
    warmed, seen = set(seen), seen
    seen.clear()
    eng.generate([Request([1 + i] * (20 + 13 * i), max_new=2) for i in range(64)])
    assert seen and seen <= warmed, sorted(seen - warmed)


def test_bad_requests_fail_at_submit_and_the_engine_serves_on(models):
    """Past ``submit`` an error ends the engine thread, so each of these must stop there."""
    model = tk_models.load(str(models / "lfm3-vl-2b"), DEV)
    eng = Engine(model, slots=2, page=16, max_len=1024, new_cap=64, pages=64, prefix_entries=0)
    enc = model.encoders["image"]
    unit = enc.patch * enc.merge
    image = torch.zeros(2 * unit, 2 * unit, 3, dtype=torch.uint8)
    n = enc.tokens((2 * enc.merge, 2 * enc.merge))
    with pytest.raises(ValueError, match="at least one token"):
        eng.submit(Request([1, 2, 3], max_new=0))
    with pytest.raises(ValueError, match="image placeholders"):
        eng.submit(Request([model.image_token] * (n + 1) + [1], max_new=2, images=[image]))
    good = Request([1] * 4 + [model.image_token] * n + [1], max_new=2, images=[image])
    assert len(eng.generate([good])[0].tokens) == 2


def test_serving_folds_its_spans_as_it_runs(models):
    """A serving engine never reaches ``run``'s end: its events must not pile up until then."""
    model = tk_models.load(str(models / "lfm3-vl-2b"), DEV)
    eng = Engine(model, slots=4, page=16, max_len=1024, new_cap=64, pages=256, prefix_entries=0)
    stop = threading.Event()
    done = threading.Semaphore(0)
    thread = threading.Thread(target=eng.run, args=(stop,))
    thread.start()
    try:
        for i in range(32):
            eng.submit(Request([1 + i] * 40, max_new=2, on_done=lambda _: done.release()))
        for _ in range(32):
            assert done.acquire(timeout=120)
        for _ in range(100):
            if not eng.stats.spans:
                break
            threading.Event().wait(0.05)
        assert not eng.stats.spans and eng.stats.prefill_gpu_s > 0
    finally:
        stop.set()
        thread.join()


def test_streamed_tokens_arrive_in_runs_and_add_up(models, prompts):
    """``on_tokens`` delivers a reply in several runs before it ends, and they join to it."""
    model = tk_models.load(str(models / "lfm3-vl-2b"), DEV)
    eng = Engine(model, slots=8, page=16, max_len=1024, new_cap=64, pages=600, stream_every=2)
    eng.warmup()
    plain = eng.generate(requests(prompts))
    runs = [[] for _ in prompts]
    reqs = requests(prompts)
    for r, got in zip(reqs, runs):
        r.on_tokens = got.append
    results = eng.generate(reqs)
    for base, res, got in zip(plain, results, runs):
        assert sum(got, []) == res.tokens == base.tokens
        assert len(got) > 1
