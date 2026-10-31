"""In-Python use: ``LLM(path).generate(prompts, params)`` / ``LLM(path).chat(conversations,
params)``; the HTTP server drives the same object through ``submit_chat``.

The engine loop runs in a background thread serving its inbox. Each conversation's whole host
preparation (chat template, image decode and resize, tokenize, placeholder expansion) runs in a
process pool, the images written into the shared page-locked ``ImageRing``; when it completes the
request enters the engine directly, so preparation overlaps the GPU and the serving process does
no per-image work. The ring slots return once the engine's image copies complete. Stops: the
generation config's EOS ids plus the tokenizer's EOS, and any ``stop_token_ids`` given.
"""

import importlib
import json
import multiprocessing
import queue
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tokenomos import models
from tokenomos._rs import grammar as rs_grammar
from tokenomos.api.prep import worker
from tokenomos.api.prep.ring import ImageRing
from tokenomos.api.template import ChatTemplate
from tokenomos.api.tokens import token_bytes
from tokenomos.core.engine import Engine, Request
from tokenomos.core.grammar import DeviceGrammar


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int = 0
    stop_token_ids: tuple[int, ...] = ()
    # a JSON schema (text or dict) the reply must follow
    json_schema: str | dict | None = None


@dataclass
class Output:
    text: str
    token_ids: list[int]
    prompt_tokens: int
    finish: str
    # grammar replies: the device state each token was sampled under
    states: list[int] | None = None


def _done(value) -> Future:
    f: Future = Future()
    f.set_result(value)
    return f


def _failed(e: BaseException) -> Future:
    f: Future = Future()
    f.set_exception(e)
    return f


class LLM:
    def __init__(
        self,
        model: str,
        device: str = "cuda",
        arch=None,
        prep_workers: int = 16,
        max_images: int = 4,
        ring_slots: int | None = None,
        grammar: bool = True,
        grammar_ws: int = 8,
        grammar_kw: dict | None = None,
        own_stream: bool = False,
        **engine_kw,
    ):
        """``arch``: a configured declaration (e.g. ``Lfm3(head="mxfp8")``); ``engine_kw`` go to
        ``Engine`` (``slots``, ``max_len``, ``new_cap``, ...). ``prep_workers`` 0 prepares inline
        (the submitting thread). ``ring_slots`` (default ``2 * prep_workers * max_images``) must
        cover every worker holding ``max_images`` slots. ``grammar``: JSON-schema replies
        (``SamplingParams.json_schema``), ``grammar_ws`` whitespace bytes allowed per gap,
        ``grammar_kw`` to ``DeviceGrammar``. ``own_stream``: the engine thread serves on a CUDA
        stream of its own, apart from other engines in the process."""
        self.own_stream = own_stream
        self.path = Path(model)
        self.model = models.load(model, device, arch)
        self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        self.template = ChatTemplate(self.path)
        gen_path = self.path / "generation_config.json"
        gen = json.loads(gen_path.read_text("utf-8")) if gen_path.exists() else {}
        eos = gen.get("eos_token_id", [])
        eos = set(eos if isinstance(eos, list) else [eos])
        tok_eos = self.template.special.get("eos_token")
        if tok_eos:
            eos.add(self.tokenizer.token_to_id(tok_eos))
        self.eos = tuple(sorted(e for e in eos if e is not None))
        dg = None
        if grammar:
            vocab = self.model.embedding.shape[0]
            store = rs_grammar.Store(
                token_bytes(self.tokenizer, vocab), list(self.eos), vocab, grammar_ws
            )
            dg = DeviceGrammar(store, device, **(grammar_kw or {}))
        self.engine = Engine(self.model, device=device, grammar=dg, **engine_kw)
        self.engine.warmup()
        proc = self.model.processor
        init = (str(self.path), proc, self.model.image_token)
        self.pool = self.ring = None
        if prep_workers:
            ctx = multiprocessing.get_context("spawn")
            if proc:
                slots = ring_slots or 2 * prep_workers * max_images
                if slots < prep_workers * max_images:
                    raise ValueError("ring_slots under prep_workers * max_images can deadlock")
                slot_bytes = importlib.import_module(proc).MAX_BYTES
                self.ring = ImageRing(slots, slot_bytes, ctx.Queue())
            ring_args = self.ring.args() if self.ring else None
            # spawned: the workers never touch the CUDA context this process holds
            self.pool = ProcessPoolExecutor(
                prep_workers,
                mp_context=ctx,
                initializer=worker.start,
                initargs=(*init, ring_args, max_images),
            )
            # start every worker now (each imports torch once), not on the first large batch
            list(self.pool.map(worker.ping, range(prep_workers * 4)))
        else:
            worker.init(*init, None, max_images)
        self._stop, self._error = threading.Event(), None
        self._thread = threading.Thread(target=self._serve, name="tokenomos-engine", daemon=True)
        self._thread.start()

    def request(self, ids: list[int], p: SamplingParams, images=()) -> Request:
        """An engine Request for prompt ``ids`` (stops: the model's EOS plus ``p``'s)."""
        grammar = None
        if p.json_schema is not None:
            if self.engine.grammar is None:
                raise ValueError("json_schema on an LLM built with grammar=False")
            schema = p.json_schema
            text = schema if isinstance(schema, str) else json.dumps(schema)
            grammar = self.engine.grammar.compile(text)
        return Request(
            ids,
            grammar=grammar,
            max_new=p.max_tokens,
            temperature=p.temperature,
            top_p=p.top_p,
            top_k=p.top_k,
            min_p=p.min_p,
            seed=p.seed,
            stop=tuple(dict.fromkeys(self.eos + tuple(p.stop_token_ids))),
            images=list(images),
        )

    # ---- the serving thread ----

    def _serve(self) -> None:
        try:
            if self.own_stream:
                torch.cuda.set_stream(torch.cuda.Stream())
            self.engine.run(self._stop)
        except BaseException as e:
            self._error = e
            raise

    def submit(self, req: Request) -> Future:
        """Queue one engine request; the future resolves to it once ``req.result`` is set."""
        fut: Future = Future()
        req.on_done = lambda r: fut.set_result(r)
        self.engine.submit(req)
        return fut

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def wait(self, futs: list[Future]) -> list[Request]:
        """Every future's Request; raises if the engine thread died."""
        pending = set(futs)
        while pending:
            if self._error is not None:
                raise RuntimeError("engine thread failed") from self._error
            _, pending = wait(pending, timeout=0.5)
        return [f.result() for f in futs]

    def output(self, req: Request) -> Output:
        r = req.result
        text = self.tokenizer.decode(r.tokens, skip_special_tokens=True)
        return Output(text, r.tokens, len(req.prompt), r.finish, r.states)

    @property
    def failure(self) -> BaseException | None:
        """What ended the engine thread, or None while it serves."""
        return self._error

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        if self.pool is not None:
            # a failed engine never returns its ring slots: workers may be waiting for one
            procs = list(self.pool._processes.values()) if self._error is not None else []
            for p in procs:
                p.terminate()
            self.pool.shutdown(cancel_futures=True)
        if self.ring is not None:
            self.ring.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- requests ----

    def generate(self, prompts: list, params: SamplingParams | None = None) -> list[Output]:
        """``prompts``: text (tokenized as is) or token id lists."""
        p = params or SamplingParams()
        futs = [
            self.submit(self.request(x if isinstance(x, list) else self.encode(x), p))
            for x in prompts
        ]
        return [self.output(r) for r in self.wait(futs)]

    def _events(self, fut: Future, tokens: queue.SimpleQueue) -> Iterator[str | Output]:
        """Text deltas as ``tokens`` deliver them (a partial UTF-8 sequence waits for its last
        token), then the finished ``Output``."""
        fut.add_done_callback(lambda _: tokens.put(None))
        ids, text = [], ""
        while True:
            try:
                new = tokens.get(timeout=0.5)
            except queue.Empty:
                if self._error is not None:
                    raise RuntimeError("engine thread failed") from self._error
                continue
            if new is None:
                break
            ids += new
            full = self.tokenizer.decode(ids, skip_special_tokens=True)
            if len(full) > len(text) and not full.endswith("�"):
                yield full[len(text) :]
                text = full
        out = self.output(fut.result())
        if len(out.text) > len(text):
            yield out.text[len(text) :]
        yield out

    def stream(self, ids: list[int], p: SamplingParams) -> Iterator[str | Output]:
        """``generate`` of one prompt as text deltas, then the ``Output``."""
        tokens: queue.SimpleQueue = queue.SimpleQueue()
        req = self.request(ids, p)
        req.on_tokens = tokens.put
        return self._events(self.submit(req), tokens)

    def stream_chat(
        self, conversation: list[dict], p: SamplingParams, **template_kw
    ) -> Iterator[str | Output]:
        """``submit_chat`` as text deltas, then the ``Output``."""
        tokens: queue.SimpleQueue = queue.SimpleQueue()
        return self._events(self.submit_chat(conversation, p, tokens.put, **template_kw), tokens)

    def submit_chat(
        self, conversation: list[dict], p: SamplingParams, on_tokens=None, **template_kw
    ) -> Future:
        """Prepare ``conversation`` (prep pool) and queue it; the future resolves to the finished
        Request (``output`` turns it into text). ``on_tokens``: ``Request.on_tokens``."""
        out: Future = Future()

        def ready(job: Future):
            release = None
            try:
                ids, images, prefix = job.result()
                if self.ring is not None:
                    slots = [s for s, _ in images]
                    release = lambda: self.ring.release(slots)  # noqa: E731
                    images = [self.ring.view(s, shape) for s, shape in images]
                req = self.request(ids, p, images)
                req.prefix, req.on_tokens, req.release = prefix, on_tokens, release
                req.on_done = lambda r: out.set_result(r)
                self.engine.submit(req)
            except BaseException as e:  # noqa: BLE001 - a bad conversation fails its future
                if release is not None:
                    release()
                out.set_exception(e)

        task = (conversation, template_kw)
        if self.pool is not None:
            job = self.pool.submit(worker.job, task)
        else:
            try:
                job = _done(worker.job(task))
            except Exception as e:  # noqa: BLE001
                job = _failed(e)
        job.add_done_callback(ready)
        return out

    def chat(
        self, conversations: list[list[dict]], params: SamplingParams | None = None, **template_kw
    ) -> list[Output]:
        """``template_kw`` go to the chat template (e.g. ``enable_thinking=False``)."""
        p = params or SamplingParams()
        futs = [self.submit_chat(c, p, **template_kw) for c in conversations]
        return [self.output(r) for r in self.wait(futs)]
