"""One conversation's whole host preparation, run in a prep worker (or inline without a pool):
chat-template render, image decode and resize (the model's processor), tokenize, each image
placeholder expanded to its image's token count; the images land in the ``ImageRing`` when one
is given. Workers keep one thread each (torch intra-op threads would oversubscribe the pool).
"""

import base64
import functools
import importlib
import io
import multiprocessing
import os
import threading
import time

import torch
from jinja2 import TemplateError
from PIL import Image
from tokenizers import Tokenizer

from tokenomos.api.prep.ring import ImageRing
from tokenomos.api.template import ChatTemplate

_S: dict = {}


def open_image(src) -> Image.Image:
    """A data URL, a path, raw bytes or a PIL image."""
    if isinstance(src, Image.Image):
        return src
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src))
    if isinstance(src, str) and src.startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(src.split(",", 1)[1])))
    return Image.open(src.removeprefix("file://"))


def init(folder: str, processor: str | None, image_token, ring_args, max_images: int) -> None:
    """``ring_args``: ``ImageRing.args()`` of the serving process's ring, or None (tensors are
    returned instead)."""
    _S["template"] = ChatTemplate(folder)
    _S["tokenizer"] = Tokenizer.from_file(f"{folder}/tokenizer.json")
    _S["processor"] = importlib.import_module(processor) if processor else None
    _S["image_token"], _S["max_images"] = image_token, max_images
    _S["ring"] = ImageRing(*ring_args) if ring_args else None
    _head_ids.cache_clear()


@functools.lru_cache(maxsize=64)
def _head_ids(text: str) -> tuple[int, ...]:
    """A conversation head's tokens: conversations of one system prompt repeat it."""
    return tuple(_S["tokenizer"].encode(text, add_special_tokens=False).ids)


def _watch_parent() -> None:
    multiprocessing.parent_process().join()
    os._exit(1)


def start(*args) -> None:
    """Pool initializer: one torch thread, then ``init``; the worker ends with its parent, even
    one killed outright (a worker waiting on the ring would otherwise wait forever)."""
    torch.set_num_threads(1)
    threading.Thread(target=_watch_parent, daemon=True).start()
    init(*args)


def ping(_) -> None:
    time.sleep(0.05)


def expand(ids: list[int], counts: list[int], token, wrap=None) -> list[int]:
    """Each ``token`` placeholder repeated to its image's count; ``wrap(n)``, the processor's
    when it defines one, gives a placeholder's whole expansion (markers around the tokens)."""
    counts, out = iter(counts), []
    for t in ids:
        if t != token:
            out.append(t)
        elif wrap is None:
            out += [t] * next(counts)
        else:
            out += wrap(next(counts))
    return out


def shared_prefix(messages: list[dict], ids: list[int], kw: dict) -> int:
    """Leading tokens other conversations may share: the conversation before its last message
    (as far as its rendering is a prefix of ``ids``; nothing when the template refuses that
    partial conversation, as Qwen3.5's does without a user turn), and never past the first
    image."""
    n = 0
    if len(messages) > 1:
        try:
            head, _ = _S["template"].render(messages[:-1], add_generation_prompt=False, **kw)
        except (ValueError, TemplateError):
            head = ""
        h = _head_ids(head)
        while n < min(len(h), len(ids)) and h[n] == ids[n]:
            n += 1
    tok = _S["image_token"]
    first = next((i for i, t in enumerate(ids) if t == tok), None) if tok is not None else None
    if first is None:
        return n
    return min(n, first) if n else first


def job(task):
    """(messages, template kwargs) -> (prompt ids, images, shared prefix length, stamps): images
    as ``(slot, shape)`` pairs with a ring, else uint8 tensors; stamps (``time.time``): the job's
    start, its images prepared, and its end (after any wait for ring slots)."""
    start = time.time()
    ids, images, prefix, prepared = _job(task)
    return ids, images, prefix, (start, prepared, time.time())


def _job(task):
    messages, kw = task
    text, srcs = _S["template"].render(messages, **kw)
    proc = _S["processor"]
    if srcs and proc is None:
        raise ValueError("this model takes no images")
    if len(srcs) > _S["max_images"]:
        raise ValueError(f"{len(srcs)} images over the limit {_S['max_images']}")
    images = [proc.prepare(open_image(s)) for s in srcs]
    ids = _S["tokenizer"].encode(text, add_special_tokens=False).ids
    wrap = getattr(proc, "wrap", None)
    ids = expand(ids, [proc.tokens(im) for im in images], _S["image_token"], wrap)
    prefix = shared_prefix(messages, ids, kw)
    prepared = time.time()
    ring = _S["ring"]
    if ring is None:
        return ids, images, prefix, prepared
    if any(im.numel() > ring.slot_bytes for im in images):
        raise ValueError(f"a prepared image is over the ring slot size {ring.slot_bytes}")
    return ids, [ring.write(im) for im in images], prefix, prepared
