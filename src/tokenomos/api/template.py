"""A checkpoint's chat template rendered as transformers renders it: sandboxed jinja2, ``tojson``
without ASCII escaping, ``raise_exception``, and the ``{% generation %}`` block tag (a plain
pass-through here). Image items in OpenAI form (``image_url``) are rendered as the template's
``image`` items; ``render`` also returns the image sources in placeholder order.
"""

import json
from pathlib import Path

from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class _Generation(Extension):
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.Scope(body, lineno=lineno)


def _raise(message: str):
    raise ValueError(message)


def _tojson(value, indent=None, ensure_ascii=False, sort_keys=False):
    return json.dumps(value, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)


class ChatTemplate:
    def __init__(self, folder: str | Path):
        folder = Path(folder)
        cfg = json.loads((folder / "tokenizer_config.json").read_text("utf-8"))
        path = folder / "chat_template.jinja"
        source = path.read_text("utf-8") if path.exists() else cfg["chat_template"]
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[_Generation, "jinja2.ext.loopcontrols"],
        )
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        self.template = env.from_string(source)
        self.special = {k: cfg.get(k) for k in ("bos_token", "eos_token", "pad_token")}

    def render(self, messages: list[dict], add_generation_prompt: bool = True, **kwargs):
        """(prompt text, image sources in order). An image source is whatever the item held:
        a data / http URL, a path, bytes or a PIL image."""
        images, out = [], []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                items = []
                for item in content:
                    kind = item.get("type") if isinstance(item, dict) else None
                    if kind == "image_url":
                        url = item["image_url"]
                        images.append(url["url"] if isinstance(url, dict) else url)
                        items.append({"type": "image"})
                    elif kind == "image":
                        images.append(item.get("image") or item.get("url") or item.get("path"))
                        items.append({"type": "image"})
                    else:
                        items.append(item)
                m = {**m, "content": items}
            out.append(m)
        text = self.template.render(
            messages=out, add_generation_prompt=add_generation_prompt, **self.special, **kwargs
        )
        return text, images
