"""The OpenAI-compatible HTTP server over one ``LLM``: ``GET /v1/models``, ``POST
/v1/chat/completions`` and ``POST /v1/completions`` (``n`` 1). Each request is a handler thread
that submits to the shared engine and waits on it, so concurrent clients batch together on the
GPU.

Request fields: ``messages`` / ``prompt``, ``max_tokens`` (or ``max_completion_tokens``),
``temperature``, ``top_p``, ``top_k``, ``min_p``, ``seed``, ``stop_token_ids``,
``chat_template_kwargs``, ``response_format`` (``json_schema``; also ``json_schema`` /
``guided_json`` directly), ``stream`` (server-sent events ending in ``[DONE]``),
``stream_options.include_usage``. Stop strings are refused, not ignored. ``GET /health`` answers
503 once the engine has failed.

    python -m tokenomos.api.server --model M --port 8000 --arch-kw '{"head": "mxfp8"}'
"""

import argparse
import itertools
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tokenomos import models
from tokenomos.api.llm import LLM, SamplingParams
from tokenomos.api.router import raise_open_files


def params(body: dict, default_max: int) -> SamplingParams:
    if body.get("n", 1) != 1:
        raise ValueError("only n=1 is served")
    if body.get("stop"):
        raise ValueError("stop strings are not served: use stop_token_ids")
    return SamplingParams(
        max_tokens=int(body.get("max_completion_tokens") or body.get("max_tokens") or default_max),
        temperature=float(body.get("temperature", 1.0)),
        top_p=float(body.get("top_p", 1.0)),
        top_k=int(body.get("top_k", 0) or 0),
        min_p=float(body.get("min_p", 0.0)),
        seed=int(body.get("seed") or 0),
        stop_token_ids=tuple(body.get("stop_token_ids") or ()),
        json_schema=schema_of(body),
    )


def schema_of(body: dict):
    """``response_format: {"type": "json_schema", "json_schema": {"schema": ...}}`` (OpenAI) or
    ``json_schema`` / ``guided_json`` given directly."""
    fmt = body.get("response_format") or {}
    if fmt.get("type") == "json_schema":
        return (fmt.get("json_schema") or {}).get("schema")
    return body.get("json_schema") or body.get("guided_json")


def usage(prompt: int, completion: int) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


class Handler(BaseHTTPRequestHandler):
    llm: LLM
    name: str
    default_max: int
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def _send(self, code: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, message: str) -> None:
        self._send(code, {"error": {"message": message, "type": "invalid_request_error"}})

    def _sse(self, obj) -> None:
        text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        self.wfile.write(f"data: {text}\n\n".encode())
        self.wfile.flush()

    def _stream(self, events, chat: bool, body: dict) -> None:
        """One event per text delta, then the finish event; the connection ends the reply. A
        client that left ends the writes; its request runs out on the engine."""
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        base = {
            "id": f"tk-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk" if chat else "text_completion",
            "created": int(time.time()),
            "model": self.name,
        }

        def chunk(part: dict, finish: str | None = None) -> dict:
            return {**base, "choices": [{"index": 0, **part, "finish_reason": finish}]}

        try:
            if chat:
                self._sse(chunk({"delta": {"role": "assistant", "content": ""}}))
            for event in events:
                if isinstance(event, str):
                    self._sse(chunk({"delta": {"content": event}} if chat else {"text": event}))
                    continue
                finish = "stop" if event.finish == "stop" else "length"
                self._sse(chunk({"delta": {}} if chat else {"text": ""}, finish))
                if (body.get("stream_options") or {}).get("include_usage"):
                    used = usage(event.prompt_tokens, len(event.token_ids))
                    self._sse({**base, "choices": [], "usage": used})
            self._sse("[DONE]")
        except OSError:
            pass

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            model = {"id": self.name, "object": "model", "owned_by": "tokenomos"}
            self._send(200, {"object": "list", "data": [model]})
        elif self.path.rstrip("/") == "/health":
            failed = self.llm.failure is not None
            self._send(503 if failed else 200, {"status": "failed" if failed else "ok"})
        else:
            self._error(404, f"no route {self.path}")

    def do_POST(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            p = params(body, self.default_max)
            route = self.path.rstrip("/")
            streaming = bool(body.get("stream"))
            if route == "/v1/chat/completions":
                kw = body.get("chat_template_kwargs") or {}
                if streaming:
                    events = self.llm.stream_chat(body["messages"], p, **kw)
                else:
                    fut = self.llm.submit_chat(body["messages"], p, **kw)
            elif route == "/v1/completions":
                prompt = body["prompt"]
                ids = prompt if isinstance(prompt, list) else self.llm.encode(prompt)
                if streaming:
                    events = self.llm.stream(ids, p)
                else:
                    fut = self.llm.submit(self.llm.request(ids, p))
            else:
                return self._error(404, f"no route {self.path}")
            if streaming:
                # a bad request fails here, ahead of the response headers
                events = itertools.chain([next(events)], events)
            else:
                out = self.llm.output(self.llm.wait([fut])[0])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            return self._error(400, f"{type(e).__name__}: {e}")
        except RuntimeError as e:
            return self._send(500, {"error": {"message": str(e), "type": "server_error"}})
        if streaming:
            return self._stream(events, route == "/v1/chat/completions", body)
        finish = "stop" if out.finish == "stop" else "length"
        base = {"id": f"tk-{uuid.uuid4().hex}", "created": int(time.time()), "model": self.name}
        if route == "/v1/chat/completions":
            msg = {"role": "assistant", "content": out.text}
            choice = {"index": 0, "message": msg, "finish_reason": finish}
            obj = "chat.completion"
        else:
            choice = {"index": 0, "text": out.text, "finish_reason": finish}
            obj = "text_completion"
        usage_ = usage(out.prompt_tokens, len(out.token_ids))
        self._send(200, {**base, "object": obj, "choices": [choice], "usage": usage_})


def serve(llm: LLM, host: str, port: int, name: str, default_max: int = 1024) -> None:
    handler = type("Bound", (Handler,), {"llm": llm, "name": name, "default_max": default_max})
    # listen backlog: the stdlib default (5) resets connections under a burst of clients
    server_cls = type("Server", (ThreadingHTTPServer,), {"request_queue_size": 4096})
    server = server_cls((host, port), handler)
    server.daemon_threads = True
    print(f"tokenomos serving {name} on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        llm.close()


def _exit_on_eof(pipe: int) -> None:
    """The router holds the far end of ``pipe``: its end, however it ends, ends this server."""
    while os.read(pipe, 1 << 12):
        pass
    os._exit(0)


def _watch_stdin() -> None:
    """Watch a private copy of stdin and leave ``NUL`` at fd 0: on Windows a child starting with
    the same pipe as its stdin hangs while a read on that pipe is pending."""
    pipe = os.dup(0)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    threading.Thread(target=_exit_on_eof, args=(pipe,), daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--name", default="")
    ap.add_argument("--arch-kw", default="{}", help="JSON kwargs of the model declaration")
    ap.add_argument("--engine-kw", default="{}", help="JSON kwargs of the engine")
    ap.add_argument("--prep-workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024, help="default max_tokens")
    ap.add_argument("--watch-stdin", action="store_true", help="exit when stdin closes")
    args = ap.parse_args()
    if args.watch_stdin:
        _watch_stdin()
    raise_open_files()
    arch = models.configure(args.model, **json.loads(args.arch_kw))
    llm = LLM(args.model, arch=arch, prep_workers=args.prep_workers, **json.loads(args.engine_kw))
    serve(llm, args.host, args.port, args.name or Path(args.model).name, args.max_tokens)


if __name__ == "__main__":
    main()
