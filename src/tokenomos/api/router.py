"""Data parallel serving: one ``tokenomos.api.server`` process per GPU (each its own engine and prep
pool, ports ``port + 1 + i``), behind one front port that forwards every request to the backend
with the fewest requests in flight. Backends are started, health-checked and stopped with the
router; each holds a pipe from it and exits when that closes, so a killed router leaves none.

    python -m tokenomos.api.router --model M --gpus 0,1,2,3 --port 8000 -- [server args]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import resource
except ImportError:  # Windows has no per-process open-file limit to raise
    resource = None


class Backends:
    def __init__(self, urls: list[str]):
        self.urls, self.load = urls, [0] * len(urls)
        self.lock = threading.Lock()

    def take(self) -> int:
        with self.lock:
            i = min(range(len(self.urls)), key=self.load.__getitem__)
            self.load[i] += 1
            return i

    def give(self, i: int) -> None:
        with self.lock:
            self.load[i] -= 1


class Handler(BaseHTTPRequestHandler):
    backends: Backends
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def _relay(self, reply) -> None:
        """A streaming reply, passed on as it arrives; the backend's load holds until it ends."""
        self.close_connection = True
        self.send_response(reply.status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while data := reply.read1(1 << 16):
                self.wfile.write(data)
                self.wfile.flush()
        except OSError:
            pass

    def _forward(self, method: str) -> None:
        body = (
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if method == "POST"
            else None
        )
        i = self.backends.take()
        try:
            req = urllib.request.Request(
                self.backends.urls[i] + self.path,
                body,
                {"Content-Type": "application/json"},
                method=method,
            )
            try:
                with urllib.request.urlopen(req, timeout=3600) as r:
                    if r.headers.get_content_type() == "text/event-stream":
                        return self._relay(r)
                    code, data = r.status, r.read()
            except urllib.error.HTTPError as e:
                code, data = e.code, e.read()
            except OSError as e:
                msg = json.dumps({"error": {"message": f"backend: {e}", "type": "server_error"}})
                code, data = 502, msg.encode()
        finally:
            self.backends.give(i)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._forward("GET")

    def do_POST(self) -> None:
        self._forward("POST")


def raise_open_files() -> None:
    """Soft open-file limit to the hard one: each in-flight request holds sockets."""
    if resource is None:
        return
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def wait_healthy(url: str, procs: list, timeout: float = 900) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if any(p.poll() is not None for p in procs):
            raise RuntimeError("a backend exited during startup")
        try:
            urllib.request.urlopen(url + "/health", timeout=2).read()
            return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"{url} not healthy after {timeout}s")


def main() -> None:
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1 :] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log-dir", default="")
    args = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
    raise_open_files()
    # SIGTERM unwinds through the ``finally`` below, which stops the backends
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    gpus = args.gpus.split(",")
    procs, urls = [], []
    for i, gpu in enumerate(gpus):
        port = args.port + 1 + i
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        cmd = [sys.executable, "-m", "tokenomos.api.server", "--model", args.model, "--port"]
        log = None
        if args.log_dir:
            # the backend holds its log for its whole life
            log = open(os.path.join(args.log_dir, f"backend{i}.log"), "w")  # noqa: SIM115
        cmd = [*cmd, str(port), "--watch-stdin", *rest]
        procs.append(subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=log, stderr=log))
        urls.append(f"http://127.0.0.1:{port}")
    try:
        for url in urls:
            wait_healthy(url, procs)
        handler = type("Bound", (Handler,), {"backends": Backends(urls)})
        server_cls = type("Server", (ThreadingHTTPServer,), {"request_queue_size": 4096})
        server = server_cls((args.host, args.port), handler)
        server.daemon_threads = True
        print(f"tokenomos router on http://{args.host}:{args.port}: {len(urls)} GPUs", flush=True)
        server.serve_forever()
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait()


if __name__ == "__main__":
    main()
