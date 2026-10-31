"""The server's refusals and its health: stop strings fail, a failed engine reads 503."""

import json
import socket
import threading
import urllib.error
import urllib.request

import pytest

from tokenomos.api.server import params, serve


class Stub:
    failure = None

    def close(self) -> None:
        pass


def test_stop_strings_are_refused():
    with pytest.raises(ValueError, match="stop_token_ids"):
        params({"stop": ["\n"]}, 16)
    assert params({"stop": []}, 16).max_tokens == 16


def test_health_follows_the_engine():
    stub = Stub()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    threading.Thread(target=serve, args=(stub, "127.0.0.1", port, "stub"), daemon=True).start()
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(100):
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                assert json.loads(r.read())["status"] == "ok"
            break
        except urllib.error.URLError:
            threading.Event().wait(0.05)
    else:
        pytest.fail("the server did not start")
    stub.failure = RuntimeError("engine thread failed")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(url, timeout=1)
    assert e.value.code == 503
