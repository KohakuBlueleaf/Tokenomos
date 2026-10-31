"""A server watching stdin still starts children, and ends when the pipe closes."""

import subprocess
import sys

BODY = """
import subprocess, sys, time
from tokenomos.api.server import _watch_stdin
_watch_stdin()
subprocess.run([sys.executable, "-c", "pass"], timeout=30, check=True)
print("child ran", flush=True)
time.sleep(60)
"""


def test_child_starts_and_a_closed_pipe_ends_the_server():
    p = subprocess.Popen(
        [sys.executable, "-c", BODY], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        assert p.stdout.readline().strip() == "child ran"
        p.stdin.close()
        assert p.wait(30) == 0
    finally:
        p.kill()
