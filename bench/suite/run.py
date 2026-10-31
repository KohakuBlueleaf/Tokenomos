"""One full measurement of a box, detached and resumable by nothing: ``arms.py`` is the plan. For
each card class the tests run first and a failure ends that class; the GDN tile sweep writes the
class's table; then one thread per card takes the class's arms in order. A card must read idle
before an arm starts. Every arm keeps its log and report under ``reports/suite/<box>/`` and a line
in ``summary.jsonl``. A box's paths and cards come from ``boxes.json`` (``TOKENOMOS_BOXES``; shape
in ``boxes.example.json``).

    python bench/suite/run.py --box 5090
"""

import argparse
import contextlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from arms import BOXES, Arm

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
HERE = Path(__file__).parent
CONFIG = Path(os.environ.get("TOKENOMOS_BOXES", HERE / "boxes.json"))
BUSY_MIB = 300
WAIT_IDLE_S = 900


class Box:
    def __init__(self, name: str):
        self.name = name
        self.out = ROOT / "reports" / "suite" / name
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        conf = json.loads(CONFIG.read_text("utf-8"))[name]
        self.models, self.rows, self.cards = conf["models"], conf["rows"], conf["cards"]
        self.cap = conf["cap"]
        self.kohaku, self.gbc, self.hub = conf.get("kohaku"), conf.get("gbc"), conf.get("hub")
        base = {"HF_HUB_OFFLINE": "1", "PYTHONPATH": conf["pythonpath"], **conf.get("env", {})}
        if "path" in conf:
            base["PATH"] = f"{conf['path']}{os.pathsep}{os.environ['PATH']}"
        self.env = {**os.environ, **base, "TOKENOMOS_MODELS": self.models}

    def model(self, name: str) -> str:
        """A folder of the box's models, or a snapshot of a hub ``org--name``."""
        if "--" in name:
            top = Path(self.hub) / f"models--{name}" / "snapshots"
            return str(next(top.iterdir()))
        return str(Path(self.models) / name)

    def python(self, *argv: str) -> list[str]:
        if self.cap:
            return [PY, str(HERE / "capped.py"), str(self.cap), *argv]
        return [PY, *argv]

    def record(self, row: dict) -> None:
        with self.lock, (self.out / "summary.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


def smi(card: int) -> int:
    """MiB in use on ``card``."""
    out = subprocess.run(
        [
            "nvidia-smi",
            f"--id={card}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip())


def wait_idle(card: int, baseline: int) -> bool:
    end = time.time() + WAIT_IDLE_S
    while time.time() < end:
        if smi(card) < baseline + BUSY_MIB:
            return True
        time.sleep(5)
    return False


def kill(p: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(p.pid, signal.SIGKILL)


def clear_card(card: int) -> None:
    """vLLM's engine core leaves its parent's process group: whatever still holds the card of an
    arm's own user goes."""
    if os.name == "nt":
        return
    out = subprocess.run(
        ["nvidia-smi", f"--id={card}", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    for pid in out.stdout.split():
        if Path(f"/proc/{pid}").stat().st_uid == os.getuid():
            os.kill(int(pid), signal.SIGKILL)


def spawn(cmd: list[str], env: dict, log: Path) -> subprocess.Popen:
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
    if os.name != "nt":
        kw["start_new_session"] = True
    f = log.open("a", encoding="utf-8", errors="replace")
    return subprocess.Popen(
        cmd, env=env, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=f, **kw
    )


def call(cmd: list[str], env: dict, log: Path, limit: int) -> int:
    p = spawn(cmd, env, log)
    try:
        return p.wait(limit)
    except subprocess.TimeoutExpired:
        kill(p)
        return -9


def up(url: str, p: subprocess.Popen, limit: int = 1200) -> bool:
    end = time.time() + limit
    while time.time() < end and p.poll() is None:
        try:
            urllib.request.urlopen(url + "/v1/models", timeout=3).read()
            return True
        except OSError:
            time.sleep(3)
    return False


def report(prefix: str, tag: str) -> dict:
    path = ROOT / "reports" / f"{prefix}_{tag}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    skip = ("texts", "token_ids", "tap", "tap_prefill", "slowest_steps_ms", "buckets", "rows")
    return {k: v for k, v in data.items() if k not in skip}


def http(arm: Arm, box: Box, env: dict, log: Path, server: list[str], card: int) -> int:
    port = 8100 + card
    p = spawn(server + ["--port", str(port)], env, log)
    url = f"http://127.0.0.1:{port}"
    try:
        if not up(url, p):
            return -1
        client = [
            PY,
            "bench/serve_caption.py",
            "--url",
            url,
            "--rows",
            box.rows,
            "--n",
            "1024",
            "--concurrency",
            "1024",
            "--tag",
            arm.tag,
        ]
        return call(client, {**env, "PYTHONPATH": ""}, log, arm.limit)
    finally:
        kill(p)
        clear_card(card)


def vllm_env(arm: Arm, box: Box, env: dict) -> dict:
    env = {**env, **arm.env}
    if arm.kohaku:
        return {**env, "PYTHONPATH": box.kohaku}
    env.pop("PYTHONPATH", None)
    return {**env, "VLLM_PLUGINS": ""}


def execute(arm: Arm, box: Box, card: int) -> dict:
    log = box.out / f"{arm.tag}.log"
    env = {**box.env, "CUDA_VISIBLE_DEVICES": str(card)}
    env.update(arm.env if arm.kind not in ("vllm", "vllm_serve") else {})
    model = box.model(arm.model) if arm.model else ""
    arch, engine = json.dumps(arm.arch), json.dumps(arm.engine)
    t0 = time.time()
    prefix, code = "", 0
    if arm.kind == "tests":
        code = call(box.python("-m", "pytest", "tests", "-q", "-x"), env, log, arm.limit * 2)
    elif arm.kind == "tune":
        script = "bench/profile/gdn_kernel.py"
        code = call(box.python(script, "--tag", arm.tag, "--write"), env, log, arm.limit)
    elif arm.kind == "decode":
        prefix = "decode"
        argv = [
            "bench/decode.py",
            f"--model={model}",
            f"--n={arm.seqs}",
            "--gen=128",
            f"--prefill-n={arm.prefill_seqs}",
            f"--arch-kw={arch}",
            f"--engine-kw={engine}",
            f"--tag={arm.tag}",
        ]
        code = call(box.python(*argv), env, log, arm.limit)
    elif arm.kind == "caption":
        prefix = "caption"
        argv = [
            "bench/caption.py",
            f"--model={model}",
            f"--rows={box.rows}",
            "--n=1024",
            f"--arch-kw={arch}",
            f"--engine-kw={engine}",
            f"--tag={arm.tag}",
        ]
        code = call(box.python(*argv), env, log, arm.limit)
    elif arm.kind == "serve":
        prefix = "serve_caption"
        name = "tokenomos.api.server"
        argv = ["-m", name, "--model", model, "--arch-kw", arch, "--engine-kw", engine]
        code = http(arm, box, env, log, box.python(*argv), card)
    elif arm.kind == "vllm":
        prefix = "vllm_caption"
        out = ROOT / "reports" / f"vllm_caption_{arm.tag}.json"
        argv = [
            PY,
            "bench/suite/vllm_caption.py",
            "--model",
            model,
            "--rows",
            box.rows,
            "--out",
            str(out),
        ]
        code = call(argv, vllm_env(arm, box, env), log, arm.limit)
        clear_card(card)
    elif arm.kind == "vllm_serve":
        prefix = "serve_caption"
        common = [
            "vllm",
            "serve",
            model,
            "--max-num-seqs",
            "256",
            "--gpu-memory-utilization",
            "0.90",
            "--max-model-len",
            "4096",
            "--mamba-ssm-cache-dtype",
            "bfloat16",
            "--trust-remote-code",
        ]
        code = http(arm, box, vllm_env(arm, box, env), log, common, card)
    elif arm.kind == "gbc":
        cmd = ["bash", box.gbc, str(card), arm.tag, arm.model, arm.arch["json"], "1200"]
        code = call(cmd, {**env, "LIMIT": "1500", "GBC_MAX_JOBS": "192"}, log, arm.limit)
    return {
        "arm": arm.tag,
        "card": card,
        "exit": code,
        "seconds": round(time.time() - t0, 1),
        **report(prefix, arm.tag),
    }


def run_class(box: Box, name: str, arms: list[Arm]) -> None:
    cards = box.cards[name]
    baseline = {c: smi(c) if box.cap else 0 for c in cards}
    for arm in (a for a in arms if a.kind in ("tests", "tune")):
        if not wait_idle(cards[0], baseline[cards[0]]):
            box.record({"arm": arm.tag, "skipped": "card busy"})
            return
        row = execute(arm, box, cards[0])
        box.record(row)
        if row["exit"]:
            box.record({"class": name, "stopped": arm.tag})
            return
    todo: queue.SimpleQueue = queue.SimpleQueue()
    for arm in arms:
        if arm.kind not in ("tests", "tune"):
            todo.put(arm)

    def work(card: int) -> None:
        while True:
            try:
                arm = todo.get_nowait()
            except queue.Empty:
                return
            if not wait_idle(card, baseline[card]):
                box.record({"arm": arm.tag, "card": card, "skipped": "card busy"})
                continue
            box.record(execute(arm, box, card))

    threads = [threading.Thread(target=work, args=(c,)) for c in cards]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    box.record({"class": name, "done": True})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", required=True, choices=sorted(BOXES))
    ap.add_argument("--only", default="", help="arms whose tag holds one of these, comma separated")
    args = ap.parse_args()
    box = Box(args.box)
    arms = BOXES[args.box]()
    if args.only:
        keep = args.only.split(",")
        arms = [a for a in arms if any(k in a.tag for k in keep)]
    classes = list(box.cards)
    threads = [
        threading.Thread(target=run_class, args=(box, c, [a for a in arms if a.card == c]))
        for c in classes
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("SUITE DONE", flush=True)


if __name__ == "__main__":
    main()
