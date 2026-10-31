"""Run a script or module under a hard allocator cap: ``capped.py GIB SCRIPT args`` or
``capped.py GIB -m MODULE args``, for a card shared with a desktop."""

import runpy
import sys
from pathlib import Path

import torch

if __name__ == "__main__":
    gib, target = float(sys.argv[1]), sys.argv[2]
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(gib * (1 << 30) / total, 0)
    if target == "-m":
        module = sys.argv[3]
        sys.argv = [module, *sys.argv[4:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    else:
        sys.argv = sys.argv[2:]
        sys.path.insert(0, str(Path(target).parent))
        runpy.run_path(target, run_name="__main__")
