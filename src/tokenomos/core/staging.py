"""Pinned host arenas for host-to-device copies that never block the host.

Host iteration ``i`` bump-allocates from arena ``i % n``. Entering the next iteration records an
event behind the arena's last queued copy, and reusing the arena waits on that event, so a pending
copy never sees its source change, whatever else shares the stream. The event sits ahead of the
next decode step, so the wait is over by the time the host reads that step's predecessor.
``put`` returns the device tensor (the copy is queued on the current stream).
"""

import torch

ALIGN = 64


class Staging:
    def __init__(self, nbytes: int, device, n: int = 2):
        self.arenas = [torch.empty(nbytes, dtype=torch.uint8, pin_memory=True) for _ in range(n)]
        self.sealed = [torch.cuda.Event() for _ in range(n)]
        self.pending = [False] * n
        self.device, self.cur, self.at = device, 0, 0

    def begin(self, iteration: int) -> None:
        if self.at:
            self.sealed[self.cur].record()
            self.pending[self.cur] = True
        self.cur, self.at = iteration % len(self.arenas), 0
        if self.pending[self.cur]:
            self.sealed[self.cur].synchronize()
            self.pending[self.cur] = False

    def put(self, values, dtype: torch.dtype) -> torch.Tensor:
        src = torch.as_tensor(values, dtype=dtype)
        nb = src.numel() * dtype.itemsize
        arena = self.arenas[self.cur]
        if self.at + nb > arena.numel():
            raise RuntimeError(f"staging arena full ({arena.numel()} bytes)")
        view = arena[self.at : self.at + nb].view(dtype).view(src.shape)
        view.copy_(src)
        self.at += -(-nb // ALIGN) * ALIGN
        return view.to(self.device, non_blocking=True)
