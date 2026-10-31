"""Prepared images in fixed slots of one shared, page-locked host buffer.

The serving process maps a file under ``/dev/shm`` (the temp folder where there is none, as on
Windows) and registers the mapping with CUDA, so a
slot's view copies to the device without blocking the host; prep workers map the same file and
write resized images straight into slots. Free slot ids travel in a process queue: a worker takes
one per image (blocking while none is free), the serving process returns it once the engine no
longer reads the image.
"""

import mmap
import os
import tempfile
import uuid
from pathlib import Path

import torch


def _dir() -> str:
    return "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()


class ImageRing:
    def __init__(self, slots: int, slot_bytes: int, free, path: str | None = None, pin=True):
        """Owner side when ``path`` is None (creates, fills ``free`` with every slot id, pins
        unless ``pin`` is False); worker side otherwise (maps ``path``)."""
        self.slots, self.slot_bytes, self.free = slots, slot_bytes, free
        self.owner, self.pinned = path is None, pin and path is None
        self.path = path or str(Path(_dir()) / f"tokenomos-ring-{os.getpid()}-{uuid.uuid4().hex}")
        size = slots * slot_bytes
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | (os.O_CREAT if self.owner else 0)
        fd = os.open(self.path, flags, 0o600)
        try:
            if self.owner:
                os.ftruncate(fd, size)
            self.map = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        self.buf = torch.frombuffer(self.map, dtype=torch.uint8)
        if self.pinned:
            err = torch.cuda.cudart().cudaHostRegister(self.buf.data_ptr(), size, 0)
            if int(err) != 0:
                raise RuntimeError(f"cudaHostRegister failed: {err}")
        if self.owner:
            for s in range(slots):
                free.put(s)

    def args(self) -> tuple:
        """What a worker passes back to the constructor."""
        return self.slots, self.slot_bytes, self.free, self.path

    def write(self, x: torch.Tensor) -> tuple[int, tuple[int, ...]]:
        """Worker side: ``x`` into a free slot -> (slot, shape)."""
        n = x.numel() * x.element_size()
        if n > self.slot_bytes:
            raise ValueError(f"prepared image of {n} bytes over the slot size {self.slot_bytes}")
        slot = self.free.get()
        dst = self.buf[slot * self.slot_bytes : slot * self.slot_bytes + n]
        dst.view(x.dtype).view(x.shape).copy_(x)
        return slot, tuple(x.shape)

    def view(self, slot: int, shape, dtype=torch.uint8) -> torch.Tensor:
        n = dtype.itemsize
        for d in shape:
            n *= d
        base = slot * self.slot_bytes
        return self.buf[base : base + n].view(dtype).view(shape)

    def release(self, slots) -> None:
        for s in slots:
            self.free.put(s)

    def close(self) -> None:
        if self.pinned:
            torch.cuda.cudart().cudaHostUnregister(self.buf.data_ptr())
        del self.buf
        self.map.close()
        # after the unmap: Windows refuses to remove a mapped file
        if self.owner:
            Path(self.path).unlink(missing_ok=True)
