"""An arena is reused only after every copy queued from it ran."""

import pytest
import torch

from tokenomos.core.staging import Staging

N = 4096


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_reused_arena_keeps_the_queued_copies_source():
    staging = Staging(1 << 16, "cuda", n=2)
    busy = torch.randn(8192, 8192, device="cuda")
    for _ in range(4):
        busy = torch.tanh(busy @ busy)
    staging.begin(0)
    first = staging.put(list(range(N)), torch.int64)
    staging.begin(1)
    staging.begin(2)
    staging.put([-1] * N, torch.int64)
    assert first.tolist() == list(range(N))
