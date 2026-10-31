import os
from pathlib import Path

import pytest
import torch

MODELS = Path(os.environ.get("TOKENOMOS_MODELS", "models"))
# marker -> lowest compute capability
NEEDS = {"bf16": (8, 0), "fp8": (8, 9), "sm100": (10, 0)}


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="needs a CUDA GPU")
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    for item in items:
        if "gpu" in item.keywords and cap is None:
            item.add_marker(skip_gpu)
        spec = getattr(item, "callspec", None)
        bf16 = spec is not None and any(v is torch.bfloat16 for v in spec.params.values())
        marks = [m for m in NEEDS if m in item.keywords] + ["bf16"] * bf16
        if cap is not None and any(cap < NEEDS[m] for m in marks):
            item.add_marker(pytest.mark.skip(reason=f"this GPU has no {marks[0]}"))


@pytest.fixture(scope="session")
def models() -> Path:
    if not MODELS.is_dir():
        pytest.skip(f"no model folder at {MODELS}")
    return MODELS


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
