"""The work dtype: names, the device default, no CUDA at import, and the refusal of quantized
formats under fp16."""

import subprocess
import sys

import pytest
import torch

from tokenomos.core import floats
from tokenomos.models.blocks.linear import Linear


@pytest.fixture
def fp16():
    floats.set_work("fp16")
    yield
    floats.set_work("bf16")


def test_set_work_names():
    with pytest.raises(ValueError, match="work dtype"):
        floats.set_work("fp64")


def test_fp32_keeps_fp16_weights():
    floats.set_work("fp32")
    try:
        assert floats.work() == torch.float32
        assert floats.weights() == torch.float16
    finally:
        floats.set_work("bf16")
    assert floats.weights() == torch.bfloat16


def test_default_by_capability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (7, 5))
    assert floats.default("cuda:0") == "fp16"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (8, 0))
    assert floats.default("cuda:0") == "bf16"
    assert floats.default("cpu") == "bf16"


def test_import_leaves_cuda_alone():
    code = (
        "import importlib.util as u, sys, torch\n"
        "s = u.spec_from_file_location('floats', sys.argv[1])\n"
        "s.loader.exec_module(u.module_from_spec(s))\n"
        "assert not torch.cuda.is_initialized()\n"
    )
    subprocess.run([sys.executable, "-c", code, floats.__file__], check=True)


@pytest.mark.usefixtures("fp16")
def test_quantized_refused_under_fp16():
    part = {"fmt": "mxfp8", "name": "x", "weight": torch.zeros(4, 4), "bias": None}
    with pytest.raises(ValueError, match="bf16 work dtype"):
        Linear._fuse([part])


@pytest.mark.usefixtures("fp16")
def test_weights_take_the_work_dtype():
    part = {"fmt": "bf16", "name": "x", "weight": torch.ones(4, 4), "bias": torch.ones(4)}
    lin = Linear._fuse([part])
    assert lin.weight.dtype == lin.bias.dtype == torch.float16
