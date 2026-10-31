"""Quantization scheme matching and the loader's refusal of unscaled float8 weights."""

import pytest
import torch

from tokenomos.models.blocks.linear import Linear
from tokenomos.models.load.checkpoint import QuantScheme

GROUP = {"format": "mxfp8-quantized"}


def scheme(targets, ignore=()):
    return QuantScheme(
        {
            "quant_method": "compressed-tensors",
            "config_groups": {"g": {**GROUP, "targets": targets}},
            "ignore": list(ignore),
        }
    )


def test_regex_targets():
    s = scheme([r"re:.*\.layers\.\d+\.(conv\.in_proj|feed_forward\.w1)$"])
    assert s.format_of("model.layers.3.conv.in_proj") == "mxfp8"
    assert s.format_of("model.layers.3.feed_forward.w1") == "mxfp8"
    assert s.format_of("model.layers.3.feed_forward.w2") == "bf16"
    assert s.format_of("model.layers.3.conv.in_proj_extra") == "bf16"


def test_class_target_and_ignore():
    s = scheme(["Linear"], ignore=["lm_head", r"re:.*gate_proj$"])
    assert s.format_of("model.layers.0.self_attn.q_proj") == "mxfp8"
    assert s.format_of("lm_head") == "bf16"
    assert s.format_of("model.layers.0.self_attn.gate_proj") == "bf16"


def test_unknown_format_rejected():
    with pytest.raises(ValueError):
        QuantScheme({"quant_method": "compressed-tensors", "config_groups": {"g": {"format": "x"}}})


class FakeCkpt:
    def __init__(self, tensors, quant):
        self.tensors, self.quant = tensors, quant

    def has(self, name):
        return name in self.tensors

    def get(self, name, device="cpu"):
        return self.tensors[name]


def test_float8_without_scheme_refused():
    w = torch.zeros(64, 128).to(torch.float8_e4m3fn)
    ckpt = FakeCkpt({"a.weight": w}, QuantScheme(None))
    with pytest.raises(ValueError):
        Linear.load(ckpt, ["a"], "cpu")


def test_bias_fused_and_partial():
    t = {
        "a.weight": torch.randn(4, 8),
        "a.bias": torch.randn(4),
        "b.weight": torch.randn(4, 8),
        "b.bias": torch.randn(4),
        "c.weight": torch.randn(4, 8),
    }
    ckpt = FakeCkpt(t, QuantScheme(None))
    lin = Linear.load(ckpt, ["a", "b"], "cpu")
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    ref = torch.cat(
        [x.float() @ t["a.weight"].t() + t["a.bias"], x.float() @ t["b.weight"].t() + t["b.bias"]],
        1,
    )
    torch.testing.assert_close(lin(x).float(), ref, atol=0.1, rtol=0.05)
    with pytest.raises(ValueError):
        Linear.load(ckpt, ["a", "c"], "cpu")
