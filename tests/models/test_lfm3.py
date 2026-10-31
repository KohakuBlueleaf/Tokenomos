"""LFM3 text stack against the checkpoint's own transformers code.

Unit checks: the attention mask against transformers' sliding-window mask function, RoPE
against the checkpoint's rotary class and ``apply_rotary_pos_emb``, exactly.

Teacher-forced parity on natural text (model card and license), 24 / 300 / 700 tokens (past the
256-token sliding window), packed into one forward, measured against transformers in float32:
per token, argmax agreement and the log-prob gap on the truth's argmax. Ours in bf16 must be as
close to it as transformers' own bf16 run. Negative controls (no window, wrong RoPE theta) must
move the gap well past ours. MXFP8: fused producers no less accurate than unfused, plus the
bf16-activation reference. Every number goes to ``reports/lfm3_parity.json``.
"""

import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, masking_utils
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from tokenomos import models as tk_models
from tokenomos.models import reference
from tokenomos.models.blocks.attention import Attention, mask
from tokenomos.models.blocks.rope import Rope
from tokenomos.models.contract import Step
from tokenomos.models.lfm3.arch import Lfm3

pytestmark = [pytest.mark.gpu, pytest.mark.model]
DEV = "cuda"
REPORT = Path(__file__).resolve().parents[2] / "reports" / "lfm3_parity.json"
LENGTHS = [24, 300, 700]


def write(key: str, result: dict) -> None:
    REPORT.parent.mkdir(exist_ok=True)
    data = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    data[key] = result
    REPORT.write_text(json.dumps(data, indent=1))
    print(key, result)


def compare(ours: list[torch.Tensor], ref: list[torch.Tensor]) -> dict:
    agree, gaps = [], []
    for a, b in zip(ours, ref):
        la, lb = a.log_softmax(-1), b.log_softmax(-1)
        top = lb.argmax(-1)
        agree.append((la.argmax(-1) == top).float())
        gaps.append((la.gather(-1, top[:, None]) - lb.gather(-1, top[:, None])).abs()[:, 0])
    agree, gaps = torch.cat(agree), torch.cat(gaps)
    return {
        "tokens": agree.numel(),
        "argmax_agreement": round(agree.mean().item(), 5),
        "logprob_gap_mean": round(gaps.mean().item(), 5),
        "logprob_gap_p99": round(gaps.quantile(0.99).item(), 5),
        "logprob_gap_max": round(gaps.max().item(), 5),
    }


@pytest.fixture(scope="module")
def bf16_path(models):
    return models / "lfm3-vl-2b"


@pytest.fixture(scope="module")
def seqs(bf16_path):
    tok = AutoTokenizer.from_pretrained(bf16_path, trust_remote_code=True)
    text = "\n".join((bf16_path / n).read_text("utf-8") for n in ("README.md", "LICENSE"))
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) >= sum(LENGTHS), len(ids)
    out, at = [], 0
    for n in LENGTHS:
        out.append(ids[at : at + n])
        at += n
    return out


@pytest.fixture(scope="module")
def hf_logits(bf16_path, seqs) -> dict[str, list[torch.Tensor]]:
    """The transformers model's logits in bf16 (its eager attention) and in float32 (the truth
    both bf16 implementations are measured against)."""
    out = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        hf = AutoModelForImageTextToText.from_pretrained(
            bf16_path, trust_remote_code=True, dtype=dtype, attn_implementation="eager"
        ).to(DEV)
        with torch.no_grad():
            out[name] = [
                hf(input_ids=torch.tensor([s], device=DEV)).logits[0].float() for s in seqs
            ]
        del hf
        torch.cuda.empty_cache()
    return out


def test_mask_vs_transformers():
    for window in (None, 256, 7):
        fn = (
            masking_utils.causal_mask_function
            if window is None
            else masking_utils.sliding_window_causal_mask_function(window)
        )
        n = 600
        idx = torch.arange(n)
        zero = torch.zeros((), dtype=torch.long)
        ref = fn(zero, zero, idx[:, None], idx[None, :]).expand(n, n)
        assert torch.equal(mask(n, window, "cpu"), ref)


def test_rope_vs_transformers(bf16_path):
    rotary_cls = get_class_from_dynamic_module("modeling_lfm3.Lfm3RotaryEmbedding", str(bf16_path))
    apply = sys.modules[rotary_cls.__module__].apply_rotary_pos_emb
    cfg = AutoConfig.from_pretrained(bf16_path, trust_remote_code=True).text_config
    hf_rot = rotary_cls(cfg).to(DEV)
    t, d = 900, cfg.head_dim
    positions = torch.arange(t, device=DEV) * 37
    q = torch.randn(1, 16, t, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, 4, t, d, device=DEV, dtype=torch.bfloat16)
    for kind, params in cfg.rope_parameters.items():
        cos, sin = hf_rot(q, positions[None], kind)
        rq, rk = apply(q, k, cos, sin)
        angles = int(params.get("partial_rotary_factor", 1.0) * d) // 2
        ours = Rope(d, angles, params["rope_theta"], DEV)
        step = Step(positions, *(None,) * 5)
        oq, ok = ours(step, q[0].transpose(0, 1), k[0].transpose(0, 1))
        assert torch.equal(oq, rq[0].transpose(0, 1)), kind
        assert torch.equal(ok, rk[0].transpose(0, 1)), kind


@torch.no_grad()
def test_bf16_vs_transformers(bf16_path, seqs, hf_logits):
    """Ours in bf16 is as close to the float32 truth as transformers' own bf16 run is."""
    truth = hf_logits["fp32"]
    hf_bf16 = compare(hf_logits["bf16"], truth)
    write("transformers_bf16_vs_fp32", hf_bf16)
    model = tk_models.load(str(bf16_path), DEV)
    ours = reference.forward_logits(model, seqs, DEV)
    good = compare(ours, truth)
    write("bf16_vs_fp32", good)
    write("bf16_vs_transformers_bf16", compare(ours, hf_logits["bf16"]))
    assert good["logprob_gap_mean"] <= 1.25 * hf_bf16["logprob_gap_mean"]
    assert good["argmax_agreement"] >= hf_bf16["argmax_agreement"] - 0.01

    attn = [layer.mixer for layer in model.layers if isinstance(layer.mixer, Attention)]
    sliding = [a for a in attn if a.window is not None]
    for a in sliding:
        a.window = None
    no_window = compare(reference.forward_logits(model, seqs, DEV), truth)
    for a in sliding:
        a.window = 256
    write("control_no_window", no_window)

    full = [a for a in attn if a.window is None]
    saved = [a.rope for a in full]
    wrong = Rope(128, 32, 10_000.0, DEV)
    for a in full:
        a.rope = wrong
    wrong_theta = compare(reference.forward_logits(model, seqs, DEV), truth)
    for a, r in zip(full, saved):
        a.rope = r
    write("control_wrong_theta", wrong_theta)
    assert no_window["logprob_gap_mean"] > 2 * good["logprob_gap_mean"]
    assert wrong_theta["logprob_gap_mean"] > 2 * good["logprob_gap_mean"]


@pytest.mark.sm100
@torch.no_grad()
def test_mxfp8(models, seqs, hf_logits):
    """The MXFP8 checkpoint: every quantized linear on the MXFP8 path, close to the float32
    truth, and the fused producers no less accurate than quantizing each input separately."""
    truth = hf_logits["fp32"]
    path = models / "lfm3-vl-2b-mxfp8"
    model = tk_models.load(str(path), DEV)
    assert reference.mx_layers(model) == 13 * 2 + 11 * 2 + 24 * 2
    fused = reference.forward_logits(model, seqs, DEV)
    vs_truth = compare(fused, truth)
    write("mxfp8_fused_vs_fp32", vs_truth)
    unfused_model = tk_models.load(str(path), DEV, Lfm3(fuse_quant=False))
    unfused = compare(reference.forward_logits(unfused_model, seqs, DEV), truth)
    del unfused_model
    write("mxfp8_unfused_vs_fp32", unfused)
    mx_head_model = tk_models.load(str(path), DEV, Lfm3(head="mxfp8"))
    mx_head = compare(reference.forward_logits(mx_head_model, seqs, DEV), truth)
    del mx_head_model
    write("mxfp8_head_mxfp8_vs_fp32", mx_head)
    reference.dequantize(model)
    deq = compare(reference.forward_logits(model, seqs, DEV), truth)
    write("mxfp8_weights_bf16_activations_vs_fp32", deq)
    assert vs_truth["argmax_agreement"] >= 0.9
    assert vs_truth["logprob_gap_mean"] <= 1.1 * unfused["logprob_gap_mean"]
    assert mx_head["argmax_agreement"] >= vs_truth["argmax_agreement"] - 0.02
    assert mx_head["logprob_gap_mean"] <= 1.25 * vs_truth["logprob_gap_mean"]
