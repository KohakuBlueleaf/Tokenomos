"""The fused GDN prefill against the gated delta rule in float64, token by token, over sequences
cut on and off every chunk size, with and without an initial state; a sequence split across two
calls against one call; the fp32 output path; each ``BT``; FLA's own error as the yardstick; and
a real Qwen3.5 on ``fused`` against ``fla``.
"""

import pytest
import torch
import torch.nn.functional as F

from tokenomos import models as tk_models
from tokenomos.kernels.gdn import prefill as gp
from tokenomos.kernels.gdn.fla import FLA_CHUNK_SIZE
from tokenomos.kernels.gdn.fla import chunk_gated_delta_rule as fla_rule
from tokenomos.kernels.gdn.fla.index import chunk_indices_host
from tokenomos.models import reference

pytestmark = pytest.mark.gpu
DEV = "cuda"
HK, HV, K, V = 2, 4, 128, 128
LENGTHS = [1, 15, 16, 17, 40, 64, 65, 200]
SCALE = K**-0.5
QWEN = ("qwen35-4b-bf16", "qwen35-4b-mlp4-bf16")
TIE = 0.15


def inputs(dtype, lengths):
    n = sum(lengths)
    q = F.normalize(torch.randn(n, HK, K, device=DEV), dim=-1).to(dtype)
    k = F.normalize(torch.randn(n, HK, K, device=DEV), dim=-1).to(dtype)
    v = torch.randn(n, HV, V, device=DEV).to(dtype)
    g = -F.softplus(torch.randn(n, HV, device=DEV) - 2.0)
    beta = torch.sigmoid(torch.randn(n, HV, device=DEV))
    h0 = torch.randn(len(lengths), HV, V, K, device=DEV) * 0.2
    h0[::2] = 0
    return q, k, v, g, beta, h0


def offsets(lengths) -> list[int]:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return cu


def recurrence(q, k, v, g, beta, h0, cu):
    """The rule token by token in float64: (output [T, Hv, V], final states)."""
    r = HV // HK
    o = torch.empty(q.shape[0], HV, V, dtype=torch.float64, device=DEV)
    finals = []
    for s in range(len(cu) - 1):
        h = h0[s].double()
        for t in range(cu[s], cu[s + 1]):
            kt = k[t].double().repeat_interleave(r, 0)
            qt = q[t].double().repeat_interleave(r, 0) * SCALE
            h = h * g[t].double().exp()[:, None, None]
            d = (v[t].double() - (h * kt[:, None]).sum(-1)) * beta[t].double()[:, None]
            h = h + d[..., None] * kt[:, None]
            o[t] = (h * qt[:, None]).sum(-1)
        finals.append(h)
    return o, torch.stack(finals)


def fused(q, k, v, g, beta, h0, cu, bt, out=None):
    rows, starts = gp.table(cu, bt)
    bv, warps, stages = gp.config(torch.device(DEV))[1:]
    return gp.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        SCALE,
        h0,
        torch.tensor(cu, dtype=torch.int32, device=DEV),
        torch.tensor(rows, dtype=torch.int32, device=DEV).view(-1, 2),
        torch.tensor(starts, dtype=torch.int32, device=DEV),
        out,
        (bt, bv, warps, stages),
    )


def fla(q, k, v, g, beta, h0, cu):
    rows = chunk_indices_host(cu, FLA_CHUNK_SIZE)
    o, final = fla_rule(
        q[None],
        k[None],
        v[None],
        g[None],
        beta[None],
        scale=SCALE,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=DEV),
        chunk_indices=torch.tensor(rows, dtype=torch.int32, device=DEV).view(-1, 2),
    )
    return o.view(q.shape[0], HV, V), final


def error(got, ref) -> float:
    return ((got.double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("bt", gp.BT_SIZES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_against_the_recurrence(dtype, bt):
    cu = offsets(LENGTHS)
    args = inputs(dtype, LENGTHS)
    ro, rf = recurrence(*args, cu)
    o, final = fused(*args, cu, bt)
    fo, ff = fla(*args, cu)
    # the fp16 tensor-core blocks may round, no worse than FLA's own 16-bit path by much
    assert error(o, ro) <= max(3 * error(fo, ro), 3e-3)
    assert error(final, rf) <= max(3 * error(ff, rf), 3e-3)
    # a dropped gate, chunk or initial state is far past this
    assert error(o, ro) < 0.03 and error(final, rf) < 0.03


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a_sequence_split_across_calls(dtype):
    """Prefill resumes from a stored state: two calls equal one."""
    n, cut = 150, 57
    args = inputs(dtype, [n])
    q, k, v, g, beta, _ = args
    h0 = torch.randn(1, HV, V, K, device=DEV) * 0.2
    ro, rf = recurrence(q, k, v, g, beta, h0, [0, n])
    bt = gp.config(torch.device(DEV))[0]
    a, mid = fused(q[:cut], k[:cut], v[:cut], g[:cut], beta[:cut], h0, [0, cut], bt)
    b, final = fused(q[cut:], k[cut:], v[cut:], g[cut:], beta[cut:], mid, [0, n - cut], bt)
    assert error(torch.cat((a, b)), ro) < 0.03
    assert error(final, rf) < 0.03


def test_fp32_output():
    """Under fp32 the products are fp16 and the output float32, holding what fp16 cannot."""
    args = inputs(torch.float16, LENGTHS)
    cu = offsets(LENGTHS)
    ro, _ = recurrence(*args, cu)
    out = torch.empty(args[0].shape[0], HV, V, dtype=torch.float32, device=DEV)
    o, _ = fused(*args, cu, gp.config(torch.device(DEV))[0], out)
    assert o is out and error(o, ro) < 0.01
    assert (o != o.half().float()).float().mean() > 0.5


def test_table_starts():
    rows, starts = gp.table([0, 1, 17, 49], 16)
    assert rows == [[0, 0], [1, 0], [2, 0], [2, 1]]
    assert starts == [0, 1, 2]


@pytest.mark.model
@torch.no_grad()
def test_qwen_on_fused(models):
    """Teacher-forced logits on ``fused`` against ``fla``, 300 and 37 tokens on one model."""
    path = next((models / n for n in QWEN if (models / n).is_dir()), None)
    if path is None:
        pytest.skip("no bf16 Qwen3.5 4B")
    seqs = [torch.randint(1000, 30000, (n,)).tolist() for n in (300, 37)]
    logits = {}
    for name in ("fla", "fused"):
        conf = tk_models.configure(str(path), vision=False, gdn_prefill=name)
        model = tk_models.load(str(path), DEV, conf)
        assert {m.prefill for m in model.modules() if hasattr(m, "prefill")} == {name}
        logits[name] = reference.forward_logits(model, seqs, DEV)
        del model
    for a, b in zip(logits["fla"], logits["fused"]):
        pa, pb = a.log_softmax(-1), b.log_softmax(-1)
        top = pa.argmax(-1, keepdim=True)
        gap = pa.gather(-1, top) - pa.gather(-1, pb.argmax(-1, keepdim=True))
        # a random prompt's logits are flat: a token may flip, but only within a near-tie
        assert gap.max() <= TIE
        assert (pa.gather(-1, top) - pb.gather(-1, top)).abs().mean() < 0.1
