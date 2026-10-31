"""Varlen short conv and its state update against ``F.conv1d`` on each sequence alone, with the
sequence's initial state (or zeros) prepended; with ``xmul`` the input is the bf16 product of two
strided column slices."""

import pytest
import torch
import torch.nn.functional as F

from tokenomos.kernels.conv.varlen import update_state, varlen_conv

pytestmark = [pytest.mark.gpu, pytest.mark.bf16]
DEV = "cuda"


@pytest.mark.parametrize("width", [4, 7])
@pytest.mark.parametrize("silu", [False, True])
@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("mul", [False, True])
def test_varlen_conv(width, silu, dim_first, with_bias, mul):
    dim = 512
    lengths = [1, 3, 9, 1, 40, 2]
    inits = [1, 0, 1, 1, 0, 1]
    s = len(lengths)
    slots = torch.tensor([5, 0, 2, 7, 3, 1], dtype=torch.int32, device=DEV)
    if dim_first:
        state = torch.randn(8, dim, width - 1, dtype=torch.bfloat16, device=DEV)
    else:
        # a [slots, W - 1, dim] buffer seen as [slots, dim, W - 1]: strided state rows
        state = torch.randn(8, width - 1, dim, dtype=torch.bfloat16, device=DEV).transpose(1, 2)
    before = state.clone()
    x = torch.randn(sum(lengths) + 5, dim, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(dim, width, dtype=torch.bfloat16, device=DEV) * 0.3
    bias = torch.randn(dim, dtype=torch.bfloat16, device=DEV) if with_bias else None
    cu_host = [0]
    for n in lengths:
        cu_host.append(cu_host[-1] + n)
    cu = torch.tensor(cu_host, dtype=torch.int32, device=DEV)
    lens = torch.tensor(lengths, device=DEV)
    tok_seq = torch.repeat_interleave(torch.arange(s, device=DEV), lens).to(torch.int32)
    tok_seq = torch.cat([tok_seq, torch.zeros(5, dtype=torch.int32, device=DEV)])
    init = torch.tensor(inits, dtype=torch.int32, device=DEV)
    n_tok = torch.tensor([cu_host[-1]], dtype=torch.int32, device=DEV)

    if mul:
        # x and its multiplier as column slices of one projection (token stride 3 * dim)
        proj = torch.randn(x.shape[0], 3 * dim, dtype=torch.bfloat16, device=DEV)
        xa, xm = proj[:, :dim], proj[:, 2 * dim :]
        out = varlen_conv(xa, state, w, bias, tok_seq, cu, slots, init, n_tok, silu, xmul=xm)
        update_state(xa, state, cu, slots, init, width, xmul=xm)
        x = xa * xm
    else:
        out = varlen_conv(x, state, w, bias, tok_seq, cu, slots, init, n_tok, silu)
        update_state(x, state, cu, slots, init, width)

    for i in range(s):
        a, b = cu_host[i], cu_host[i + 1]
        prev = before[slots[i]] if inits[i] else torch.zeros(dim, width - 1, device=DEV)
        full = torch.cat([prev.float(), x[a:b].float().t()], dim=1)
        ref = F.conv1d(full[None], w.float()[:, None, :], groups=dim)[0].t()
        if bias is not None:
            ref = ref + bias.float()
        if silu:
            ref = F.silu(ref)
        torch.testing.assert_close(out[a:b].float(), ref, atol=2e-2, rtol=2e-2)
        assert torch.equal(state[slots[i]], full[:, -(width - 1) :].to(torch.bfloat16))
    assert torch.equal(out[cu_host[-1] :], torch.zeros_like(out[cu_host[-1] :]))
    untouched = [r for r in range(8) if r not in slots.tolist()]
    assert torch.equal(state[untouched], before[untouched])
