"""JSON-schema grammar on the device: mask the logits by each row's grammar state, and step the
state through a transition table the host fills (``tokenomos-rust/grammar/device.rs`` defines
the states and entries).

Per slot: ``g_state`` (-1: free text), ``g_room`` / ``g_need`` (the open string's true counters,
room -1 unbounded), ``g_wait`` (1 while the next state is unknown), ``g_miss`` (-1; the token of
a missing ``(state, token)`` entry; ``-2 - shape`` for a missing ``(shape, counters)`` entry) and
``g_trace`` (the state each output token was sampled under). The table is open addressing over
int64 keys: ``state << 20 | token`` outside strings, ``SHAPE | shape << 20 | token`` inside one
(``shape_of[state]`` >= 0), and ``FLAG | shape << 32 | room << 16 | need`` (counters clamped to
``longest``); ``vals[i]`` is ``(next, stay, a, b)``. ``mask_of[state]`` is the row of ``masks``
(bitsets over the vocabulary) the state allows.
"""

import torch
import triton
import triton.language as tl

TOKEN_BITS = 20
FLAG = 1 << 62
SHAPE = 1 << 61
# 0x9E3779B97F4A7C15 as a signed int64
H_MUL = -7046029254386353131


def table_hash(key: int, cap_mask: int) -> int:
    """The host's copy of ``_probe``'s hash (int64 wrap, arithmetic shift)."""
    x = (key * H_MUL) & ((1 << 64) - 1)
    x -= (x >> 63) << 64
    return (x ^ (x >> 31)) & cap_mask


def key_token(state: int, shape: int, token: int) -> int:
    """The entry key of ``token`` from ``state`` (``shape``: its open string's, -1 outside)."""
    if shape >= 0:
        return SHAPE | shape << TOKEN_BITS | token
    return state << TOKEN_BITS | token


def key_counters(shape: int, room: int, need: int) -> int:
    return FLAG | shape << 32 | room << 16 | need


@triton.jit
def _probe(keys, vals, key, cap_mask, PROBES: tl.constexpr, H_MUL: tl.constexpr):
    """The entry index holding ``key``, or -1."""
    x = key * H_MUL
    h = (x ^ (x >> 31)) & cap_mask
    found = tl.full((), -1, tl.int64)
    ended = tl.full((), 0, tl.int32)
    for i in tl.static_range(PROBES):
        idx = (h + i) & cap_mask
        k = tl.load(keys + idx)
        hit = (k == key) & (found < 0) & (ended == 0)
        found = tl.where(hit, idx, found)
        ended = ended | (k == -1).to(tl.int32)
    return found


@triton.jit
def _mask_kernel(
    logits,
    stride,
    slot_ptr,
    g_state,
    mask_of,
    masks,
    words,
    trace,
    n_new,
    new_cap,
    V,
    BLOCK: tl.constexpr,
):
    r, blk = tl.program_id(0), tl.program_id(1)
    s = tl.load(slot_ptr + r)
    st = tl.load(g_state + s)
    if (s == 0) | (st < 0):
        return
    m = tl.load(mask_of + st)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    inside = offs < V
    w = tl.load(masks + m.to(tl.int64) * words + offs // 32, mask=inside, other=0)
    keep = ((w >> (offs % 32).to(tl.int32)) & 1) != 0
    p = logits + r.to(tl.int64) * stride + offs
    x = tl.load(p, mask=inside)
    tl.store(p, tl.where(keep, x, float("-inf")), mask=inside)
    if blk == 0:
        k = tl.minimum(tl.load(n_new + s), new_cap - 1)
        tl.store(trace + s.to(tl.int64) * new_cap + k, st)


@triton.jit
def _advance_kernel(
    tok,
    slot_ptr,
    g_state,
    g_room,
    g_need,
    g_wait,
    g_miss,
    shape_of,
    body_of,
    plain,
    keys,
    vals,
    cap_mask,
    longest,
    PROBES: tl.constexpr,
    TOKEN_BITS: tl.constexpr,
    FLAG: tl.constexpr,
    SHAPE: tl.constexpr,
    H_MUL: tl.constexpr,
):
    r = tl.program_id(0)
    s = tl.load(slot_ptr + r)
    st = tl.load(g_state + s)
    if (s == 0) | (st < 0):
        return
    t = tl.load(tok + r)
    sh = tl.load(shape_of + st)
    pc = tl.load(plain + t)
    if (sh >= 0) & (tl.load(body_of + st) != 0) & (pc >= 0):
        # a plain token in a string body: same shape, its characters off the counters
        nxt, a = sh, pc
        stay, b = tl.full((), 1, tl.int32), tl.full((), 0, tl.int32)
    else:
        t64, sh64 = t.to(tl.int64), sh.to(tl.int64)
        key = tl.where(sh >= 0, SHAPE | (sh64 << TOKEN_BITS), st.to(tl.int64) << TOKEN_BITS) | t64
        i = _probe(keys, vals, key, cap_mask, PROBES, H_MUL)
        if i < 0:
            tl.store(g_wait + s, 1)
            tl.store(g_miss + s, t)
            return
        nxt = tl.load(vals + i * 4)
        stay = tl.load(vals + i * 4 + 1)
        a = tl.load(vals + i * 4 + 2)
        b = tl.load(vals + i * 4 + 3)
    if stay == 0:
        tl.store(g_state + s, nxt)
        tl.store(g_room + s, a)
        tl.store(g_need + s, b)
        return
    room = tl.load(g_room + s)
    room = tl.where(room >= 0, room - a, room)
    need = tl.maximum(tl.load(g_need + s) - a, 0)
    tl.store(g_room + s, room)
    tl.store(g_need + s, need)
    cr = tl.where(room < 0, longest, tl.minimum(room, longest))
    cn = tl.minimum(need, longest)
    key = FLAG | (nxt.to(tl.int64) << 32) | (cr.to(tl.int64) << 16) | cn.to(tl.int64)
    j = _probe(keys, vals, key, cap_mask, PROBES, H_MUL)
    if j < 0:
        tl.store(g_wait + s, 1)
        tl.store(g_miss + s, -2 - nxt)
    else:
        tl.store(g_state + s, tl.load(vals + j * 4))


def apply(logits, slot, g_state, mask_of, masks, trace, n_new) -> None:
    """``logits`` [N, V] in place: row ``r`` keeps the tokens its slot's state allows (free-text
    rows untouched); records the state at ``trace[slot, n_new[slot]]``."""
    n, v = logits.shape
    block = 2048
    _mask_kernel[(n, triton.cdiv(v, block))](
        logits,
        logits.stride(0),
        slot,
        g_state,
        mask_of,
        masks,
        masks.shape[1],
        trace,
        n_new,
        trace.shape[1],
        v,
        BLOCK=block,
    )


def advance(tok, slot, st, g) -> None:
    """Each grammar row's state after its sampled token (``st``: the slot table's ``g_*``,
    ``g``: the ``DeviceGrammar``)."""
    _advance_kernel[(tok.shape[0],)](
        tok,
        slot,
        st.g_state,
        st.g_room,
        st.g_need,
        st.g_wait,
        st.g_miss,
        g.shape_of,
        g.body_of,
        g.plain,
        g.keys,
        g.vals,
        g.keys.shape[0] - 1,
        g.longest,
        PROBES=g.probes,
        TOKEN_BITS=TOKEN_BITS,
        FLAG=FLAG,
        SHAPE=SHAPE,
        H_MUL=H_MUL,
    )


def empty_table(bits: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    cap = 1 << bits
    return (
        torch.full((cap,), -1, dtype=torch.int64, device=device),
        torch.zeros(cap, 4, dtype=torch.int32, device=device),
    )
