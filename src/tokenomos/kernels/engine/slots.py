"""The engine's per-step bookkeeping kernels over the slot table (all device-side, graphable).

- ``alloc``: give each row's slot enough pages for ``target`` positions, popped from the
  device free stack (``stack[:top]``) in row order;
- ``post``: after sampling, the sampled token into ``last_tok`` / ``out_tok``, counters, and the
  finish test (a stop token or ``max_new`` tokens) clearing ``alive``;
- ``release``: finished slots' pages pushed back onto the stack;
- ``compact``: the active list keeps its alive slots, in order;
- ``append``: slots staged by the host join the active list;
- ``push``: pages the host hands back (an evicted prefix) onto the stack.

Slot 0 is the null slot: never alive, rows pointing at it change nothing. Allocation and release
are single programs with prefix sums, so page assignment is deterministic.
"""

import triton
import triton.language as tl


@triton.jit
def _alloc_kernel(
    slot_ptr,
    target_ptr,
    n_ptr,
    alive,
    table,
    nblocks,
    stack,
    top_ptr,
    stride_t,
    P: tl.constexpr,
    MAXNEED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.arange(0, BLOCK)
    rm = r < tl.load(n_ptr)
    s = tl.load(slot_ptr + r, mask=rm, other=0)
    live = rm & (s > 0) & (tl.load(alive + s, mask=rm, other=0) != 0)
    have = tl.load(nblocks + s, mask=live, other=0)
    want = (tl.load(target_ptr + r, mask=live, other=0) + P - 1) // P
    need = tl.where(live, tl.maximum(want - have, 0), 0)
    excl = tl.cumsum(need, 0) - need
    top = tl.load(top_ptr)
    row = table + s.to(tl.int64) * stride_t + have
    if MAXNEED:
        for j in tl.static_range(MAXNEED):
            take = need > j
            tl.store(row + j, tl.load(stack + top - 1 - excl - j, mask=take, other=0), mask=take)
    else:
        for j in range(tl.max(need, 0)):
            take = need > j
            tl.store(row + j, tl.load(stack + top - 1 - excl - j, mask=take, other=0), mask=take)
    tl.store(nblocks + s, have + need, mask=live)
    tl.store(top_ptr, top - tl.sum(need, 0))


@triton.jit
def _post_kernel(
    tok_ptr,
    slot_ptr,
    n_ptr,
    alive,
    last_tok,
    out_tok,
    n_new,
    max_new,
    seq_len,
    stops,
    adv,
    stride_o,
    NSTOP: tl.constexpr,
):
    r = tl.program_id(0)
    if r < tl.load(n_ptr):
        s = tl.load(slot_ptr + r)
        if s > 0:  # noqa: SIM102 - the alive load must not run for the null slot
            if tl.load(alive + s) != 0:
                tok = tl.load(tok_ptr + r)
                n = tl.load(n_new + s)
                tl.store(last_tok + s, tok)
                tl.store(out_tok + s.to(tl.int64) * stride_o + n, tok)
                n += 1
                tl.store(n_new + s, n)
                tl.store(seq_len + s, tl.load(seq_len + s) + adv)
                st = tl.load(stops + s * NSTOP + tl.arange(0, NSTOP))
                hit = tl.sum((st == tok).to(tl.int32)) > 0
                if hit | (n >= tl.load(max_new + s)):
                    tl.store(alive + s, 0)


@triton.jit
def _release_kernel(
    list_ptr, n_ptr, alive, table, nblocks, shared, stack, top_ptr, stride_t, BLOCK: tl.constexpr
):
    r = tl.arange(0, BLOCK)
    rm = r < tl.load(n_ptr)
    s = tl.load(list_ptr + r, mask=rm, other=0)
    dead = rm & (s > 0) & (tl.load(alive + s, mask=rm, other=1) == 0)
    first = tl.load(shared + s, mask=dead, other=0)
    own = tl.where(dead, tl.maximum(tl.load(nblocks + s, mask=dead, other=0) - first, 0), 0)
    at = tl.load(top_ptr) + tl.cumsum(own, 0) - own
    for j in range(tl.max(own, 0)):
        take = own > j
        page = tl.load(table + s.to(tl.int64) * stride_t + first + j, mask=take, other=0)
        tl.store(stack + at + j, page, mask=take)
    tl.store(nblocks + s, 0, mask=dead)
    tl.store(shared + s, 0, mask=dead)
    tl.store(top_ptr, tl.load(top_ptr) + tl.sum(own, 0))


@triton.jit
def _push_kernel(pages, count_ptr, stack, top_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    c = tl.load(count_ptr)
    top = tl.load(top_ptr)
    tl.store(stack + top + i, tl.load(pages + i, mask=i < c, other=0), mask=i < c)
    tl.store(top_ptr, top + c)


@triton.jit
def _compact_kernel(list_ptr, n_ptr, alive, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    rm = r < tl.load(n_ptr)
    s = tl.load(list_ptr + r, mask=rm, other=0)
    keep = rm & (s > 0) & (tl.load(alive + s, mask=rm, other=0) != 0)
    k = keep.to(tl.int32)
    at = tl.cumsum(k, 0) - k
    tl.debug_barrier()
    tl.store(list_ptr + r, 0, mask=rm)
    tl.debug_barrier()
    tl.store(list_ptr + at, s, mask=keep)
    tl.store(n_ptr, tl.sum(k, 0))


@triton.jit
def _append_kernel(list_ptr, n_ptr, staged, count_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    c = tl.load(count_ptr)
    n = tl.load(n_ptr)
    tl.store(list_ptr + n + i, tl.load(staged + i, mask=i < c, other=0), mask=i < c)
    tl.store(n_ptr, n + c)
    tl.store(count_ptr, 0)


def _block(n: int) -> int:
    return max(16, triton.next_power_of_2(n))


def alloc(slot, target, n, st, maxneed: int = 0):
    """Rows ``0 .. n[0] - 1``: ``slot`` [rows], ``target`` [rows] positions to hold; at most
    ``maxneed`` new pages per row (0: any, in a loop over the batch's largest need)."""
    _alloc_kernel[(1,)](
        slot,
        target,
        n,
        st.alive,
        st.table,
        st.nblocks,
        st.stack,
        st.top,
        st.table.stride(0),
        P=st.page,
        MAXNEED=maxneed,
        BLOCK=_block(slot.shape[0]),
        num_warps=4,
    )


def post(tok, slot, n, st, adv: int):
    """``st``: the slot table (``SlotTable``); ``adv``: positions each row's sequence gained."""
    _post_kernel[(slot.shape[0],)](
        tok,
        slot,
        n,
        st.alive,
        st.last_tok,
        st.out_tok,
        st.n_new,
        st.max_new,
        st.seq_len,
        st.stops,
        adv,
        st.out_tok.stride(0),
        NSTOP=st.stops.shape[1],
        num_warps=1,
    )


def release(lst, n, st):
    """Pages of the list's dead slots (past their ``shared`` prefix) back onto the stack."""
    _release_kernel[(1,)](
        lst,
        n,
        st.alive,
        st.table,
        st.nblocks,
        st.shared,
        st.stack,
        st.top,
        st.table.stride(0),
        BLOCK=_block(lst.shape[0]),
        num_warps=4,
    )


def push(pages, count, st):
    """``pages[:count[0]]`` back onto the free stack (a prefix entry's pages, evicted)."""
    _push_kernel[(1,)](pages, count, st.stack, st.top, BLOCK=_block(pages.shape[0]), num_warps=4)


def compact(lst, n, alive):
    _compact_kernel[(1,)](lst, n, alive, BLOCK=_block(lst.shape[0]), num_warps=4)


def append(lst, n, staged, count):
    _append_kernel[(1,)](lst, n, staged, count, BLOCK=_block(staged.shape[0]), num_warps=4)
