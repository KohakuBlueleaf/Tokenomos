"""Bookkeeping kernels against a Python model of the slot table, over random episodes.

Each episode admits slots (append), allocates pages for growing targets, posts sampled tokens
(stop tokens and ``max_new`` finish slots), releases finished slots' pages and compacts the
active list, comparing every device array with the model after every operation. Invariant: the
free stack plus every slot's pages is exactly pages 1 .. P-1, each once.
"""

import random

import pytest
import torch

from tokenomos.core.state import NSTOP, SlotTable
from tokenomos.kernels.engine import slots as K

pytestmark = pytest.mark.gpu
DEV = "cuda"
PAGE = 4


class Model:
    def __init__(self, st: SlotTable):
        self.alive = [0] * (st.slots + 1)
        self.seq_len = [0] * (st.slots + 1)
        self.n_new = [0] * (st.slots + 1)
        self.max_new = [0] * (st.slots + 1)
        self.stops = [[-1] * NSTOP for _ in range(st.slots + 1)]
        self.pages = [[] for _ in range(st.slots + 1)]
        self.out = [[] for _ in range(st.slots + 1)]
        self.stack = list(range(1, st.pages))
        self.lst = []

    def alloc(self, targets):
        for s, t in targets:
            if self.alive[s]:
                for _ in range(max(-(-t // PAGE) - len(self.pages[s]), 0)):
                    self.pages[s].append(self.stack.pop())

    def post(self, toks):
        for s, tok in zip(self.lst, toks):
            if self.alive[s]:
                self.out[s].append(tok)
                self.n_new[s] += 1
                self.seq_len[s] += 1
                if tok in self.stops[s] or self.n_new[s] >= self.max_new[s]:
                    self.alive[s] = 0

    def release(self):
        for s in self.lst:
            if not self.alive[s]:
                self.stack.extend(self.pages[s])
                self.pages[s] = []

    def compact(self):
        self.lst = [s for s in self.lst if self.alive[s]]


def check(st: SlotTable, m: Model):
    def host(t):
        return t.tolist()

    assert host(st.alive) == m.alive
    assert host(st.n_new) == m.n_new
    assert host(st.seq_len) == m.seq_len
    n = st.n.item()
    assert host(st.lst)[:n] == m.lst and all(x == 0 for x in host(st.lst)[n:])
    assert st.top.item() == len(m.stack)
    assert host(st.stack)[: len(m.stack)] == m.stack
    table, nb = host(st.table), host(st.nblocks)
    for s in range(1, st.slots + 1):
        assert nb[s] == len(m.pages[s]) and table[s][: nb[s]] == m.pages[s]
        assert host(st.out_tok[s])[: m.n_new[s]] == m.out[s]
    held = sorted(m.stack + [p for ps in m.pages for p in ps])
    assert held == list(range(1, st.pages))


@pytest.mark.parametrize("seed", range(4))
def test_episode(seed):
    rng = random.Random(seed)
    slots, pages = 24, 400
    st = SlotTable(slots, pages, PAGE, 128, 40, DEV)
    m = Model(st)
    free = list(range(1, slots + 1))
    i32 = dict(dtype=torch.int32, device=DEV)
    for _ in range(60):
        k = rng.randint(0, min(4, len(free)))
        new = [free.pop(rng.randrange(len(free))) for _ in range(k)]
        for s in new:
            m.alive[s], m.seq_len[s], m.n_new[s] = 1, rng.randint(1, 30), 0
            m.max_new[s] = rng.randint(1, 30)
            m.stops[s] = [rng.randint(0, 50) for _ in range(2)] + [-1] * (NSTOP - 2)
            m.out[s] = []
            st.alive[s], st.seq_len[s], st.n_new[s] = 1, m.seq_len[s], 0
            st.max_new[s] = m.max_new[s]
            st.stops[s] = torch.tensor(m.stops[s], **i32)
        st.staged[: len(new)] = torch.tensor(new, **i32) if new else st.staged[:0]
        st.count.fill_(len(new))
        K.append(st.lst, st.n, st.staged, st.count)
        m.lst += new
        check(st, m)
        rows = st.lst.clone()
        target = st.seq_len[rows.long()] + 1
        K.alloc(rows, target, st.n, st, maxneed=8 if seed % 2 else 0)
        m.alloc([(s, m.seq_len[s] + 1) for s in m.lst])
        check(st, m)
        toks = [rng.randint(0, 60) for _ in m.lst]
        tok = torch.zeros(slots, **i32)
        tok[: len(toks)] = torch.tensor(toks, **i32) if toks else tok[:0]
        K.post(tok, rows, st.n, st, adv=1)
        m.post(toks)
        check(st, m)
        K.release(st.lst, st.n, st)
        m.release()
        check(st, m)
        dead = [s for s in m.lst if not m.alive[s]]
        K.compact(st.lst, st.n, st.alive)
        m.compact()
        check(st, m)
        free += dead


def test_finish_control():
    """Control: a slot whose stop token is sampled must die; one without must not."""
    st = SlotTable(2, 16, PAGE, 64, 8, DEV)
    st.alive[1:] = 1
    st.max_new[1:] = 8
    st.stops[1, 0] = 7
    st.lst[:2] = torch.tensor([1, 2], dtype=torch.int32)
    st.n.fill_(2)
    K.post(torch.tensor([7, 7], dtype=torch.int32, device=DEV), st.lst, st.n, st, adv=1)
    assert st.alive.tolist() == [0, 0, 1]
