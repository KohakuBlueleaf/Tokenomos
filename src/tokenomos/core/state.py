"""Device-resident engine state: the slot table, the page pool, the active list.

Slot ``s`` (1 .. slots) holds one request; slot 0 is the null slot every padded row points at.
Per slot: ``alive``, ``seq_len`` (positions in the cache), ``last_tok`` (the token the next
forward feeds), ``n_new`` / ``max_new`` / ``out_tok`` (generated tokens, at most ``new_cap``),
``stops`` (padded with -1), the sampling parameters, ``rope_delta`` (a multimodal-RoPE prompt's
generated positions minus its sequence positions), and the page table (``table`` / ``nblocks``;
the first ``shared`` pages belong to a prefix and are not freed with the slot).
Pages live on a device free stack ``stack[:top]``; page 0 is never on it. The active list
``lst[:n]`` is the decode batch; the host stages new slots in ``staged[:count]``. The ``g_*``
arrays hold each slot's grammar state (``kernels.grammar.device``).
"""

import torch

NSTOP = 8


class SlotTable:
    def __init__(self, slots: int, pages: int, page: int, max_len: int, new_cap: int, device):
        self.slots, self.pages, self.page = slots, pages, page
        self.max_len, self.new_cap = max_len, new_cap
        s1 = slots + 1

        def i32(*shape, fill=0):
            return torch.full(shape, fill, dtype=torch.int32, device=device)

        def f32(*shape, fill=0.0):
            return torch.full(shape, fill, dtype=torch.float32, device=device)

        self.alive, self.seq_len, self.last_tok = i32(s1), i32(s1), i32(s1)
        self.n_new, self.max_new = i32(s1), i32(s1)
        self.out_tok = i32(s1, new_cap)
        self.stops = i32(s1, NSTOP, fill=-1)
        self.temp, self.top_p, self.min_p = f32(s1), f32(s1, fill=1.0), f32(s1)
        self.top_k, self.seed = i32(s1), i32(s1)
        self.rope_delta = i32(s1)
        self.table = i32(s1, -(-max_len // page))
        self.nblocks, self.shared = i32(s1), i32(s1)
        self.stack = torch.arange(1, pages, dtype=torch.int32, device=device)
        self.top = i32(1, fill=pages - 1)
        self.lst, self.n = i32(slots), i32(1)
        self.staged, self.count = i32(slots), i32(1)
        self.g_state, self.g_wait, self.g_miss = i32(s1, fill=-1), i32(s1), i32(s1, fill=-1)
        self.g_room, self.g_need = i32(s1, fill=-1), i32(s1)
        self.g_trace = i32(s1, new_cap, fill=-1)
