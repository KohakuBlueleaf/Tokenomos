"""The README's teaser: a request's path from a client to the device that runs itself.

Five columns, left to right: the ways in (``LLM``, the OpenAI server and router, a connector); the
work done apart from the serving loop (prep pool, image ring, staging); the host loop, one step
behind the device; the device, with the slot table, the active list and the decode graph; the
formats each card runs. Events return to the host along the bottom.

    python scripts/docs/teaser.py docs/assets/teaser.jpg
"""

import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib import patches

INK, MUTED, PAPER, CARD, LINE = "#1f2328", "#57606a", "#ffffff", "#f6f8fa", "#d0d7de"
BLUE, VIOLET, GOLD, GREEN, GREY = "#0969da", "#8250df", "#bf8700", "#1a7f37", "#6e7781"

CODE = [
    "from tokenomos import (",
    "    LLM, SamplingParams,",
    ")",
    'with LLM("/models/lfm3-vl-2b",',
    '         arch=Lfm3(head="bf16"),',
    "         max_len=8192) as llm:",
    "    outs = llm.chat(",
    "        conversations,",
    "        SamplingParams(",
    "            max_tokens=1024,",
    "            json_schema=schema,",
    "        ),",
    "    )",
]
GRAPH = [
    [("page alloc", GOLD), ("embed", VIOLET), ("layers", VIOLET), ("logits", VIOLET)],
    [("event", BLUE), ("compact", GOLD), ("release", GOLD), ("grammar,\nsample", VIOLET)],
]
CARDS = [
    ("RTX 5090", "sm120", "MXFP8 · NVFP4 · FP8", "FlashInfer · b12x", VIOLET),
    ("RTX 4090", "sm89", "bf16 · FP8 · W8A8", "Windows too", BLUE),
    ("RTX 3090", "sm86", "bf16 · fp16 · W8A8", "fp32 accumulate", GREEN),
    ("RTX 2080 Ti", "sm75", "fp16 · W8A8", "int8 tensor cores", GOLD),
]


def box(ax, x, y, w, h, face=CARD, edge=LINE, lw=1.4, r=0.1, z=1):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=face,
            edgecolor=edge,
            lw=lw,
            zorder=z,
        )
    )


def arrow(ax, a, b, colour=MUTED, lw=1.8, dashed=False):
    ax.annotate(
        "",
        xy=b,
        xytext=a,
        zorder=6,
        arrowprops={
            "arrowstyle": "-|>,head_length=0.6,head_width=0.3",
            "color": colour,
            "lw": lw,
            "linestyle": (0, (5, 3)) if dashed else "-",
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def text(ax, x, y, s, size, colour=INK, bold=False, mono=False, **kw):
    weight = "bold" if bold else "normal"
    family = "monospace" if mono else None
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=colour,
        fontweight=weight,
        family=family,
        va=kw.pop("va", "top"),
        zorder=7,
        **kw,
    )


def card(ax, x, y, w, h, title, body, s, small=2.5):
    box(ax, x, y, w, h)
    text(ax, x + 0.15, y + h - 0.1, title, s, bold=True)
    text(ax, x + 0.15, y + h - 0.47, body, s - small, colour=MUTED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--font", type=float, default=14.0)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    s = args.font

    fig = plt.figure(figsize=(21, 9.6), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 9.6)
    ax.axis("off")
    top = 9.35

    # ---- 1. the ways in ----------------------------------------------------------------------
    x1, w1 = 0.2, 3.75
    text(ax, x1 + 0.05, top, "many requests", s + 4, bold=True)
    text(ax, x1 + 0.05, top - 0.38, "from Python, HTTP or a connector", s)
    box(ax, x1, 4.0, w1, 4.7)
    for i, line in enumerate(CODE):
        text(ax, x1 + 0.15, 8.55 - i * 0.355, line, s - 2.2, mono=True)
    ways = [
        ("LLM, in this process", "chat · generate · stream_chat", 2.75),
        ("OpenAI server, router", "SSE · one server per GPU", 1.5),
        ("Togeari connector", "a host process per card", 0.25),
    ]
    for title, body, y in ways:
        card(ax, x1, y, w1, 1.05, title, body, s - 1)

    # ---- 2. the work kept apart -----------------------------------------------------------------
    x2, w2 = 4.3, 3.35
    text(ax, x2, top, "prepared apart", s + 4, bold=True)
    text(ax, x2, top - 0.38, "the serving loop never does it", s)
    card(
        ax,
        x2,
        6.45,
        w2,
        2.25,
        "Prep pool",
        "template, tokens, image decode\nand resize, in worker processes\n\n"
        "935-1313 rows/s at 16 workers",
        s,
        small=1.6,
    )
    card(
        ax,
        x2,
        4.6,
        w2,
        1.65,
        "Image ring",
        "shared, page-locked memory:\nworkers write pixels once,\nthe device copies them",
        s,
        small=1.6,
    )
    card(
        ax,
        x2,
        2.75,
        w2,
        1.65,
        "Staging arenas",
        "every host-to-device copy,\nreused past an event behind\nits last copy",
        s,
        small=1.6,
    )
    box(ax, x2, 0.25, w2, 2.3, face=PAPER, edge=GREEN, lw=1.8)
    text(ax, x2 + 0.15, 2.45, "No starved step", s, colour=GREEN, bold=True)
    text(
        ax,
        x2 + 0.15,
        2.08,
        "the whole frontend runs at 6x or\nmore of the fastest caption rate;\n"
        "the GPU sits uncovered for\n0.1-2.9% of a caption run",
        s - 1.6,
    )
    arrow(ax, (x1 + w1, 7.5), (x2, 7.5), colour=INK)

    # ---- 3. the host loop --------------------------------------------------------------------
    x3, w3 = 8.0, 3.7
    text(ax, x3, top, "Host loop", s + 4, colour=BLUE, bold=True)
    text(ax, x3, top - 0.38, "one thread, one step behind", s)
    box(ax, x3, 0.25, w3, 8.45, face=PAPER, edge=BLUE, lw=2.0, z=0)
    rows = [
        ("Admission", "a free slot, a gathered batch,\nreserved pages, a chunk budget", 8.55),
        (
            "Encode, prefill",
            "one packed vision batch; chunks\nof every row; the last samples",
            7.25,
        ),
        ("Replay", "one CUDA graph per bucket:\nthe smallest holding the list", 5.95),
        ("Read the last event", "free slots, return reservations,\nwatch the free stack", 4.65),
    ]
    for title, body, y in rows:
        text(ax, x3 + 0.2, y, title, s, bold=True)
        text(ax, x3 + 0.2, y - 0.36, body, s - 2.5, colour=MUTED)
    text(ax, x3 + 0.2, 3.3, "Host ms per decode step", s, bold=True)
    box(ax, x3 + 0.2, 1.45, w3 - 0.4, 1.4, face=CARD, edge=LINE, lw=1.0, r=0.05, z=2)
    for i, (rows_n, ms) in enumerate(
        (("8 rows", "0.12"), ("64 rows", "0.13"), ("512 rows", "0.41"))
    ):
        y = 2.75 - i * 0.42
        text(ax, x3 + 0.35, y, rows_n, s - 2, mono=True)
        text(ax, x3 + w3 - 0.35, y, ms, s - 2, colour=BLUE, mono=True, bold=True, ha="right")
    text(
        ax,
        x3 + 0.2,
        1.25,
        "The next step is queued before this\none's event is read: the host never\n"
        "stands between two steps",
        s - 2.5,
    )
    arrow(ax, (x2 + w2, 7.5), (x3, 7.5), colour=INK)

    # ---- 4. the device ---------------------------------------------------------------------------
    x4, w4 = 12.05, 5.5
    text(ax, x4, top, "Device", s + 4, colour=VIOLET, bold=True)
    text(ax, x4, top - 0.38, "replays by itself: every per-row decision", s)
    box(ax, x4, 0.25, w4, 8.45, face=PAPER, edge=VIOLET, lw=2.0, z=0)
    half = (w4 - 0.55) / 2
    card(
        ax,
        x4 + 0.2,
        7.0,
        half,
        1.55,
        "Slot table",
        "alive, length, last token,\nstops, sampling, page table;\nrow 0 is the null slot",
        s - 0.5,
    )
    card(
        ax,
        x4 + 0.35 + half,
        7.0,
        half,
        1.55,
        "State pools",
        "KV pages and a free stack,\nrings, conv and GDN rows,\nprefix snapshots, grammar",
        s - 0.5,
    )
    box(ax, x4 + 0.2, 6.0, w4 - 0.4, 0.82, face="#fff8c5", edge=GOLD)
    text(ax, x4 + 0.35, 6.73, "Active list", s, bold=True)
    text(
        ax, x4 + 0.35, 6.37, "the rows of this step: prefill appends, compact closes gaps", s - 2.5
    )
    text(ax, x4 + 0.2, 5.78, "Decode graph, a step", s, colour=VIOLET, bold=True)
    bw, gap = 1.1, 0.14
    for r, row in enumerate(GRAPH):
        y = 4.3 - r * 1.25
        for c, (name, colour) in enumerate(row):
            bx = x4 + 0.2 + c * (bw + gap)
            box(ax, bx, y, bw, 0.95, face=PAPER, edge=colour, lw=1.8, r=0.06, z=2)
            text(ax, bx + bw / 2, y + 0.475, name, s - 2.5, ha="center", va="center")
            if c:
                a, b = (bx - gap, y + 0.475), (bx, y + 0.475)
                arrow(ax, a if r == 0 else b, b if r == 0 else a, lw=1.4)
    arrow(ax, (x4 + 0.2 + 3.5 * bw + 3 * gap, 4.3), (x4 + 0.2 + 3.5 * bw + 3 * gap, 4.0), lw=1.4)
    text(
        ax,
        x4 + 0.2,
        2.8,
        "A step costs its bucket: rows that finished or sit out\npoint at the null slot "
        "and change nothing.",
        s - 2.5,
        colour=MUTED,
    )
    text(ax, x4 + 0.2, 1.75, "GPU idle per step, 512 rows", s, bold=True)
    text(ax, x4 + 0.2, 1.33, "0.23 ms", s + 6, colour=VIOLET, bold=True)
    text(ax, x4 + 2.05, 1.26, "against stock vLLM's 23.21 ms", s - 1)
    text(ax, x4 + 0.2, 0.68, "LFM3-VL 2B, RTX 5090, by nsys", s - 2.5, colour=MUTED)
    arrow(ax, (x3 + w3, 6.4), (x4, 6.4), colour=INK)

    # ---- 5. the cards ---------------------------------------------------------------------------
    x5, w5 = 17.9, 2.9
    text(ax, x5, top, "one engine", s + 4, bold=True)
    text(ax, x5, top - 0.38, "the card picks the kernels", s)
    for i, (name, sm, formats, note, colour) in enumerate(CARDS):
        y = 7.35 - i * 1.55
        box(ax, x5, y, w5, 1.35, face=CARD, edge=colour, lw=1.6)
        text(ax, x5 + 0.12, y + 1.25, name, s - 1, colour=colour, bold=True)
        text(ax, x5 + w5 - 0.12, y + 1.23, sm, s - 3, colour=MUTED, ha="right")
        text(ax, x5 + 0.12, y + 0.86, formats, s - 2.5)
        text(ax, x5 + 0.12, y + 0.48, note, s - 2.5, colour=MUTED)
    text(
        ax,
        x5,
        1.2,
        "LFM3 · Qwen3.5-3.8 · Gemma 4\nTriton and TileLang kernels,\nFlashInfer where it runs",
        s - 2.5,
        colour=MUTED,
    )

    # The event goes back to the host from the end of the graph
    arrow(ax, (x4 + 0.2, 3.525), (x3 + w3, 3.525), colour=BLUE, lw=2.0)

    fig.savefig(args.out, dpi=args.dpi, facecolor=PAPER, pil_kwargs={"quality": 90})
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
