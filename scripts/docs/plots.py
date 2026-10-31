"""The README's figures, from the tables in ``docs/measurements``.

``beat.jpg`` the host and the GPU over four steps; ``decode.jpg`` decode rate and GPU idle per step
against vLLM (``lfm3-vl-2b.md``, ``gemma4.md``, ``qwen35.md``); ``captions.jpg`` root captions per
model and engine (``captions.md``).

    python scripts/docs/plots.py docs/assets
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches

INK, MUTED, PAPER, LINE = "#1f2328", "#57606a", "#ffffff", "#d0d7de"
OURS, KOHAKU, STOCK = "#0969da", "#e6730d", "#6e7781"
VIOLET, GOLD = "#8250df", "#bf8700"

ROWS = [8, 32, 64, 128, 256, 512]
LFM_OURS = [4129, 15648, 29257, 48726, 66298, 79030]
LFM_VLLM = [2773, 7863, 10552, 17261, 20416, 22448]
LFM_IDLE_OURS = [0.09, 0.08, 0.09, 0.13, 0.16, 0.23]
LFM_IDLE_VLLM = [0.28, 1.39, 3.37, 4.18, 7.54, 14.37]
E4B_OURS = [2136, 8020, 13837, 20346, 25544, 29524]
E4B_VLLM = [1832, 6432, 10900, 15266, 17462, 17213]

# model, then in process and HTTP for Tokenomos, KohakuvLLM, stock vLLM; None: does not run or not
# run
CAPTIONS = [
    ("LFM3-VL 2B", 130.30, 135.05, 26.23, 31.73, None, None),
    ("Qwen3.5 4B", 64.91, 62.17, 22.34, 20.20, 23.96, 21.79),
    ("Gemma 4 E4B", 58.93, 55.57, None, None, 21.30, 20.04),
    ("Qwen3.5 9B", 48.99, 46.43, 21.80, 20.82, 21.87, 18.09),
    ("Gemma 4 12B", 22.88, 22.70, None, None, 21.06, 20.71),
    ("Qwen3.8 27B", 18.40, 18.85, 4.94, 4.88, 3.85, 4.26),
    ("Gemma 4 31B", 3.60, 3.61, None, None, 3.25, 3.79),
]


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=12)
    ax.grid(axis="y", color=LINE, lw=0.6)
    ax.set_axisbelow(True)


def label(ax, x, y, s, colour) -> None:
    ax.annotate(
        s,
        (x, y),
        xytext=(8, 0),
        textcoords="offset points",
        color=INK,
        fontsize=12,
        va="center",
        fontweight="bold",
    )
    ax.plot([x], [y], "o", color=colour, ms=8, mec=PAPER, mew=1.5, zorder=4)


def lines(ax, ours, theirs, title, name="vLLM") -> None:
    ax.plot(ROWS, ours, color=OURS, lw=2.4, marker="o", ms=6, mec=PAPER)
    ax.plot(ROWS, theirs, color=STOCK, lw=2.4, marker="o", ms=6, mec=PAPER)
    label(ax, ROWS[-1], ours[-1], f"Tokenomos {ours[-1]:,}", OURS)
    label(ax, ROWS[-1], theirs[-1], f"{name} {theirs[-1]:,}", STOCK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ROWS, [str(r) for r in ROWS])
    ax.minorticks_off()
    ax.set_xlim(7, 3400)
    ax.set_ylim(0)
    ax.set_xlabel("sequences", color=MUTED, fontsize=12)
    ax.set_title(title, loc="left", fontsize=14, color=INK, fontweight="bold")
    style(ax)


def decode(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), facecolor=PAPER)
    lines(axes[0], LFM_OURS, LFM_VLLM, "LFM3-VL 2B MXFP8: decode tok/s", "KohakuvLLM")
    lines(axes[1], E4B_OURS, E4B_VLLM, "Gemma 4 E4B NVFP4: decode tok/s")
    ax = axes[2]
    ax.plot(ROWS, LFM_IDLE_OURS, color=OURS, lw=2.4, marker="o", ms=6, mec=PAPER)
    ax.plot(ROWS, LFM_IDLE_VLLM, color=STOCK, lw=2.4, marker="o", ms=6, mec=PAPER)
    label(ax, ROWS[-1], LFM_IDLE_OURS[-1], f"Tokenomos {LFM_IDLE_OURS[-1]} ms", OURS)
    label(ax, ROWS[-1], LFM_IDLE_VLLM[-1], f"KohakuvLLM {LFM_IDLE_VLLM[-1]} ms", STOCK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ROWS, [str(r) for r in ROWS])
    ax.minorticks_off()
    ax.set_xlim(7, 3400)
    ax.set_ylim(0)
    ax.set_xlabel("sequences", color=MUTED, fontsize=12)
    ax.set_title(
        "GPU idle per step, ms (LFM3, nsys)", loc="left", fontsize=14, color=INK, fontweight="bold"
    )
    style(ax)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(out / "decode.jpg", dpi=150, facecolor=PAPER, pil_kwargs={"quality": 90})


def captions(out: Path) -> None:
    n = len(CAPTIONS)
    fig, ax = plt.subplots(figsize=(13, 0.98 * n + 1.4), facecolor=PAPER)
    engines = ((OURS, 0.27, "Tokenomos"), (KOHAKU, 0.0, "KohakuvLLM"), (STOCK, -0.27, "stock vLLM"))
    for i, (_, *rates) in enumerate(CAPTIONS):
        for j, (colour, lane, _) in enumerate(engines):
            inproc, http = rates[2 * j : 2 * j + 2]
            y = n - 1 - i + lane
            if inproc is None:
                continue
            ax.plot([inproc, http], [y, y], color=colour, lw=2, alpha=0.5, zorder=1)
            ax.plot([http], [y], "o", color=colour, ms=8, mec=PAPER, mew=1, zorder=3)
            ax.plot([inproc], [y], "s", mfc="none", mec=colour, ms=9, mew=1.8, zorder=4)
            ax.annotate(
                f"{inproc:.2f} / {http:.2f}",
                (max(inproc, http), y),
                xytext=(9, 0),
                textcoords="offset points",
                va="center",
                fontsize=10.5,
                color=INK,
            )
    ax.set_yticks(range(n), [c[0] for c in reversed(CAPTIONS)], fontsize=13, color=INK)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 20, 50, 100, 200], ["3", "5", "10", "20", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_xlim(2.5, 320)
    ax.set_xlabel(
        "images per second, one RTX 5090, 1024 rows, in process / HTTP (log scale)",
        color=MUTED,
        fontsize=12,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=12, left=False)
    ax.tick_params(axis="y", colors=INK)
    ax.grid(axis="x", color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    marks = [plt.Line2D([], [], color=colour, lw=3, label=name) for colour, _, name in engines] + [
        plt.Line2D(
            [], [], mec=MUTED, mfc="none", marker="s", ls="", ms=9, mew=1.8, label="in process"
        ),
        plt.Line2D([], [], color=MUTED, marker="o", ls="", ms=8, label="HTTP"),
    ]
    ax.legend(handles=marks, loc="lower right", frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "captions.jpg", dpi=150, facecolor=PAPER, pil_kwargs={"quality": 90})


def beat(out: Path) -> None:
    fig = plt.figure(figsize=(14, 2.75), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2.75)
    ax.axis("off")

    def bar(x, y, w, h, s, face, edge):
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, lw=1.4))
        ax.text(x + w / 2, y + h / 2, s, ha="center", va="center", fontsize=11, color=INK)

    ax.text(0.15, 2.3, "host", fontsize=14, fontweight="bold", color=INK, va="center")
    ax.text(0.15, 0.85, "GPU", fontsize=14, fontweight="bold", color=INK, va="center")
    width = 3.1
    for i, k in enumerate(("k", "k+1", "k+2", "k+3")):
        x = 1.1 + i * width
        bar(x, 2.0, 1.1, 0.6, f"queue {k}", "#ddf4ff", OURS)
        if i < 3:
            prior = ("k−1", "k", "k+1")[i]
            bar(x + 1.1, 2.0, width - 1.25, 0.6, f"wait event {prior}", "#f6f8fa", LINE)
        before = ("k−1", "k", "k+1", "k+2")[i]
        bar(x, 0.55, width, 0.6, f"step {before}", "#fbefff", VIOLET)
        if i:
            ax.annotate(
                "",
                xy=(x - 0.12, 2.0),
                xytext=(x, 1.15),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": GOLD,
                    "lw": 1.6,
                    "linestyle": (0, (5, 3)),
                },
            )
    ax.text(
        1.1,
        0.12,
        "Dashed: the event of a finished step. The next step is already on the GPU when the host "
        "reads it.",
        fontsize=12,
        color=MUTED,
    )
    fig.savefig(out / "beat.jpg", dpi=150, facecolor=PAPER, pil_kwargs={"quality": 90})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    beat(args.out)
    decode(args.out)
    captions(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
