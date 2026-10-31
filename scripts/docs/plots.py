"""The README's figures, from ``results.py``.

``beat.jpg`` the host and the GPU over four steps; ``decode.jpg``, ``prefill.jpg`` rates per model
and engine; ``step.jpg`` a decode step's time; ``captions.jpg`` root captions; ``spend.jpg`` where a
caption run goes; ``gbc.jpg`` the full graph; ``turing.jpg``, ``precision.jpg``, ``kernels.jpg`` the
RTX 3090 and 2080 Ti; ``windows.jpg`` the RTX 4090.

    python scripts/docs/plots.py docs/assets
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import results as r
from matplotlib import colors, patches, ticker

INK, MUTED, PAPER, LINE = "#1f2328", "#57606a", "#ffffff", "#d0d7de"
OURS, KOHAKU, STOCK = "#0969da", "#e6730d", "#6e7781"
VIOLET, GOLD, GREEN = "#8250df", "#bf8700", "#1a7f37"
ENGINES = (("Tokenomos", OURS), ("KohakuvLLM", KOHAKU), ("stock vLLM", STOCK))
FORMATS = {"bf16": STOCK, "W8A8": VIOLET, "FP8": OURS, "FP8, GDN too": GOLD, "FP8 dynamic": GREEN}


def pale(colour: str):
    return colors.to_rgba(colour, 0.45)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=150, facecolor=PAPER, pil_kwargs={"quality": 90})
    plt.close(fig)


def style(ax, grid="y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=12)
    ax.grid(axis=grid, color=LINE, lw=0.6)
    ax.set_axisbelow(True)


def title(ax, s: str) -> None:
    ax.set_title(s, loc="left", fontsize=13, color=INK, fontweight="bold")


def legend(fig, pairs, ncol=None) -> None:
    marks = [plt.Line2D([], [], color=colour, lw=4, label=name) for name, colour in pairs]
    fig.legend(
        handles=marks, loc="upper center", ncol=ncol or len(marks), frameon=False, fontsize=13
    )


def lines(ax, rows, series, name, fmt) -> None:
    """Draw (colour, values) pairs; end labels keep apart by 7% of the axis."""
    drawn = [(colour, values) for colour, values in series if values is not None]
    for colour, values in drawn:
        ax.plot(rows, values, color=colour, lw=2.2, marker="o", ms=5, mec=PAPER)
    top = max(max(values) for _, values in drawn)
    ax.set_ylim(0, top * 1.16)
    floor = None
    for colour, values in sorted(drawn, key=lambda d: d[1][-1]):
        y = values[-1] if floor is None else max(values[-1], floor)
        floor = y + top * 0.07
        ax.annotate(
            fmt(values[-1]),
            (rows[-1], y),
            xytext=(7, 0),
            textcoords="offset points",
            annotation_clip=False,
            color=colour,
            fontsize=11,
            fontweight="bold",
            va="center",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(rows, [str(n) for n in rows])
    ax.minorticks_off()
    ax.set_xlim(rows[0] * 0.85, rows[-1] * 2.4)
    title(ax, name)
    style(ax)


def hbars(ax, names, values, tints, name, fmt, log=False, low=None, second=None) -> None:
    """One bar a row, its value at its end; ``second`` adds a paler bar under each row."""
    shown = [v for v in values + (second or []) if v is not None]
    height = 0.42 if second else 0.72
    for i, (value, tint) in enumerate(zip(values, tints)):
        pairs = [(value, tint, -height / 2 if second else 0)]
        if second:
            pairs.append((second[i], pale(tint), height / 2))
        for v, c, shift in pairs:
            if v is None:
                continue
            ax.barh(i + shift, v - (low or 0), height=height, color=c, left=low or 0)
            ax.annotate(
                fmt(v),
                (v, i + shift),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=10.5,
                color=INK,
            )
    if log:
        ax.set_xscale("log")
        ax.set_xlim(min(shown) / 1.5, max(shown) * 6)
        ax.minorticks_off()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        ax.set_xlim(low or 0, (low or 0) + (max(shown) - (low or 0)) * 1.3)
    ax.set_yticks(range(len(names)), names, fontsize=12)
    ax.set_ylim(len(names) - 0.5, -0.5)
    title(ax, name)
    style(ax, grid="x")
    ax.tick_params(axis="y", colors=INK, left=False)


def rates(out: Path, data, name: str, unit: str, last=None) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.4), facecolor=PAPER)
    for ax, (model, rows, ours, kohaku, stock) in zip(axes.flat, data):
        series = ((STOCK, stock), (KOHAKU, kohaku), (OURS, ours))
        lines(ax, rows, series, f"{model}: {unit}", lambda v: f"{v:,}")
    if last:
        last(axes.flat[-1])
    else:
        axes.flat[-1].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("sequences", color=MUTED, fontsize=12)
    legend(fig, ENGINES)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.0, h_pad=2.0)
    save(fig, out, name)


def idle(ax) -> None:
    series = [(c, r.STEP[n][1]) for n, c in reversed(ENGINES)]
    lines(ax, r.ROWS, series, "LFM3 MXFP8: GPU idle ms per step", lambda v: f"{v:.2f}")


def step(out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.9), facecolor=PAPER)
    top = max(b + i for busy, idling in r.STEP.values() for b, i in zip(busy, idling))
    for ax, (name, colour) in zip(axes, ENGINES):
        busy, idling = r.STEP[name]
        xs = range(len(r.ROWS))
        ax.bar(xs, busy, color=colour, width=0.72)
        ax.bar(xs, idling, bottom=busy, color=pale(colour), width=0.72)
        for x, b, i in zip(xs, busy, idling):
            ax.annotate(
                f"{b + i:.2f}",
                (x, b + i),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=10.5,
                color=INK,
            )
        ax.set_xticks(xs, [str(n) for n in r.ROWS])
        ax.set_ylim(0, top * 1.12)
        ax.set_xlabel("sequences", color=MUTED, fontsize=12)
        title(ax, f"{name}: ms per step")
        style(ax)
    ax = axes[-1]
    ax.plot(r.ROWS, r.HOST, color=OURS, lw=2.2, marker="o", ms=5, mec=PAPER)
    ax.set_xscale("log", base=2)
    ax.set_xticks(r.ROWS, [str(n) for n in r.ROWS])
    ax.minorticks_off()
    ax.set_ylim(0, 0.5)
    for x, y in zip(r.ROWS, r.HOST):
        ax.annotate(
            f"{y:.2f}",
            (x, y),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=10.5,
            color=INK,
        )
    ax.set_xlabel("sequences", color=MUTED, fontsize=12)
    title(ax, "Tokenomos: host ms per step")
    style(ax)
    legend(fig, (("GPU busy (solid)", INK), ("GPU idle (pale)", pale(INK))))
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.0)
    save(fig, out, "step.jpg")


def captions(out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.4), facecolor=PAPER)
    for ax, (model, *values) in zip(axes.flat, r.CAPTIONS):
        x, ticks = 0.0, []
        for j, (engine, colour) in enumerate(ENGINES):
            inproc, http = values[2 * j : 2 * j + 2]
            if inproc is None:
                continue
            looped = (model, engine) in r.LOOPS
            for k, (v, c) in enumerate(((inproc, colour), (http, pale(colour)))):
                loop = looped and k == 0
                ax.bar(x + k, v, width=0.92, color=c, hatch="///" if loop else None, ec=PAPER)
                ax.annotate(
                    f"{v:.2f}" + (" loops" if loop else ""),
                    (x + k, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10.5,
                    color=INK,
                )
            ticks.append((x + 0.5, engine))
            x += 2.6
        ax.set_xticks([t for t, _ in ticks], [n for _, n in ticks], fontsize=11, color=INK)
        ax.set_ylim(0, max(v for v in values if v) * 1.14)
        title(ax, model)
        style(ax)
    axes.flat[-1].axis("off")
    axes[0][0].set_ylabel("images per second", color=MUTED, fontsize=12)
    axes[1][0].set_ylabel("images per second", color=MUTED, fontsize=12)
    legend(fig, ENGINES + (("in process (solid)", INK), ("HTTP (pale)", pale(INK))))
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.0, h_pad=2.0)
    save(fig, out, "captions.jpg")


def spend(out: Path) -> None:
    fig, (share, ceil) = plt.subplots(1, 2, figsize=(16, 5.2), facecolor=PAPER)
    names = [row[0] for row in r.SPEND]
    start = [0.0] * len(names)
    stages = (("vision", GOLD), ("prefill", VIOLET), ("decode", OURS))
    for j, (_, colour) in enumerate(stages):
        part = [row[j + 1] for row in r.SPEND]
        share.barh(names, part, left=start, color=colour, height=0.72, ec=PAPER)
        for i, (v, s) in enumerate(zip(part, start)):
            if v >= 8:
                share.text(s + v / 2, i, f"{v:g}%", color=PAPER, ha="center", va="center")
        start = [s + v for s, v in zip(start, part)]
    share.set_ylim(len(names) - 0.5, -0.5)
    share.set_xlim(0, 100)
    title(share, "Share of the wall, 1024 rows")
    style(share, grid="x")
    share.tick_params(axis="y", colors=INK, left=False)
    measured = [row[4] for row in r.SPEND]
    ceiling = [row[5] for row in r.SPEND]
    hbars(
        ceil,
        names,
        measured,
        [OURS] * len(names),
        "img/s, run (solid) against the sum of stage ceilings (pale)",
        lambda v: f"{v:g}",
        log=True,
        second=ceiling,
    )
    legend(fig, stages)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=3.0)
    save(fig, out, "spend.jpg")


def gbc(out: Path) -> None:
    fig, (runs, jobs) = plt.subplots(
        1, 2, figsize=(16, 6.2), facecolor=PAPER, gridspec_kw={"width_ratios": [3, 2]}
    )
    cards = {"RTX 5090": OURS, "RTX 3090": GREEN, "RTX 2080 Ti": GOLD}
    names = [f"{card[4:]}  {vlm}" for card, vlm, _ in r.GBC]
    values = [v for _, _, v in r.GBC]
    tints = [cards[card] for card, _, _ in r.GBC]
    hbars(runs, names, values, tints, "Whole graph, img/s", lambda v: f"{v:.2f}")
    tones = (VIOLET, OURS, GOLD, GREEN)
    for (vlm, (x, y)), tone in zip(r.GBC_JOBS.items(), tones):
        jobs.plot(x, y, color=tone, lw=2.2, marker="o", ms=5, mec=PAPER)
        jobs.annotate(
            f"{vlm} {y[-1]:.2f}",
            (x[-1], y[-1]),
            xytext=(7, 4 if vlm == "Gemma 4 E4B" else -8 if vlm == "Qwen3.5 9B" else 0),
            textcoords="offset points",
            color=tone,
            fontsize=11,
            fontweight="bold",
            va="center",
        )
    jobs.set_xscale("log", base=2)
    jobs.set_xticks([96, 192, 384, 768], ["96", "192", "384", "768"])
    jobs.minorticks_off()
    jobs.set_xlim(85, 2200)
    jobs.set_ylim(0, 5.2)
    jobs.set_xlabel("resident jobs", color=MUTED, fontsize=12)
    title(jobs, "RTX 5090, img/s by resident jobs")
    style(jobs)
    legend(fig, tuple(cards.items()))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=3.0)
    save(fig, out, "gbc.jpg")


def turing(out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19, 12), facecolor=PAPER, sharey=True)
    names = [row[0] for row in r.TURING]
    tones = {"LFM3": OURS, "4B": VIOLET, "E4B": GOLD}
    tints = [tones[n.split()[0]] for n in names]
    metrics = (
        ("decode tok/s", lambda v: f"{v:,}"),
        ("prefill tok/s", lambda v: f"{v:,}"),
        ("host ms per step", lambda v: f"{v:.2f}"),
        ("img/s", lambda v: f"{v:.2f}"),
    )
    for c, card in enumerate(("RTX 3090", "RTX 2080 Ti")):
        for m, (metric, fmt) in enumerate(metrics):
            values = [row[c + 1][m] if row[c + 1] else None for row in r.TURING]
            hbars(axes[c][m], names, values, tints, f"{card}: {metric}", fmt)
    legend(fig, (("LFM3-VL 2B", OURS), ("Qwen3.5 4B", VIOLET), ("Gemma 4 E4B", GOLD)))
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.0, h_pad=2.0)
    save(fig, out, "turing.jpg")


def precision(out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), facecolor=PAPER, sharey=True)
    names = [row[0] for row in r.PRECISION]
    tints = [OURS if n.startswith("LFM3") else VIOLET for n in names]
    speed, gib, argmax, gap = ([row[i] for row in r.PRECISION] for i in range(1, 5))
    hbars(axes[0], names, gap, tints, "Mean logit gap to float32", lambda v: f"{v:.6f}", log=True)
    hbars(axes[1], names, argmax, tints, "Argmax agreement", lambda v: f"{v:.5f}", low=0.98)
    hbars(axes[2], names, speed, tints, "Decode tok/s, 128 rows", lambda v: f"{v:,}")
    hbars(axes[3], names, gib, tints, "GiB", lambda v: f"{v:.2f}")
    legend(fig, (("LFM3-VL 2B", OURS), ("Qwen3.5 4B", VIOLET)))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    save(fig, out, "precision.jpg")


def grouped(ax, groups, labels, tones, name, fmt, log=True) -> None:
    """Bars per group; a None draws nothing."""
    width = 0.8 / len(labels)
    for j, tone in enumerate(tones):
        for i, (_, values) in enumerate(groups):
            if values[j] is None:
                continue
            x = i + (j - (len(labels) - 1) / 2) * width
            ax.bar(x, values[j], width=width * 0.94, color=tone)
            ax.annotate(
                fmt(values[j]),
                (x, values[j]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9.5,
                color=INK,
            )
    if log:
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticks(range(len(groups)), [g for g, _ in groups], fontsize=11, color=INK)
    title(ax, name)
    style(ax)


def kernels(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), facecolor=PAPER)
    labels = ("int8, i32", "TileLang fp16, f32", "cuBLAS fp16", "cuBLAS fp32")
    tones = (VIOLET, GREEN, GOLD, STOCK)
    groups = [(card, values) for card, *values in r.GEMM]
    grouped(axes[0], groups, labels, tones, "GEMM, trillion MAC/s", lambda v: f"{v:g}", log=False)
    axes[0].legend(
        handles=[plt.Line2D([], [], color=c, lw=4, label=n) for n, c in zip(labels, tones)],
        frameon=False,
        fontsize=10.5,
        loc="upper right",
    )
    routes = ("fused", "FlashInfer", "FLA")
    tones = (OURS, GOLD, STOCK)
    for ax, j, model in ((axes[1], 1, "Qwen3.5 4B"), (axes[2], 2, "Qwen3.8 27B")):
        groups = [(row[0][4:], row[j]) for row in r.GDN]
        grouped(ax, groups, routes, tones, f"GDN prefill {model}, µs a call", lambda v: f"{v:,}")
    axes[1].legend(
        handles=[plt.Line2D([], [], color=c, lw=4, label=n) for n, c in zip(routes, tones)],
        frameon=False,
        fontsize=11,
        loc="upper left",
    )
    fig.tight_layout(w_pad=2.0)
    save(fig, out, "kernels.jpg")


def windows(out: Path) -> None:
    fig = plt.figure(figsize=(19, 11.5), facecolor=PAPER)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.5])
    panels = (
        ("LFM3", "LFM3-VL 2B"),
        ("4B", "Qwen3.5 4B"),
        (("9B", "E4B"), "Qwen3.5 9B, Gemma E4B"),
    )
    sizes = [32, 128, 256]
    for c, (keys, name) in enumerate(panels):
        ax = fig.add_subplot(grid[0, c])
        for model, fmt, decode, *_ in r.WINDOWS:
            if model not in (keys if isinstance(keys, tuple) else (keys,)):
                continue
            pts = [(s, v) for s, v in zip(sizes, decode) if v is not None]
            tag = fmt if isinstance(keys, str) else f"{model} {fmt}"
            ax.plot(*zip(*pts), color=FORMATS[fmt], lw=2.2, marker="o", ms=5, mec=PAPER, label=tag)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.minorticks_off()
        ax.set_ylim(0)
        ax.set_xlabel("sequences", color=MUTED, fontsize=12)
        title(ax, f"{name}: decode tok/s")
        style(ax)
        ax.legend(frameon=False, fontsize=10.5)
    names = [f"{row[0]} {row[1]}" for row in r.WINDOWS]
    tints = [FORMATS[row[1]] for row in r.WINDOWS]
    ax = fig.add_subplot(grid[1, 0])
    prefill = [row[3] for row in r.WINDOWS]
    hbars(ax, names, prefill, tints, "Prefill tok/s", lambda v: f"{v:,}")
    ax = fig.add_subplot(grid[1, 1])
    host = [row[4] for row in r.WINDOWS]
    hbars(ax, names, host, tints, "Host ms per step", lambda v: f"{v:.2f}")
    ax = fig.add_subplot(grid[1, 2])
    inproc = [row[5] for row in r.WINDOWS]
    http = [row[6] for row in r.WINDOWS]
    hbars(
        ax,
        names,
        inproc,
        tints,
        "img/s, in process (solid) and HTTP (pale)",
        lambda v: f"{v:.2f}",
        second=http,
    )
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    save(fig, out, "windows.jpg")


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
    save(fig, out, "beat.jpg")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    beat(args.out)
    rates(args.out, r.DECODE, "decode.jpg", "decode tok/s", idle)
    rates(args.out, r.PREFILL, "prefill.jpg", "prefill tok/s")
    step(args.out)
    captions(args.out)
    spend(args.out)
    gbc(args.out)
    turing(args.out)
    precision(args.out)
    kernels(args.out)
    windows(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
