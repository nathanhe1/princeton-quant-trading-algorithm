"""Render the research charts from the CSVs that chart_data.py caches.

Run from ``research/`` with ``../.venv/bin/python charts.py``. Writes PNGs to ``charts/``.
All markouts are the 120-batch proxy, out of sample (fold coefficients), in $ per session.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "charts", "data")
OUT = os.path.join(HERE, "charts")
SYMS = ["HPPO", "BNKY", "BRRY", "CRRT"]

# Reference palette (light mode): surface, ink, chrome, first three categorical slots.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE, RED = "#2a78d6", "#e34948"
DIVERGING = LinearSegmentedColormap.from_list(
    "rb", ["#b3302f", "#ec9796", "#f0efec", "#86b6ef", "#1c5cab"]
)
SHORT = {
    "base": "base (backtest rule)",
    "through=2": "through ≥ 2 ticks",
    "share=0.3": "30% of trade size",
    "share=0.1": "10% of trade size",
    "share=0.3+through=2": "30% + through 2",
    "adverse": "adverse at-price fills",
    "share=0.3+through=2+adverse": "realistic: 30% + through 2 + adverse",
    "all combined (share=0.1)": "harshest: 10% + through 2 + adverse",
}
REALISTIC = "share=0.3+through=2+adverse"

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
        "savefig.dpi": 200,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "text.parse_math": False,  # "$" is currency here, not mathtext
    }
)


def money(value):
    return f"-${-value:,.0f}" if value < 0 else f"${value:,.0f}"


def dollar_axis(ax, axis="x"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: money(v))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)


def suptitle(fig, title, subtitle):
    fig.text(0.0, 1.06, title, fontsize=13, fontweight="bold", color=INK, transform=fig.transFigure)
    fig.text(0.0, 1.015, subtitle, fontsize=9.5, color=INK2, transform=fig.transFigure)


def strip(ax, groups, labels, highlight=None):
    """One row per group: session dots, a mean tick, and mean / worst labels at the right."""
    rng = np.random.default_rng(0)
    xmax = max(g.max() for g in groups)
    for row, values in enumerate(groups):
        jitter = rng.uniform(-0.17, 0.17, len(values))
        ax.scatter(values, row + jitter, s=34, color=BLUE, alpha=0.75, edgecolors=SURFACE, linewidths=1.2, zorder=3)
        ax.plot([values.mean()] * 2, [row - 0.32, row + 0.32], color=INK, lw=2.2, solid_capstyle="round", zorder=4)
        ax.text(
            xmax * 1.04, row, f"{money(values.mean())}   worst {money(values.min())}",
            va="center", fontsize=9, color=INK if labels[row] == highlight else INK2,
            fontweight="bold" if labels[row] == highlight else "normal",
        )
    ax.set_yticks(range(len(groups)), labels)
    for tick in ax.get_yticklabels():
        if tick.get_text() == highlight:
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    dollar_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=34, color=BLUE, alpha=0.75, label="one session (24)")
    ax.plot([], [], color=INK, lw=2.2, label="mean")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, handletextpad=0.4)


def stress_scenarios():
    s = pd.read_csv(f"{DATA}/stress_sessions.csv")
    order = list(SHORT)
    groups = [s[s.scenario == sc].mk120.to_numpy() for sc in order]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    strip(ax, groups, [SHORT[sc] for sc in order], highlight=SHORT[REALISTIC])
    ax.set_xlabel("120-batch proxy markout per session (out of sample)")
    suptitle(
        fig,
        "Stress test: markout per session under harsher fill models",
        "alpha.py, 3-fold CV, 24 sessions per row. Mean and worst session at right.",
    )
    fig.savefig(f"{OUT}/01_stress_scenarios.png")
    plt.close(fig)


def heat(ax, table, title, fmt, cmap_center=0.0):
    values = table.to_numpy(dtype=float)
    span = np.nanmax(np.abs(values - cmap_center)) or 1
    norm = TwoSlopeNorm(vcenter=cmap_center, vmin=cmap_center - span, vmax=cmap_center + span)
    ax.imshow(values, cmap=DIVERGING, norm=norm, aspect="auto")
    for (r, c), v in np.ndenumerate(values):
        shade = abs(norm(v) - 0.5) > 0.33
        ax.text(c, r, fmt(v), ha="center", va="center", fontsize=9, color="#ffffff" if shade else INK)
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, table.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, table.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, pad=24)


def stress_by_symbol():
    f = pd.read_csv(f"{DATA}/stress_fills.csv")
    order = list(SHORT)
    maker = f[f.liq == "MAKER"].groupby(["scenario", "sym"])[["mk120", "shares"]].sum()
    cps = (maker.mk120 / maker.shares).unstack().reindex(index=order, columns=SYMS)
    dollars = (f.groupby(["scenario", "sym"]).mk120.sum() / 100 / 24).unstack().reindex(index=order, columns=SYMS)
    for table in (cps, dollars):
        table.index = [SHORT[i] for i in table.index]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.0), gridspec_kw={"wspace": 0.05})
    heat(a, cps, "Maker edge, ¢ per share", lambda v: f"{v:.2f}")
    heat(b, dollars, "Markout, $ per session (maker + taker)", lambda v: money(v))
    b.set_yticklabels([])
    suptitle(
        fig,
        "Where the edge survives: by symbol and fill scenario",
        "Darker blue = more markout (a red cell would mean negative). Each panel has its own scale.",
    )
    fig.savefig(f"{OUT}/02_stress_by_symbol.png")
    plt.close(fig)


def stress_volume_vs_edge():
    s = pd.read_csv(f"{DATA}/stress_sessions.csv")
    f = pd.read_csv(f"{DATA}/stress_fills.csv")
    order = list(SHORT)
    shares = s.groupby("scenario").shares.mean().reindex(order)
    m = f[f.liq == "MAKER"].groupby("scenario")[["mk120", "shares"]].sum().reindex(order)
    cps = m.mk120 / m.shares
    labels = [SHORT[i] for i in order]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(order))
    a.barh(y, shares / 1000, height=0.56, color=BLUE)
    for yi, v in zip(y, shares):
        a.text(v / 1000 + shares.max() / 1000 * 0.015, yi, f"{v / 1000:,.0f}k", va="center", fontsize=9, color=INK2)
    a.set_title("Shares filled per session")
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}k"))
    colors = [BLUE if v >= 0 else RED for v in cps]
    b.barh(y, cps, height=0.56, color=colors)
    for yi, v in zip(y, cps):
        b.text(v + (0.03 if v >= 0 else -0.03), yi, f"{v:.2f}¢", va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK2)
    b.axvline(0, color=AXIS, lw=1)
    b.set_title("Maker markout per share (¢)")
    lo, hi = min(cps.min(), 0), cps.max()
    b.set_xlim(lo - 0.35 * (hi - lo) * (lo < 0), hi * 1.18)
    for ax in (a, b):
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    a.set_yticks(y, labels)
    a.invert_yaxis()
    suptitle(
        fig,
        "Volume falls faster than edge per share",
        "Mean over 24 out-of-sample sessions. Haircuts cut volume; adverse fills cut the per-share edge.",
    )
    fig.savefig(f"{OUT}/03_stress_volume_vs_edge.png")
    plt.close(fig)


def versions():
    s = pd.read_csv(f"{DATA}/versions_sessions.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"hspace": 0.45})
    for ax, scenario in zip(axes, ["base", REALISTIC]):
        p = s[s.scenario == scenario].pivot(index="sess", columns="strategy", values="mk120")
        d = p["alpha (current)"] - p["alpha_v1"]
        x = np.arange(len(d))
        ax.bar(x, d, width=0.62, color=[BLUE if v >= 0 else RED for v in d])
        ax.axhline(0, color=AXIS, lw=1)
        ax.plot([-0.5, len(d) - 0.5], [d.mean()] * 2, color=INK, lw=1.4, ls=(0, (1, 2)))
        ax.text(len(d) - 0.3, d.mean(), f"mean {'+' if d.mean() >= 0 else '-'}{money(abs(d.mean()))}", va="center", ha="left", fontsize=9, color=INK)
        dollar_axis(ax, "y")
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        better = int((d > 0).sum())
        ax.set_title(
            f"{SHORT[scenario]}\n"
            f"mean {money(p['alpha_v1'].mean())} → {money(p['alpha (current)'].mean())},  "
            f"worst {money(p['alpha_v1'].min())} → {money(p['alpha (current)'].min())},  better in {better}/24 sessions",
            fontsize=10, linespacing=1.5,
        )
        ax.set_xticks(x, d.index, fontsize=8.5)
        ax.set_xlim(-0.6, len(d) + 2.2)
    suptitle(
        fig,
        "Skew change (alpha_v1 → current): change in markout per session",
        "Current minus v1, same sessions and fold coefficients. Skew was tuned on s01–s08 only.",
    )
    fig.savefig(f"{OUT}/04_v1_vs_current.png")
    plt.close(fig)


def horizons():
    r2 = pd.read_csv(f"{DATA}/horizon_r2.csv").set_index("H")
    s = pd.read_csv(f"{DATA}/horizon_sessions.csv")
    hs = list(r2.index)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.55})
    width = 0.24
    x = np.arange(len(SYMS))
    for i, h in enumerate(hs):
        bars = a.bar(x + (i - 1) * (width + 0.02), r2.loc[h, SYMS], width=width, color=SERIES[i], label=f"{h} batches")
        a.bar_label(bars, fmt="%.2f", fontsize=7, color=INK2, padding=2)
    a.set_xticks(x, SYMS)
    a.tick_params(axis="x", length=0)
    a.set_title("Held-out R² of the fair-value model")
    a.grid(True, axis="y")
    a.set_axisbelow(True)
    a.legend(loc="upper left", fontsize=9, title="target horizon", title_fontsize=9)
    groups = [s[s.H == h].mk120.to_numpy() for h in hs]
    strip(b, groups, [f"H = {h}" for h in hs], highlight="H = 120")
    b.set_title("Strategy markout per session (120-batch proxy)", pad=22)
    suptitle(
        fig,
        "Fair-value target horizon: 120 batches stays best",
        "Fold coefficients refit per horizon; R² and markout on the held-out 8 sessions of each fold.",
    )
    fig.savefig(f"{OUT}/05_horizons.png")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = {
        "stress_sessions.csv": [stress_scenarios, stress_volume_vs_edge],
        "stress_fills.csv": [stress_by_symbol],
        "versions_sessions.csv": [versions],
        "horizon_r2.csv": [horizons],
    }
    for file, funcs in jobs.items():
        if not os.path.exists(f"{DATA}/{file}"):
            print(f"skip: {file} missing (run chart_data.py)")
            continue
        for func in funcs:
            func()
            print("wrote", func.__name__)
