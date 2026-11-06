"""Worked example of alpha.py's buy/sell decision on one HPPO book (prices in cents).
Run: .venv/bin/python diagrams/price_ladder.py  ->  diagrams/price-ladder.png
The numbers follow the code: make_edge 2.5, take_edge 6, fee 0.4, SKEW 4.5, CAP 400, EDGE_MULT 1.
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BUY, SELL = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "text.parse_math": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

BID, BQ, ASK, AQ = 17977, 300, 17979, 100
MICRO = (BID * AQ + ASK * BQ) / (BQ + AQ)  # 17978.5: leans to the ask, the thin side
FEE, TAKE, MAKE, SKEW, CAP = 0.4, 6.0, 2.5, 4.5, 400


def panel(ax, title, forecast, position, note):
    fair = MICRO + forecast
    centre = fair - SKEW * position / CAP
    our_bid = min(math.floor(centre - MAKE), ASK - 1)
    our_ask = max(math.ceil(centre + MAKE), BID + 1)
    take_buy = ASK + FEE + TAKE  # fair above this -> buy IOC at the ask
    take_sell = BID - FEE - TAKE  # fair below this -> sell IOC at the bid
    lo, hi = 17968, 17990
    ax.set_ylim(lo - 0.5, hi + 0.5)
    ax.set_xlim(0, 10)
    for p in range(lo, hi + 1):
        ax.axhline(p, color=GRID, lw=0.6, zorder=0)
    ax.set_yticks(range(lo, hi + 1, 2))
    ax.tick_params(axis="y", colors=MUTED, labelcolor=INK2, labelsize=8.5, length=0)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # take zones
    ax.axhspan(take_buy, hi + 0.5, color=BUY, alpha=0.08, zorder=0)
    ax.axhspan(lo - 0.5, take_sell, color=SELL, alpha=0.08, zorder=0)
    ax.text(9.8, hi - 0.2, "TAKE zone: fair up here\n= buy now at the market ask", ha="right", va="top", fontsize=8.5, color=INK2)
    ax.text(9.8, lo + 0.2, "TAKE zone: fair down here\n= sell now at the market bid", ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.axhline(take_buy, color=BUY, lw=1, zorder=1)
    ax.axhline(take_sell, color=SELL, lw=1, zorder=1)

    # market book (left column)
    ax.barh(ASK, BQ / 100 * 0 + AQ / 100, left=0.3, height=0.7, color=SELL, alpha=0.35)
    ax.barh(BID, BQ / 100, left=0.3, height=0.7, color=BUY, alpha=0.35)
    ax.text(0.3, ASK + 0.55, f"market ask {ASK} x {AQ}", fontsize=8.5, color=INK2, va="bottom")
    ax.text(0.3, BID - 0.55, f"market bid {BID} x {BQ}", fontsize=8.5, color=INK2, va="top")

    # reference levels (middle column)
    def level(y, label, color, lw):
        ax.plot([3.6, 6.2], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=3)
        ax.text(6.35, y, label, va="center", fontsize=8.5, color=INK)

    level(MICRO, f"microprice {MICRO:.1f}", MUTED, 2)
    level(fair, f"fair = micro {forecast:+.1f} forecast = {fair:.1f}", INK, 2.6)
    if position:
        level(centre, f"centre = fair - skew for +{position} long = {centre:.2f}", INK2, 1.6)

    takes = fair > take_buy or fair < take_sell
    if takes:
        side = "BUY" if fair > take_buy else "SELL"
        px = ASK if side == "BUY" else BID
        ax.annotate("", xy=(2.2, px), xytext=(3.5, fair), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
        ax.text(2.2, px - 1.2, f"{side} 100 at {px} now (IOC)", fontsize=9, color=INK, fontweight="bold")
    # our quotes (always placed unless in a news pause)
    ax.scatter([3.0], [our_ask], s=90, color=SELL, edgecolors=SURFACE, linewidths=2, zorder=4)
    ax.scatter([3.0], [our_bid], s=90, color=BUY, edgecolors=SURFACE, linewidths=2, zorder=4)
    ax.text(2.75, our_ask, f"our ask {our_ask}", ha="right", va="center", fontsize=8.5, color=INK)
    ax.text(2.75, our_bid, f"our bid {our_bid}", ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=26)
    ax.text(0, 1.012, note, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")


fig, axes = plt.subplots(1, 2, figsize=(13, 7.4), gridspec_kw={"wspace": 0.12})
panel(axes[0], "1. Normal: rest a bid and an ask (MAKE)", forecast=3.0, position=200,
      note="Model expects +3.0c; we are long 200, so quotes lean down to sell. fair - ask = 2.5c, below 6.4c: no take.")
panel(axes[1], "2. Strong signal: cross the spread (TAKE)", forecast=8.0, position=0,
      note="Model expects +8.0c: fair - ask = 7.5c, more than 0.4c fee + 6c take_edge. Buy now, and still quote.")
fig.text(0.125, 0.985, "How alpha.py decides to buy and sell (HPPO example, prices in cents)", fontsize=13.5,
         fontweight="bold", color=INK, va="top")
fig.text(0.125, 0.955,
         "Quotes sit make_edge (2.5c) either side of fair, shifted against inventory, and never cross the market. "
         "Takes need fair beyond the touch by fee + take_edge (6.4c).",
         fontsize=9.5, color=INK2, va="top")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price-ladder.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
print("wrote", out)
