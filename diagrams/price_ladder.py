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
    ax.set_xlim(0, 14)
    for p in range(lo, hi + 1):
        ax.axhline(p, color=GRID, lw=0.6, zorder=0)
    ax.set_yticks(range(lo, hi + 1, 2))
    ax.tick_params(axis="y", colors=MUTED, labelcolor=INK2, labelsize=8.5, length=0)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # column headers
    for x, h in ((0.2, "market book"), (4.6, "our resting quotes"), (8.2, "where we think price is going")):
        ax.text(x, hi + 1.1, h, fontsize=8.5, color=MUTED, va="bottom")

    # take zones
    ax.axhspan(take_buy, hi + 0.5, color=BUY, alpha=0.08, zorder=0)
    ax.axhspan(lo - 0.5, take_sell, color=SELL, alpha=0.08, zorder=0)
    ax.axhline(take_buy, color=BUY, lw=1, zorder=1)
    ax.axhline(take_sell, color=SELL, lw=1, zorder=1)
    ax.text(13.8, take_buy - 0.25, f"BUY line {take_buy:.1f} = ask + 6.4", ha="right", va="top", fontsize=8.5, color=INK2)
    ax.text(13.8, take_sell + 0.25, f"SELL line {take_sell:.1f} = bid - 6.4", ha="right", va="bottom", fontsize=8.5, color=INK2)

    # column 1: market book, bar length = size
    ax.barh(ASK, AQ / 250, left=0.2, height=0.7, color=SELL, alpha=0.45)
    ax.barh(BID, BQ / 250, left=0.2, height=0.7, color=BUY, alpha=0.45)
    ax.text(0.2 + AQ / 250 + 0.15, ASK, f"ask {ASK} x {AQ}", fontsize=8.5, color=INK2, va="center")
    ax.text(0.2 + BQ / 250 + 0.15, BID, f"bid {BID} x {BQ}", fontsize=8.5, color=INK2, va="center")

    # column 2: our quotes
    ax.scatter([4.8], [our_ask], s=90, color=SELL, edgecolors=SURFACE, linewidths=2, zorder=4)
    ax.scatter([4.8], [our_bid], s=90, color=BUY, edgecolors=SURFACE, linewidths=2, zorder=4)
    ax.text(5.1, our_ask, f"our ask {our_ask}", va="center", fontsize=8.5, color=INK)
    ax.text(5.1, our_bid, f"our bid {our_bid}", va="center", fontsize=8.5, color=INK)

    # column 3: reference levels
    def level(y, label, color, lw):
        ax.plot([8.2, 9.2], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=3)
        ax.text(9.35, y, label, va="center", fontsize=8.5, color=INK)

    level(MICRO, f"microprice {MICRO:.1f}", MUTED, 2)
    level(fair, f"fair {fair:.1f} (micro {forecast:+.1f})", INK, 2.6)
    if position:
        level(centre, f"centre {centre:.2f} (fair - skew, long {position})", INK2, 1.6)

    if fair > take_buy or fair < take_sell:
        side = "BUY" if fair > take_buy else "SELL"
        px = ASK if side == "BUY" else BID
        ax.annotate("", xy=(0.2 + AQ / 250 + 0.1, px + 0.35), xytext=(8.1, fair),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
        ax.text(1.2, fair + 0.2, f"{side} 100 at {px} now (IOC)", fontsize=9.5, color=INK, fontweight="bold", va="bottom")
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=24)
    ax.text(0, -0.025, note, transform=ax.transAxes, fontsize=9.2, color=INK2, va="top", linespacing=1.45)


fig, axes = plt.subplots(1, 2, figsize=(15, 9.2), gridspec_kw={"wspace": 0.1})
fig.subplots_adjust(top=0.86, bottom=0.1)
panel(axes[0], "1. Normal: rest a bid and an ask (MAKE)", forecast=3.0, position=200,
      note="Model expects +3.0c. We are long 200, so both quotes lean down to sell it off.\n"
           "fair - ask = 2.5c, below 6.4c: no take. We wait to be filled.")
panel(axes[1], "2. Strong signal: cross the spread (TAKE)", forecast=8.0, position=0,
      note="Model expects +8.0c. fair - ask = 7.5c, more than 0.4c fee + 6c take_edge,\n"
           "so we buy now at the ask, and still rest quotes around the new fair.")
fig.text(0.125, 0.965, "How alpha.py decides to buy and sell (HPPO example, prices in cents)",
         ha="left", va="bottom", fontsize=14, fontweight="bold", color=INK)
fig.text(0.125, 0.958,
         "Quotes sit make_edge (2.5c) either side of fair, shifted against inventory, never crossing the market. "
         "A take needs fair beyond the touch by fee + take_edge (6.4c).",
         fontsize=9.8, color=INK2, va="top")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price-ladder.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
print("wrote", out)
