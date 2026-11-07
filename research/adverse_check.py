"""Do we buy into falls and sell into rises? For one logger file (first session only):
1. maker and taker fills: the mid move in the 30 s before the fill and the 60 s after, per side;
2. inventory vs trend: correlation of our position with the next 60 s move (positive = we hold the
   right way) and with the last 60 s move (negative = we accumulate against the trend);
3. 60 s markout split into the spread we earn at the fill and the drift after it.
    ../.venv/bin/python adverse_check.py sandbox_runs/<file>.jsonl
"""
import sys
from collections import defaultdict

import numpy as np

from r1_report import SYMS, load

E = load(sys.argv[1])
B = [e for e in E if e["ev"] == "batch"]
F = [e for e in E if e["ev"] == "fill"]
mid = defaultdict(dict)
for b in B:
    for s, r in b["raw"].items():
        if r[0] is not None and r[2] is not None:
            mid[s][b["batch"]] = (r[0] + r[2]) / 2
print(f"{len(F)} fills, batches {B[0]['batch']}..{B[-1]['batch']}\n")
print("1. mid move (cents) in the 30 s before / 60 s after our fills, share-weighted")
print(f"   {'':12}{'buys: before':>14}{'after':>8}{'sells: before':>15}{'after':>8}   shares")
for s in SYMS:
    for liq in ("MAKER", "TAKER"):
        row = {}
        for side in ("BUY", "SELL"):
            pre = post = q = 0.0
            for f in F:
                if f["sym"] != s or liq not in f["liq"] or side not in f["side"]:
                    continue
                b = f["batch"]
                m0, mb, ma = mid[s].get(b), mid[s].get(b - 60), mid[s].get(b + 120)
                if None in (m0, mb, ma):
                    continue
                pre += f["qty"] * (m0 - mb)
                post += f["qty"] * (ma - m0)
                q += f["qty"]
            row[side] = (pre / q, post / q, q) if q else (0, 0, 0)
        print(f"   {s} {liq:6}  {row['BUY'][0]:+10.2f} {row['BUY'][1]:+7.2f}  {row['SELL'][0]:+12.2f} {row['SELL'][1]:+7.2f}"
              f"   {int(row['BUY'][2]):7} / {int(row['SELL'][2]):7}")

print("\n2. inventory vs trend (correlation over batches, sampled every 10 s)")
pos = defaultdict(int)
fills_at = defaultdict(list)
for f in F:
    fills_at[f["batch"]].append(f)
P = {s: {} for s in SYMS}
for b in B:
    for f in fills_at.get(b["batch"], []):
        pos[f["sym"]] += f["qty"] * (1 if "BUY" in f["side"] else -1)
    for s in SYMS:
        P[s][b["batch"]] = pos[s]
for s in SYMS:
    xs = [(P[s][b], mid[s][b + 120] - mid[s][b], mid[s][b] - mid[s][b - 120])
          for b in range(120, B[-1]["batch"] - 120, 20) if b in P[s] and {b, b + 120, b - 120} <= mid[s].keys()]
    a = np.array(xs, dtype=float)
    if len(a) > 10 and a[:, 0].std() > 0:
        print(f"   {s}: position vs next 60 s {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:+.2f}   "
              f"vs last 60 s {np.corrcoef(a[:, 0], a[:, 2])[0, 1]:+.2f}   "
              f"holding P&L {np.sum(a[:, 0] * a[:, 1]) / 6 / 100:+,.0f} $ (approx)")

print("\n3. maker 60 s markout = spread at the fill + drift after (cents/share)")
for s in SYMS:
    sp = dr = q = 0.0
    for f in F:
        if f["sym"] != s or "MAKER" not in f["liq"]:
            continue
        b, sg = f["batch"], (1 if "BUY" in f["side"] else -1)
        m0, ma = mid[s].get(b), mid[s].get(b + 120)
        if None in (m0, ma):
            continue
        sp += f["qty"] * sg * (m0 - f["px"])
        dr += f["qty"] * sg * (ma - m0)
        q += f["qty"]
    if q:
        print(f"   {s}: spread {sp / q:+.2f}  drift {dr / q:+.2f}  total {(sp + dr) / q:+.2f}")
