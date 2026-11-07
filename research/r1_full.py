"""R1 across the logger restarts: markout by symbol/maker-taker, BNKY detail, every news item."""
import json, sys
from collections import defaultdict
import numpy as np
FILES = sys.argv[1:]
mid, F, B, N = {}, [], {}, []
for p in FILES:
    for l in open(p, errors="replace"):
        try: e = json.loads(l)
        except ValueError: continue
        if e["ev"] == "batch":
            B[e["batch"]] = e
            for s, r in e["raw"].items():
                if r[0] and r[2]: mid[(s, e["batch"])] = (r[0] + r[2]) / 2
        elif e["ev"] == "fill": F.append(e)
        elif e["ev"] == "news": N.append(e)
SY = ("HPPO", "BNKY", "CRRT", "BRRY")
def mk(f, h=120):
    m = mid.get((f["sym"], f["batch"] + h))
    if m is None: return None
    sg = 1 if "BUY" in f["side"] else -1
    return sg * f["qty"] * (m - f["px"]) - (0.4 if "TAKER" in f["liq"] else -0.2) * f["qty"]
agg = defaultdict(lambda: [0.0, 0])
for f in F:
    v = mk(f)
    if v is None: continue
    k = (f["sym"], "T" if "TAKER" in f["liq"] else "M")
    agg[k][0] += v; agg[k][1] += f["qty"]
print("1. R1 60 s markout by symbol (vs visible mid, fees in)")
tot = 0
for s in SY:
    row = []
    for lq in "MT":
        v, q = agg[(s, lq)]; tot += v
        row.append(f"{'maker' if lq == 'M' else 'taker'} ${v/100:+6.0f} on {q:6,} ({v/max(q,1):+.2f}c)")
    print(f"   {s}: " + " | ".join(row))
print(f"   total ${tot/100:+.0f}")
print("\n2. BNKY")
bf = [f for f in F if f["sym"] == "BNKY" and "MAKER" in f["liq"]]
for side in ("BUY", "SELL"):
    xs = [f for f in bf if side in f["side"]]
    sp = sum((1 if side == "BUY" else -1) * f["qty"] * (mid.get(("BNKY", f["batch"]), f["px"]) - f["px"]) for f in xs)
    dr = sum((1 if side == "BUY" else -1) * f["qty"] * (mid.get(("BNKY", f["batch"] + 120), 0) - mid.get(("BNKY", f["batch"]), 0)) for f in xs if ("BNKY", f["batch"] + 120) in mid)
    q = sum(f["qty"] for f in xs)
    print(f"   maker {side}: {q:,} sh, spread {sp/q:+.2f}c, drift {dr/q:+.2f}c")
pos = [(b, B[b]["pos"]["BNKY"]) for b in sorted(B)]
at_cap = sum(1 for _, p in pos if abs(p) >= 1400) / len(pos)
print(f"   time at |pos| >= 1400 (soft cap 1500): {at_cap:.0%}")
xs = [(B[b]["pos"]["BNKY"], mid[("BNKY", b + 120)] - mid[("BNKY", b)]) for b in sorted(B)[::20] if ("BNKY", b + 120) in mid and ("BNKY", b) in mid]
a = np.array(xs, float); print(f"   position vs next 60 s move corr {np.corrcoef(a[:,0], a[:,1])[0,1]:+.2f}")
spread = [B[b]["raw"]["BNKY"][2] - B[b]["raw"]["BNKY"][0] for b in B if B[b]["raw"]["BNKY"][0] and B[b]["raw"]["BNKY"][2]]
print(f"   BNKY spread median {np.median(spread):.0f} ticks; 60 s move sd {np.std([mid[('BNKY',b+120)]-mid[('BNKY',b)] for b in sorted(B)[::20] if ('BNKY',b+120) in mid and ('BNKY',b) in mid]):.1f}c")
# by quarter of session
for lo in range(0, 5400, 1350):
    v = sum(mk(f) or 0 for f in bf if lo <= f["batch"] < lo + 1350); q = sum(f["qty"] for f in bf if lo <= f["batch"] < lo + 1350)
    print(f"   BNKY maker minutes {lo//120:2}-{(lo+1350)//120:2}: ${v/100:+.0f} on {q:,} ({v/max(q,1):+.2f}c)")
print("\n3. news")
for n in N:
    b = n["batch"]
    print(f"   b{b} ({b/120:.1f} min): {n['headline'][:90]}")
    for s in SY:
        m0 = mid.get((s, b - 1))
        if m0:
            print(f"      {s}: " + " ".join(f"{k/2:g}s {mid[(s, b+k)]-m0:+.0f}" for k in (2, 10, 20, 60, 120, 240) if (s, b + k) in mid))
