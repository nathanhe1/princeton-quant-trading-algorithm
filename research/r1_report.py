"""After R1: turn the logger's file into the numbers that set R2 (RUNBOOK "After R1, before R2").

    ../.venv/bin/python r1_report.py sandbox_runs/<R1 file>.jsonl

1. Markout curves at 1, 10, 30, 60 and 120 s, by symbol and maker/taker, against our own mid.
   Put these next to the official report's curves (true value): if the official 60 s figure looks
   like our 120 s one, the true value leads the mid and a longer forecast horizon is worth
   re-testing for R2.
2. Our share of market volume per symbol (the real fill share; the backtest assumes 100%, stress
   30% or 10%).
3. News: every item, what read_news made of it, and the mid path 1-120 s after it (repricing speed:
   news_pause and news_trade).
4. Maker c/share vs the backtest's v10 norms (base 1.76, stress ~1.2-2.0).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "strategies"))
from alpha import read_news  # noqa: E402

SYMS = ("HPPO", "BNKY", "CRRT", "BRRY")
HORIZONS = (2, 20, 60, 120, 240)  # batches: 1, 10, 30, 60, 120 s


def load(path):
    E = []
    for line in open(path, errors="replace"):
        try:
            E.append(json.loads(line))
        except ValueError:
            pass
    return E


def main(path):
    E = load(path)
    B = [e for e in E if e["ev"] == "batch"]
    F = [e for e in E if e["ev"] == "fill"]
    mid = {}
    for b in B:
        for s, r in b["raw"].items():
            if r[0] is not None and r[2] is not None:
                mid[(s, b["batch"])] = (r[0] + r[2]) / 2
    last = B[-1]["batch"] if B else 0
    print(f"{Path(path).name}: batches {B[0]['batch'] if B else '-'}..{last}, {len(F)} fills")

    print("\n1. markout c/share vs our mid h later  (h = " + ", ".join(f"{h / 2:g}s" for h in HORIZONS) + ")")
    agg = defaultdict(lambda: [[0.0, 0] for _ in HORIZONS])
    for f in F:
        sgn = 1 if "BUY" in f["side"] else -1
        liq = "MAKER" if "MAKER" in f["liq"] else "TAKER"
        for i, h in enumerate(HORIZONS):
            m = mid.get((f["sym"], min(f["batch"] + h, last)))
            if m is not None:
                a = agg[(f["sym"], liq)][i]
                a[0] += sgn * f["qty"] * (m - f["px"])
                a[1] += f["qty"]
    for (s, liq), rows in sorted(agg.items()):
        print(f"   {s} {liq:5} " + "  ".join(f"{r[0] / r[1]:+6.2f}" if r[1] else "     -" for r in rows)
              + f"   ({rows[3][1]} shares)")

    print("\n2. our share of market volume (public tape includes our fills)")
    tape = defaultdict(int)
    for b in B:
        for t in b.get("tape", []):
            tape[t[0]] += t[2]
    ours = defaultdict(int)
    for f in F:
        ours[f["sym"]] += f["qty"]
    for s in SYMS:
        if tape.get(s):
            print(f"   {s}: {ours[s]:,} of {tape[s]:,} shares = {ours[s] / tape[s]:.0%}")
        else:
            print(f"   {s}: ours {ours[s]:,} (no tape logged: re-run with a logger that records it)")

    print("\n3. news")
    for n in (e for e in E if e["ev"] == "news"):
        print(f"   b{n['batch']}: {n['headline']!r} -> read {read_news(n['headline'], n.get('body', ''))}")
        for s in SYMS:
            m0 = mid.get((s, n["batch"] - 1)) or mid.get((s, n["batch"]))
            if m0:
                path_ = [mid.get((s, n["batch"] + k)) for k in (2, 10, 20, 60, 120, 240)]
                print(f"      {s}: 1s/5s/10s/30s/60s/120s " + " ".join(f"{p - m0:+.1f}" if p else "?" for p in path_))

    print("\n4. maker c/share at 60 s vs backtest v10 (base 1.76; realistic stress ~1.2-2.0):")
    tot = [0.0, 0]
    for (s, liq), rows in agg.items():
        if liq == "MAKER":
            tot[0] += rows[3][0]
            tot[1] += rows[3][1]
    if tot[1]:
        print(f"   all symbols: {tot[0] / tot[1]:+.2f} c/share on {tot[1]:,} shares")


if __name__ == "__main__":
    main(sys.argv[1])
