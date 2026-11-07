"""Summarise a sandbox_probe.py log: python research/sandbox_report.py [run.jsonl] (default: newest)."""

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

SYMS = ("HPPO", "BNKY", "CRRT", "BRRY")
H = 120  # markout horizon in batches (60 s)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main(path: Path) -> None:
    ev = [json.loads(line) for line in path.open()]
    by = defaultdict(list)
    for e in ev:
        by[e["ev"]].append(e)
    B = by["batch"]
    print(f"log {path.name}: {len(B)} batches ({B[0]['batch']}..{B[-1]['batch']})" if B else "no batches")
    start = by["start"][0] if by["start"] else {}
    print("params at start:", start.get("params"))
    for p in by["params"]:
        print("  params changed at", p["t"], p["changed"])

    # --- gaps: missed batches, timing, messages
    nums = [b["batch"] for b in B]
    missed = sum(b - a - 1 for a, b in zip(nums, nums[1:]) if b - a > 1)
    dts = [b["dt_ms"] for b in B]
    msgs = [b["msgs"] for b in B]
    gaps = [b["gap_ms"] for b in B if b["gap_ms"] is not None]
    print(f"\non_batch ms: median {st.median(dts):.2f}  p99 {pct(dts, .99):.2f}  max {max(dts):.2f}")
    print(f"batch gap ms: median {st.median(gaps):.0f}  p99 {pct(gaps, .99):.0f}  max {max(gaps):.0f}"
          f"  | batches skipped by the runner: {missed}")
    print(f"messages/batch: mean {st.mean(msgs):.1f}  p99 {pct(msgs, .99)}  max {max(msgs)}"
          f"  | batches at 20: {sum(m >= 20 for m in msgs)}")

    # --- rejects
    rc = Counter(r["code"] for r in by["reject"])
    print("\nrejects by code:", dict(rc) or "none")
    for code in rc:
        ex = next(r for r in by["reject"] if r["code"] == code)
        print(f"  e.g. {code}: batch {ex['batch']} {ex['detail']}")

    # --- external(): when we sit at the raw touch, does ext step past our size?
    print("\nexternal() at our price (raw touch -> external):")
    bad = 0
    shown = 0
    alone = Counter()
    joined = Counter()
    for b in B:
        for s in SYMS:
            r, x = b["raw"][s], b["ext"][s]
            if not x or None in r:
                continue
            for side, px, rem in b["ours"][s]:
                buy = "BUY" in side
                if (buy and px == r[0]) or (not buy and px == r[2]):
                    rq = r[1] if buy else r[3]
                    xp, xq = (x[0], x[1]) if buy else (x[2], x[3])
                    if rq > rem:
                        joined[s] += 1
                        ok = xp == px and xq == rq - rem
                    else:
                        alone[s] += 1
                        ok = (xp < px) if buy else (xp > px)
                    bad += not ok
                    if shown < 6:
                        print(f"  b{b['batch']} {s} {side[-4:]} ours {rem}@{px} raw {rq}@{r[0] if buy else r[2]}"
                              f" -> ext {xq}@{xp} {'ok' if ok else 'MISMATCH'}")
                        shown += 1
    print(f"  at touch with others: {dict(joined)}; alone: {dict(alone)}; mismatches: {bad}")
    print("  (open orders are logged after on_batch, so a quote placed this batch shows here only if"
          " it was already resting; treat a few mismatches as timing, not a bug)")

    # --- fills and markouts vs the raw mid H batches later
    mid = {s: {} for s in SYMS}
    for b in B:
        for s in SYMS:
            r = b["raw"][s]
            if None not in (r[0], r[2]):
                mid[s][b["batch"]] = (r[0] + r[2]) / 2
    F = by["fill"]
    print(f"\nfills: {len(F)}  shares {sum(f['qty'] for f in F)}")
    agg = defaultdict(lambda: [0, 0.0, 0])
    for f in F:
        m = mid[f["sym"]].get(f["batch"] + H)
        sgn = 1 if "BUY" in f["side"] else -1
        k = (f["sym"], f["liq"].split(".")[-1])
        agg[k][0] += f["qty"]
        if m is not None:
            agg[k][1] += sgn * (m - f["px"]) * f["qty"]
            agg[k][2] += f["qty"]
    print(f"  {'sym':5} {'liq':6} {'shares':>7} {'mkout c/sh (60s mid)':>22}")
    for (s, liq), (q, mk, qm) in sorted(agg.items()):
        print(f"  {s:5} {liq:6} {q:7d} {mk / qm if qm else float('nan'):22.2f}")
    tot_q = sum(v[2] for v in agg.values())
    if tot_q:
        print(f"  all: {sum(v[1] for v in agg.values()) / tot_q:.2f} c/share over {tot_q} marked shares;"
              f" ${sum(v[1] for v in agg.values()) / 100:.0f} total")
    secs = len(B) / 2
    print(f"  fill rate: {sum(f['qty'] for f in F) / (secs / 60):.0f} shares/min "
          f"(backtest proxy reference: see REPORT.md)")

    # --- news, halts
    print(f"\nnews: {len(by['news'])}")
    for n in by["news"]:
        nb = n["batch"]
        print(f"  b{nb}: {n['headline']!r} {n['body'][:120]!r}")
        for s in SYMS:
            m0 = mid[s].get(nb - 1) or mid[s].get(nb)
            path_ = [mid[s].get(nb + k) for k in (2, 10, 20, 40, 60, 120)]
            if m0:
                print(f"    {s}: mid {m0:.1f} -> +1s/5s/10s/20s/30s/60s "
                      + " ".join(f"{(p - m0):+.1f}" if p else "  ?" for p in path_))
    for p in by["news_pause"]:
        print(f"  pause_until {p['pause_until']} rebase_at {p['rebase_at']}")
    paused = [b["batch"] for b in B if b["paused"]]
    print(f"  batches paused: {len(paused)}")
    print(f"halts: {[(h['batch'], h['sym'], h['halted']) for h in by['halt']] or 'none'}")
    for h in by["halt"]:
        if not h["halted"]:
            after = [b for b in B if h["batch"] <= b["batch"] < h["batch"] + 20]
            quoted = sum(1 for b in after if b["ours"][h["sym"]])
            print(f"  {h['sym']} resumed b{h['batch']}: batches with our quotes in next 20: {quoted}")
    print("broadcasts:", [b["text"] for b in by["broadcast"]] or "none")
    print("reconnects:", len(by["reconnect"]))
    last = B[-1]
    print("\nend positions:", last["pos"], "| est PnL $%.2f" % (last["pnl"] / 1000))
    if by["stop"]:
        print("stop:", by["stop"][0])


if __name__ == "__main__":
    d = Path(__file__).resolve().parent / "sandbox_runs"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else max(d.glob("*.jsonl")))
