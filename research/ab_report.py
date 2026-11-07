"""Pool interleaved sandbox A/B runs (campaign.py / PROBE_ALTERNATE) by question and compare arms on
markout per share (fills vs the raw mid 120 batches later). Each 5-minute block is one observation.
    ../.venv/bin/python ab_report.py [jsonl ...]   (default: every run with arms in sandbox_runs/)
"""
import glob
import json
import math
import sys
from collections import defaultdict


def runs(paths):
    for p in paths:
        with open(p) as fh:
            first = json.loads(fh.readline())
        if first.get("ev") == "start" and first.get("arms"):
            yield p, first


def blocks(path, arms, block=600, detail=None):
    """One observation per (session segment, 5-minute block). A logger that runs on into the next
    session restarts the batch count; each restart opens a new segment so the sessions stay apart."""
    E = []
    for line in open(path, errors="replace"):
        try:
            E.append(json.loads(line))
        except ValueError:  # a line damaged while the file was being written
            pass
    seg, prev, mid, fills = 0, -1, {}, []
    for e in E:
        if e["ev"] == "batch":
            if e["batch"] < prev:
                seg += 1
            prev = e["batch"]
            for s, r in e["raw"].items():
                if r[0] is not None and r[2] is not None:
                    mid[(seg, s, e["batch"])] = (r[0] + r[2]) / 2
        elif e["ev"] == "fill":
            fills.append((seg, e))
    labels = [json.dumps(a, sort_keys=True) for a in arms]
    out = defaultdict(lambda: [0.0, 0])
    for seg, f in fills:
        m = mid.get((seg, f["sym"], f["batch"] + 120))
        if m is None:
            continue
        q = f["qty"] * (1 if "BUY" in f["side"] else -1)
        k = f["batch"] // block
        o = out[(seg, k)]
        o[0] += q * (m - f["px"])
        o[1] += f["qty"]
        if detail is not None:
            d = detail[(labels[k % len(arms)], f["sym"], "M" if "MAKER" in f["liq"] else "T")]
            d[0] += q * (m - f["px"])
            d[1] += f["qty"]
    return [(labels[k % len(arms)], seg, mk, sh) for (seg, k), (mk, sh) in sorted(out.items()) if sh]


def main(paths):
    pooled = defaultdict(lambda: defaultdict(list))
    detail = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for p, first in runs(paths):
        arms = json.loads(first["arms"])
        tag = (first.get("tag") or "untagged").split(":")[0]
        bl = blocks(p, arms, detail=detail[tag])
        nseg = len({b[1] for b in bl})
        print(f"{p.split('/')[-1]} {first.get('tag')}: {len(bl)} blocks over {nseg} session(s)")
        for lab, seg, mk, sh in bl:
            pooled[tag][lab].append((mk, sh, f"{p}#{seg}"))
    for tag, arms in pooled.items():
        print(f"\n== {tag} ==")
        for lab, xs in arms.items():
            mk, sh = sum(x[0] for x in xs), sum(x[1] for x in xs)
            per = [x[0] / 100 for x in xs]
            m = sum(per) / len(per)
            sd = math.sqrt(sum((v - m) ** 2 for v in per) / max(1, len(per) - 1))
            by_sess = defaultdict(lambda: [0.0, 0])
            for x in xs:
                by_sess[x[2]][0] += x[0]
                by_sess[x[2]][1] += x[1]
            sess = "  ".join(f"{v[0] / v[1]:+.2f}" for v in by_sess.values() if v[1])
            print(f"  {lab:40} blocks {len(xs):3}  {mk / sh:+.2f} c/sh on {sh:7} sh  ${mk / 100:+8.0f}  "
                  f"per block ${m:+.0f} ± {sd / math.sqrt(len(per)):.0f}   by session c/sh: {sess}")
        if "-v" in sys.argv:
            for (lab, sym, liq), (mk, sh) in sorted(detail[tag].items()):
                if sh:
                    print(f"     {lab:40} {sym} {liq} {mk / sh:+6.2f} c/sh on {sh:7}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "-v"]
    main(args or sorted(glob.glob("sandbox_runs/*.jsonl")))
