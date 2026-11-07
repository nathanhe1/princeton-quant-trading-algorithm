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


def blocks(path, arms, block=600):
    E = []
    for line in open(path, errors="replace"):
        try:
            E.append(json.loads(line))
        except ValueError:  # a line damaged while the file was being written
            pass
    B = [e for e in E if e["ev"] == "batch"]
    mid = {}
    for b in B:
        for s, r in b["raw"].items():
            if r[0] is not None and r[2] is not None:
                mid[(s, b["batch"])] = (r[0] + r[2]) / 2
    out = defaultdict(lambda: [0.0, 0])
    for f in (e for e in E if e["ev"] == "fill"):
        m = mid.get((f["sym"], f["batch"] + 120))
        if m is None:
            continue
        q = f["qty"] * (1 if "BUY" in f["side"] else -1)
        k = f["batch"] // block
        o = out[k]
        o[0] += q * (m - f["px"])
        o[1] += f["qty"]
    labels = [json.dumps(a, sort_keys=True) for a in arms]
    return [(labels[k % len(arms)], mk, sh) for k, (mk, sh) in sorted(out.items()) if sh]


def main(paths):
    pooled = defaultdict(lambda: defaultdict(list))
    for p, first in runs(paths):
        arms = json.loads(first["arms"])
        tag = (first.get("tag") or "untagged").split(":")[0]
        bl = blocks(p, arms)
        print(f"{p.split('/')[-1]} {first.get('tag')}: {len(bl)} blocks")
        for lab, mk, sh in bl:
            pooled[tag][lab].append((mk, sh))
    for tag, arms in pooled.items():
        print(f"\n== {tag} ==")
        for lab, xs in arms.items():
            mk, sh = sum(x[0] for x in xs), sum(x[1] for x in xs)
            per = [x[0] / 100 for x in xs]
            m = sum(per) / len(per)
            sd = math.sqrt(sum((v - m) ** 2 for v in per) / max(1, len(per) - 1))
            print(f"  {lab:40} blocks {len(xs):3}  {mk / sh:+.2f} c/sh on {sh:7} sh  ${mk / 100:+8.0f}  "
                  f"per block ${m:+.0f} ± {sd / math.sqrt(len(per)):.0f}")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob("sandbox_runs/*.jsonl")))
