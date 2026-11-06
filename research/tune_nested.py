"""Nested CV for the strategy knobs (make_edge, size_scale, take_edge).

For each outer fold k (8 test sessions): every config is scored on the other 16 sessions, each
session with coefficients fit on the remaining single fold only (only{j}_std.json), so neither the
coefficients nor the choice ever see fold k. The pick maximises the realistic stressed mean subject
to base >= 90% of the default's base on the same 16 sessions. The pick is then scored on fold k with
the usual fold coefficients. Output: honest out-of-sample before/after.
Run from research/: ../.venv/bin/python tune_nested.py
"""
import itertools
import pandas as pd
from cv import pick
from evalx import evaluate_pairs

NEW = "../strategies/alpha.py"
REAL = "share=0.3+through=2+adverse"
FOLD = {1: "s01-s08", 2: "s09-s16", 3: "s17-s24"}
DEFAULT = {"make_edge": 2.5, "size_scale": 1.5, "take_edge": 6.0}
GRID = [dict(make_edge=m, size_scale=z, take_edge=t)
        for m, z, t in itertools.product((2.0, 2.5, 3.0, 3.5), (1.0, 1.5), (6.0, 10.0))]


def key(c):
    return f"make {c['make_edge']} size {c['size_scale']} take {c['take_edge']}"


def inner_pairs(k):
    i, j = [x for x in (1, 2, 3) if x != k]
    return [(f"only{j}_std.json", n) for n in pick(FOLD[i])] + [(f"only{i}_std.json", n) for n in pick(FOLD[j])]


def main():
    # score every config once on every (inner coef, session) pair it can appear in
    pairs = sorted({p for k in (1, 2, 3) for p in inner_pairs(k)})
    scores = {}
    for c in GRID:
        for scen in ("base", REAL):
            s, _ = evaluate_pairs(NEW, pairs, scenario=scen, over=c)
            s["coef"] = [p[0] for p in pairs]
            scores[(key(c), scen)] = s
        print("scored", key(c), flush=True)
    rows = []
    for k in (1, 2, 3):
        ip = set(inner_pairs(k))
        def inner(c, scen):
            s = scores[(key(c), scen)]
            m = [(cf, n) in ip for cf, n in zip(s.coef, s.sess)]
            return s[m].mk120.mean()
        floor = 0.9 * inner(DEFAULT, "base")
        ok = [c for c in GRID if inner(c, "base") >= floor]
        best = max(ok, key=lambda c: inner(c, REAL))
        test = [(f"fold{k}_std.json", n) for n in pick(FOLD[k])]
        for label, c in (("default", DEFAULT), ("picked", best)):
            for scen in ("base", REAL):
                s, _ = evaluate_pairs(NEW, test, scenario=scen, over=c)
                rows.append(dict(fold=k, which=label, config=key(c), scenario=scen,
                                 inner_mean=inner(c, scen), test_mean=s.mk120.mean(), test_worst=s.mk120.min()))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("charts/data/tune_nested.csv", index=False)
    print(df.round(0).to_string(index=False))
    for scen in ("base", REAL):
        d = df[df.scenario == scen].groupby("which")[["inner_mean", "test_mean"]].mean()
        print(scen, "\n", d.round(0).to_string())


if __name__ == "__main__":
    main()
