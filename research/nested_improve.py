"""Nested-CV choice of MAX_IMPROVE (ticks a quote may sit better than the external best price on its
side). Every candidate runs once on every session (3-fold out-of-sample coefficients); per outer fold
the choice is made on the other two folds and reported on the held-out one. Objective: mean of
realistic stress, stress without the look-ahead adverse rule, and 10% trade share without any
through rule (so the choice cannot ride on the through-rule's definition).
    ../.venv/bin/python nested_improve.py
"""
import pandas as pd

from cv import FOLDS, pick
from evalx import evaluate

ST, NA, S1 = "share=0.3+through=2+adverse", "share=0.3+through=2", "share=0.1"
CANDS = [None, 2, 3, 4, 5, 6]
CASES = [("base", 1.0), (ST, 1.0), (NA, 1.0), (S1, 1.0), (ST, -1.0)]
OBJ = [(ST, 1.0), (NA, 1.0), (S1, 1.0)]

if __name__ == "__main__":
    rows = []
    for c in CANDS:
        for scen, mult in CASES:
            s, _ = evaluate("../strategies/alpha.py", coefs="raw", scenario=scen,
                            attrs={"MAX_IMPROVE": c, "COEF_MULT": mult})
            for _, r in s.iterrows():
                rows.append(dict(cand=str(c), scen=scen, mult=mult, sess=r.sess, mk=r.mk120))
            print(f"improve {c}: {scen[:22]:22} x{mult}: mean {s.mk120.mean():.0f} worst {s.mk120.min():.0f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("charts/data/nested_improve.csv", index=False)
    obj = d[d.apply(lambda r: (r.scen, r.mult) in OBJ, axis=1)].groupby(["cand", "sess"]).mk.mean().unstack(0)
    picks = []
    for _, span in FOLDS:
        test = pick(span)
        train = [x for x in obj.index if x not in test]
        best = obj.loc[train].mean().idxmax()
        picks.append(best)
        print(f"outer fold {span}: pick {best}  (train {obj.loc[train].mean().round(0).to_dict()})")
    out = []
    for (scen, mult), g in d.groupby(["scen", "mult"]):
        piv = g.pivot(index="sess", columns="cand", values="mk")
        nested = pd.concat([piv.loc[pick(span), c] for (_, span), c in zip(FOLDS, picks)])
        out.append(dict(scen=scen[:22], coef=mult, off=piv["None"].mean(), off_worst=piv["None"].min(),
                        nested=nested.mean(), nested_worst=nested.min()))
    print(pd.DataFrame(out).round(0).to_string(index=False))
