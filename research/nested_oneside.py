"""Nested-CV choice of one-sided quoting (PROMPTS.md prompt 3; ROADMAP P1 item 10).
Each candidate (ONE_SIDE_K, ONE_SIDE_WIDEN) runs once on every session with 3-fold out-of-sample
coefficients; for each outer fold the choice is made on the other two folds and reported on the
held-out one. Objective: mean of realistic stress, stress without the look-ahead adverse rule, and
stress with a wrong-signed (x-1) and an overconfident (x2) model.
    ../.venv/bin/python nested_oneside.py
"""
import pandas as pd

from cv import FOLDS, pick
from evalx import evaluate

ST, NA = "share=0.3+through=2+adverse", "share=0.3+through=2"
CANDS = [(0.0, 0.0), (1.0, 2.0), (1.0, 0.0), (2.0, 2.0), (2.0, 0.0), (3.0, 0.0)]
CASES = [("base", 1.0), (ST, 1.0), (NA, 1.0), (ST, -1.0), (ST, 2.0)]
OBJ = [(ST, 1.0), (NA, 1.0), (ST, -1.0), (ST, 2.0)]

if __name__ == "__main__":
    rows = []
    for K, W in CANDS:
        for scen, mult in CASES:
            s, f = evaluate("../strategies/alpha.py", coefs="raw", scenario=scen,
                            attrs={"ONE_SIDE_K": K, "ONE_SIDE_WIDEN": W, "COEF_MULT": mult})
            mk = f[f.liq.str.contains("MAKER")]
            for _, r in s.iterrows():
                rows.append(dict(cand=f"K={K} W={W}", scen=scen, mult=mult, sess=r.sess, mk=r.mk120))
            print(f"K={K} W={W} {scen} x{mult}: mean {s.mk120.mean():.0f} worst {s.mk120.min():.0f} "
                  f"maker {mk.mk120.sum() / max(1, mk.q.abs().sum()):.2f}c/sh on {mk.q.abs().sum() / 24:,.0f}/session",
                  flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("charts/data/nested_oneside.csv", index=False)
    obj = d[d.apply(lambda r: (r.scen, r.mult) in OBJ, axis=1)].groupby(["cand", "sess"]).mk.mean().unstack(0)
    picks = []
    for _, span in FOLDS:
        test = pick(span)
        train = [x for x in obj.index if x not in test]
        best = obj.loc[train].mean().idxmax()
        picks.append(best)
        print(f"outer fold {span}: pick {best}  (train objective {obj.loc[train].mean().round(0).to_dict()})")
    out = []
    for (scen, mult), g in d.groupby(["scen", "mult"]):
        piv = g.pivot(index="sess", columns="cand", values="mk")
        nested = pd.concat([piv.loc[pick(span), c] for (_, span), c in zip(FOLDS, picks)])
        out.append(dict(scen=scen[:22], coef=mult, off=piv["K=0.0 W=0.0"].mean(), nested=nested.mean(),
                        nested_worst=nested.min(), off_worst=piv["K=0.0 W=0.0"].min()))
    print(pd.DataFrame(out).round(0).to_string(index=False))
