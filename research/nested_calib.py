"""Nested-CV choice of alpha.py's CALIB_FULL (online calibration, ROADMAP P0 item 1).
Every candidate runs once on every session (3-fold out-of-sample coefficients); the choice for each
outer fold is made on the other two folds' sessions and reported on the held-out one.
Objective: mean of realistic stress with the live model and with a wrong-signed (x-1) and an
overconfident (x2) model, so a choice must earn its keep and survive failure.
    ../.venv/bin/python nested_calib.py
"""
import pandas as pd

from cv import FOLDS, pick
from evalx import evaluate

ST = "share=0.3+through=2+adverse"
CANDS = (1.0, 0.75, 0.5)
CASES = [("base", 1.0), (ST, 1.0), (ST, -1.0), (ST, 2.0)]
OBJ = [(ST, 1.0), (ST, -1.0), (ST, 2.0)]

if __name__ == "__main__":
    rows = []
    for T in CANDS:
        for scen, mult in CASES:
            s, _ = evaluate("../strategies/alpha.py", coefs="raw", scenario=scen,
                            attrs={"CALIB_FULL": T, "COEF_MULT": mult})
            for _, r in s.iterrows():
                rows.append(dict(T=T, scen=scen, mult=mult, sess=r.sess, mk=r.mk120))
            print(f"T={T} {scen} x{mult}: mean {s.mk120.mean():.0f} worst {s.mk120.min():.0f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("charts/data/nested_calib.csv", index=False)
    obj = d[d.apply(lambda r: (r.scen, r.mult) in OBJ, axis=1)].groupby(["T", "sess"]).mk.mean().unstack(0)
    picks = []
    for _, span in FOLDS:
        test = pick(span)
        train = [x for x in obj.index if x not in test]
        best = obj.loc[train].mean().idxmax()
        picks.append(best)
        print(f"outer fold {span}: train objective {obj.loc[train].mean().round(0).to_dict()} -> T={best}")
    out = []
    for (scen, mult), g in d.groupby(["scen", "mult"]):
        piv = g.pivot(index="sess", columns="T", values="mk")
        nested = pd.concat([piv.loc[pick(span), T] for (_, span), T in zip(FOLDS, picks)])
        out.append(dict(scen="stress" if "adverse" in scen else scen, coef=mult,
                        **{f"T={T}": piv[T].mean() for T in CANDS}, nested=nested.mean(), nested_worst=nested.min()))
    print(pd.DataFrame(out).round(0).to_string(index=False))
