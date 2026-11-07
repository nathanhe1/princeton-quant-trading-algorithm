"""Nested-CV choice among candidate configurations (JSON: name -> {class attrs..., "_over": {GUI
params}}). Each candidate runs once on every session (3-fold out-of-sample coefficients); per outer
fold the choice is made on the other two folds (objective: mean of realistic stress, stress without
adverse, 10% share) and reported on the held-out one.
    ../.venv/bin/python nested_generic.py '<json>' <tag>
"""
import json
import sys

import pandas as pd

from cv import FOLDS, pick
from evalx import evaluate

ST, NA, S1 = "share=0.3+through=2+adverse", "share=0.3+through=2", "share=0.1"
CASES = [("base", 1.0), (ST, 1.0), (NA, 1.0), (S1, 1.0), (ST, -1.0)]
OBJ = [(ST, 1.0), (NA, 1.0), (S1, 1.0)]

if __name__ == "__main__":
    cands, tag = json.loads(sys.argv[1]), sys.argv[2]
    rows = []
    for name, cfg in cands.items():
        a = dict(cfg)
        over = a.pop("_over", None)
        for scen, mult in CASES:
            s, _ = evaluate("../strategies/alpha.py", coefs="raw", scenario=scen, attrs={**a, "COEF_MULT": mult}, over=over)
            rows += [dict(cand=name, scen=scen, mult=mult, sess=r.sess, mk=r.mk120) for _, r in s.iterrows()]
            print(f"{name:18} {scen[:22]:22} x{mult}: mean {s.mk120.mean():.0f} worst {s.mk120.min():.0f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f"charts/data/nested_{tag}.csv", index=False)
    obj = d[d.apply(lambda r: (r.scen, r.mult) in OBJ, axis=1)].groupby(["cand", "sess"]).mk.mean().unstack(0)
    picks = []
    for _, span in FOLDS:
        test = pick(span)
        train = [x for x in obj.index if x not in test]
        best = obj.loc[train].mean().idxmax()
        picks.append(best)
        print(f"outer fold {span}: pick {best}  (train {obj.loc[train].mean().round(0).to_dict()})")
    first = next(iter(cands))
    out = []
    for (scen, mult), g in d.groupby(["scen", "mult"]):
        piv = g.pivot(index="sess", columns="cand", values="mk")
        nested = pd.concat([piv.loc[pick(span), c] for (_, span), c in zip(FOLDS, picks)])
        out.append(dict(scen=scen[:22], coef=mult, base_cfg=piv[first].mean(), base_worst=piv[first].min(),
                        nested=nested.mean(), nested_worst=nested.min()))
    print(pd.DataFrame(out).round(0).to_string(index=False))
