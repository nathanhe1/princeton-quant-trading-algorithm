"""3-fold out-of-sample evaluation: each fold's sessions run with coefficients fitted on the others."""
import json, sys, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from bt import load_cls, run, pick
FOLDS = [("fold1.json", "s01-s08"), ("fold2.json", "s09-s16"), ("fold3.json", "s17-s24")]
import os
def one(args):
    path, coef, sess, over, attrs = args
    return run(load_cls(path), [sess], json.load(open(coef)), over, attrs)
def cv(path, over=None, attrs=None):
    jobs = [(path, c, s, over, attrs) for c, rng in FOLDS for s in pick(rng)]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        out = list(ex.map(one, jobs))
    return pd.concat([o[0] for o in out]), pd.concat([o[1] for o in out if len(o[1])])
def summarize(fills):
    g = fills.groupby(["sym", "liq"]).agg(shares=("q", lambda x: x.abs().sum()), mk120=("mk120", "sum"), mk240=("mk240", "sum"))
    g["c120"] = g.mk120 / g.shares; g["c240"] = g.mk240 / g.shares
    g["mk120"] /= 2400; g["mk240"] /= 2400   # $ per session
    g["shares"] /= 24
    return g.round(2)
