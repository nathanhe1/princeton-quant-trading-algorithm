"""Check the variant strategy computes the same features as feat_r2 (one session, backtest)."""
import json, os, sys, logging
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feat_r2 as F
from bt import load_cls
from load import DATA
from pqt_client.backtest_v2 import load_session
from pqt_client.v2 import backtest_strategy

def check(name, cand, sess="s03"):
    strat = load_cls(f"variants/alpha_{name}.py")()
    strat.model = json.load(open(f"variants/{name}_fold1.json"))
    rec = {}
    ob = strat.on_batch
    def wrap(state):
        ob(state)
        if hasattr(strat, "last_x") and state.batch >= 240:
            rec[state.batch] = dict(strat.last_x)
            del strat.last_x
    strat.on_batch = wrap
    logging.disable(logging.INFO)
    backtest_strategy(load_session(DATA, sess), strat)
    build, clipk = F.builder(cand)
    fr = build(F.load_all()[sess])
    clip = strat.model.get("clip") or {}
    worst = {}
    for b, x in rec.items():
        for k in strat.model["coef"]["HPPO"]:
            v = fr.at[b, k]
            if k in clip: v = max(-clip[k], min(clip[k], v))
            worst[k] = max(worst.get(k, 0), abs(x[k] - v))
    bad = {k: v for k, v in worst.items() if v > 1e-6}
    print(name, len(rec), "batches; max abs diff", max(worst.values()), "mismatched:", bad)

if __name__ == "__main__":
    for name, (cand, _) in __import__("make_variants").VARIANTS.items():
        check(name, cand)
