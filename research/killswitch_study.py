"""How deep does realized PnL fall within a session? Sets the kill-switch floor: it must never
trip in healthy sessions and should trip in broken ones. Replays alpha.py (3-fold out-of-sample
coefficients) and records the lowest realized PnL reached at any batch.
    ../.venv/bin/python killswitch_study.py
"""
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pqt_client.backtest_v2 import load_session
from pqt_client.v2 import backtest_strategy

import stress
from bt import load_cls
from cv import FOLDS, pick
from load import DATA

ST = "share=0.3+through=2+adverse"


def one(job):
    name, coef, scen, mult, data = job
    logging.disable(logging.INFO)
    cls = load_cls(os.path.abspath("../strategies/alpha.py"))
    lows = []

    class T(cls):
        def on_batch(self, state):
            super().on_batch(state)
            lows.append(state.portfolio.realized_pnl_mills)

    t = T()
    if coef:
        m = json.load(open(coef))
        t.model = {**m, "coef": {s: {k: v * mult for k, v in c.items()} for s, c in m["coef"].items()}}
    sc = next(s for s in stress.SCENARIOS if s.name == scen)
    orig = stress._install(sc, 2026) if scen != "base" else None
    try:
        r = backtest_strategy(load_session(data, name), t)
    finally:
        if orig is not None:
            stress._Replay.rest_fills = orig
    return name, scen, mult, min(lows) / 1000, r.realized_pnl_mills / 1000


if __name__ == "__main__":
    jobs = []
    for k, (cf, span) in enumerate(FOLDS, 1):
        for n in pick(span):
            for scen, mult in (("base", 1.0), (ST, 1.0), (ST, -1.0)):
                jobs.append((n, cf, scen, mult, DATA))
    jobs.append(("sandbox-0201", None, "base", 1.0, "sandbox_data"))
    jobs.append(("sandbox-0201", None, ST, 1.0, "sandbox_data"))
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(one, jobs))
    import pandas as pd
    d = pd.DataFrame(out, columns=["sess", "scen", "mult", "min_realized", "end_realized"])
    d.to_csv("charts/data/killswitch_study.csv", index=False)
    for (scen, mult), g in d.groupby(["scen", "mult"]):
        print(f"{scen[:22]:22} x{mult:+g}: lowest realized PnL in session: min {g.min_realized.min():8.0f}  "
              f"5th pct {np.percentile(g.min_realized, 5):8.0f}  median {g.min_realized.median():8.0f} | end median {g.end_realized.median():8.0f}  (n={len(g)})")
