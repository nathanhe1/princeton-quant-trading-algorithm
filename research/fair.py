"""Which estimator of 'now' best predicts the mid H batches later? (proxy for V)"""
import numpy as np, pandas as pd
from load import *
H = 120
res = {}
coefs = []
for n in sessions():
    t = tops(n)
    M = {s: t[s]["mid"].ffill() for s in SYMS}
    U = {s: t[s]["micro"].ffill() for s in SYMS}
    imp = {"HPPO": U["BNKY"] + 2 * U["BRRY"], "BNKY": U["HPPO"] - 2 * U["BRRY"],
           "BRRY": (U["HPPO"] - U["BNKY"]) / 2}
    for s in ["HPPO", "BNKY", "BRRY", "CRRT"]:
        fut = M[s].shift(-H)
        cands = {"mid": M[s], "micro": U[s]}
        if s in imp:
            cands["implied"] = imp[s]
            for w in (0.25, 0.5, 0.75, 0.9):
                cands[f"blend{w}"] = w * imp[s] + (1 - w) * U[s]
        for k, v in cands.items():
            e = (fut - v).dropna()
            res.setdefault((s, k), []).append((e ** 2).mean())
        # regression of future mid change on (micro-mid), (implied-micro)
        y = (fut - M[s])
        X = pd.DataFrame({"micro_mid": U[s] - M[s]})
        if s in imp: X["imp_micro"] = imp[s] - U[s]
        d = pd.concat([y.rename("y"), X], axis=1).dropna()
        b = np.linalg.lstsq(d[X.columns].values, d.y.values, rcond=None)[0]
        coefs.append(dict(sess=n, sym=s, **dict(zip(X.columns, b))))
r = pd.Series({k: np.mean(v) for k, v in res.items()}).unstack()
print("MSE of mid(t+60s) - estimator(t), avg over sessions (cents^2):")
print(r.round(2).to_string())
c = pd.DataFrame(coefs).drop(columns="sess").groupby("sym")
print("\nregression of mid change over 60s on gaps (mean, std across sessions):")
print(pd.concat([c.mean().add_suffix("_m"), c.std().add_suffix("_sd")], axis=1).round(3).to_string())
