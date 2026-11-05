"""Fit the fair-value model and write coefficients as JSON.
usage: python fit.py OUT.json [H] [first_session last_session]"""
import json, sys, numpy as np, pandas as pd
from load import *
LAGS = [10, 30, 60, 120, 240]
FLOWS = [30, 120]

def features(n):
    t = tops(n); tr = trades(n)
    M = {s: t[s]["mid"].ffill() for s in SYMS}; U = {s: t[s]["micro"].ffill() for s in SYMS}
    tr["sq"] = np.where(tr.aggressor == "BUY", tr.qty, -tr.qty)
    flow = tr.pivot_table(index="batch", columns="symbol", values="sq", aggfunc="sum").reindex(M["HPPO"].index).fillna(0)
    F = pd.DataFrame(index=M["HPPO"].index)
    for s in SYMS:
        for L in LAGS: F[f"{s}_r{L}"] = U[s] - U[s].shift(L)
        for L in FLOWS: F[f"{s}_f{L}"] = flow[s].rolling(L).sum() / 1000
        F[f"{s}_um"] = U[s] - M[s]
    F["gapB"] = (U["HPPO"] - U["BNKY"]) / 2 - U["BRRY"]
    return F, M, U

if __name__ == "__main__":
    out = sys.argv[1]; H = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    ss = sessions()
    if len(sys.argv) > 4 and sys.argv[3] == "ex":
        ss = [s for s in ss if not (sys.argv[4] <= s <= sys.argv[5])]
    elif len(sys.argv) > 4: ss = [s for s in ss if sys.argv[3] <= s <= sys.argv[4]]
    data = [features(n) for n in ss]
    cols = list(data[0][0].columns)
    coef = {}
    for s in SYMS:
        X = pd.concat([d[0] for d in data]); y = pd.concat([d[1][s].shift(-H) - d[2][s] for d in data])
        d = pd.concat([X, y.rename("y")], axis=1).dropna()
        A = d[cols].values
        b = np.linalg.solve(A.T @ A + 1e-3 * len(A) * np.eye(len(cols)), A.T @ d.y.values)
        coef[s] = {c: round(float(v), 5) for c, v in zip(cols, b)}
    json.dump({"H": H, "train": ss, "lags": LAGS, "flows": FLOWS, "coef": coef}, open(out, "w"), indent=1)
    print("wrote", out, "from", len(ss), "sessions")
