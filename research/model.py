"""Pooled linear model: predict mid(t+H) - micro(t) from features at t. Leave-one-session-out."""
import numpy as np, pandas as pd, sys
from load import *
LAGS = [10, 30, 60, 120, 240]
def features(n):
    t = tops(n); tr = trades(n)
    M = {s: t[s]["mid"].ffill() for s in SYMS}; U = {s: t[s]["micro"].ffill() for s in SYMS}
    tr["sq"] = np.where(tr.aggressor == "BUY", tr.qty, -tr.qty)
    flow = tr.pivot_table(index="batch", columns="symbol", values="sq", aggfunc="sum").reindex(M["HPPO"].index).fillna(0)
    F = pd.DataFrame(index=M["HPPO"].index)
    for s in SYMS:
        for L in LAGS: F[f"{s}_r{L}"] = U[s] - U[s].shift(L)
        for L in (30, 120): F[f"{s}_f{L}"] = flow[s].rolling(L).sum() / 1000
        F[f"{s}_um"] = U[s] - M[s]
    F["gapB"] = (U["HPPO"] - U["BNKY"]) / 2 - U["BRRY"]
    Y = {}
    for H in (30, 60, 120, 240, 480):
        for s in SYMS: Y[(s, H)] = M[s].shift(-H) - U[s]
    return F, Y
data = {n: features(n) for n in sessions()}
cols = list(next(iter(data.values()))[0].columns)
rows = []; betas = {}
for s in SYMS:
    for H in (30, 60, 120, 240, 480):
        preds, ys = [], []
        for hold in data:
            Xtr = pd.concat([data[n][0] for n in data if n != hold]); ytr = pd.concat([data[n][1][(s, H)] for n in data if n != hold])
            d = pd.concat([Xtr, ytr.rename("y")], axis=1).dropna()
            A = d[cols].values; lam = 1e-3 * len(A)
            b = np.linalg.solve(A.T @ A + lam * np.eye(len(cols)), A.T @ d.y.values)
            Xt, yt = data[hold][0], data[hold][1][(s, H)]
            dt = pd.concat([Xt, yt.rename("y")], axis=1).dropna()
            preds.append(dt[cols].values @ b); ys.append(dt.y.values)
        p, y = np.concatenate(preds), np.concatenate(ys)
        r2 = 1 - ((y - p) ** 2).mean() / (y ** 2).mean()
        rows.append(dict(sym=s, H=H, oos_r2=r2, pred_sd=p.std(), y_sd=y.std(), slope=np.polyfit(p, y, 1)[0]))
        betas[(s, H)] = b
print(pd.DataFrame(rows).round(3).to_string())
B = pd.DataFrame({k: v for k, v in betas.items() if k[1] == 120}, index=cols)
print(B.round(3).to_string())
