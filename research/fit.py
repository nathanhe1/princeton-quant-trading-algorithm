"""Fit the fair-value model and write coefficients as JSON.
usage: python fit.py OUT.json [H] [first_session last_session | ex first last] [--scaled [--lam S=v,...]]

Default (baked in alpha.py): ridge with penalty 1e-3 x n on raw, unscaled features.
--scaled: ridge on features scaled to unit RMS (no intercept, no centring), per-symbol penalty
chosen by nested CV in overfit.py, mapped back to raw units. Higher OOS R^2 but worse markout
under realistic stress (research/REPORT.md section 1), so not used. --raw is accepted and ignored."""
import json, sys, numpy as np, pandas as pd
from load import *
LAGS = [10, 30, 60, 120, 240]
FLOWS = [30, 120]
# chosen by the inner CV in overfit.py (median over the three outer folds)
LAM = {"HPPO": 0.3, "BNKY": 0.3, "CRRT": 0.1, "BRRY": 0.3}

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

def solve(A, y, lam, raw):
    if raw:
        return np.linalg.solve(A.T @ A + 1e-3 * len(A) * np.eye(A.shape[1]), A.T @ y)
    scale = np.sqrt((A ** 2).mean(0))
    Z = A / scale
    return np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ y) / scale

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raw = "--scaled" not in sys.argv
    lam = dict(LAM)
    if "--lam" in sys.argv:
        for kv in sys.argv[sys.argv.index("--lam") + 1].split(","):
            k, v = kv.split("="); lam[k] = float(v)
        args = [a for a in args if a != sys.argv[sys.argv.index("--lam") + 1]]
    out = args[0]; H = int(args[1]) if len(args) > 1 else 120
    ss = sessions()
    if len(args) > 3 and args[2] == "ex":
        ss = [s for s in ss if not (args[3] <= s <= args[4])]
    elif len(args) > 3: ss = [s for s in ss if args[2] <= s <= args[3]]
    data = [features(n) for n in ss]
    cols = list(data[0][0].columns)
    coef = {}
    for s in SYMS:
        X = pd.concat([d[0] for d in data]); y = pd.concat([d[1][s].shift(-H) - d[2][s] for d in data])
        d = pd.concat([X, y.rename("y")], axis=1).dropna()
        b = solve(d[cols].values, d.y.values, lam[s], raw)
        coef[s] = {c: round(float(v), 5) for c, v in zip(cols, b)}
    meta = {"fit": "raw ridge 1e-3" if raw else "scaled ridge", "lam": None if raw else lam}
    json.dump({"H": H, "train": ss, "lags": LAGS, "flows": FLOWS, **meta, "coef": coef}, open(out, "w"), indent=1)
    print("wrote", out, "from", len(ss), "sessions", meta)
