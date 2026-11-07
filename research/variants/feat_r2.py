"""Candidate fair-value features vs the 33-feature baseline: 3-fold leave-sessions-out R^2.

    cd research && ../.venv/bin/python variants/feat_r2.py [candidate ...]

Same fit as fit.py's default (raw ridge, penalty 1e-3 * n, no intercept), same target
(mid 120 batches ahead minus micro now), same folds (fit on 16 sessions, score the other 8),
R^2 = 1 - SSE / sum(y^2) pooled over the 24 held-out sessions (as overfit.py). Rows t >= 240 only,
for every candidate, so all are scored on identical samples. Anything fitted from data (clip levels)
is computed on the training sessions only.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cv import FOLDS  # noqa: E402
from load import SYMS, sessions, tops, trades  # noqa: E402

H = 120
LAGS = [10, 30, 60, 120, 240]
FLOWS = [30, 120]
BIG = {"HPPO": 80, "BNKY": 160, "CRRT": 130, "BRRY": 40}  # ~ 90th pct of print size
EW_HL = (15, 60)  # EWMA flow half-lives, batches
VOL_WIN = 600  # realised-vol window (batches)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_cache.pkl")


def raw(n):
    t = tops(n)
    tr = trades(n)
    idx = t["HPPO"].index
    out = {}
    for s in SYMS:
        d = t[s].ffill()
        x = tr[tr.symbol == s]
        sq = np.where(x.aggressor == "BUY", x.qty, -x.qty)
        big = x.qty >= BIG[s]
        g = lambda v: pd.Series(v, index=x.batch.values).groupby(level=0).sum().reindex(idx).fillna(0)  # noqa: E731
        out[s] = pd.DataFrame({"mid": d["mid"], "micro": d["micro"], "bq": d["bq"], "aq": d["aq"], "spr": d["spr"],
                               "flow": g(sq), "fbig": g(np.where(big, sq, 0)), "fsmall": g(np.where(big, 0, sq)),
                               "vol": g(x.qty.values)})
    return out


def ewsum(x, hl):
    """e_t = d * e_{t-1} + x_t (a decayed sum, same units as a window sum)."""
    d = 0.5 ** (1 / hl)
    return x.ewm(alpha=1 - d, adjust=False).mean() / (1 - d)


def base(R):
    F = {}
    for s in SYMS:
        u, m = R[s]["micro"], R[s]["mid"]
        for L in LAGS:
            F[f"{s}_r{L}"] = u - u.shift(L)
        for L in FLOWS:
            F[f"{s}_f{L}"] = R[s]["flow"].rolling(L).sum() / 1000
        F[f"{s}_um"] = u - m
    F["gapB"] = (R["HPPO"]["micro"] - R["BNKY"]["micro"]) / 2 - R["BRRY"]["micro"]
    return pd.DataFrame(F)


def rvol(R, s):
    """Rolling realised vol of 10-batch micro changes, relative to a fixed long-run level."""
    d = R[s]["micro"].diff(10)
    v = np.sqrt((d ** 2).rolling(VOL_WIN, min_periods=60).mean())
    return v


# ---------------------------------------------------------------- candidates (frame builders)
def c_flow_size(R):
    F = base(R)
    for s in SYMS:
        for L in FLOWS:
            F = F.drop(columns=f"{s}_f{L}")
            F[f"{s}_fb{L}"] = R[s]["fbig"].rolling(L).sum() / 1000
            F[f"{s}_fs{L}"] = R[s]["fsmall"].rolling(L).sum() / 1000
    return F


def c_flow_ewma(R):
    F = base(R)
    for s in SYMS:
        for L in FLOWS:
            F = F.drop(columns=f"{s}_f{L}")
        for hl in EW_HL:
            F[f"{s}_e{hl}"] = ewsum(R[s]["flow"], hl) / 1000
    return F


def c_flow_ewma3(R):
    F = base(R)
    for s in SYMS:
        for L in FLOWS:
            F = F.drop(columns=f"{s}_f{L}")
        for hl in (5, 20, 80):
            F[f"{s}_e{hl}"] = ewsum(R[s]["flow"], hl) / 1000
    return F


def c_imb(R):
    F = base(R)
    for s in SYMS:
        bq, aq = R[s]["bq"], R[s]["aq"]
        F[f"{s}_imb"] = (bq - aq) / (bq + aq)
    return F


def c_imb_spr(R):
    F = c_imb(R)
    for s in SYMS:
        F[f"{s}_spr"] = R[s]["spr"]
        F[f"{s}_imbspr"] = F[f"{s}_imb"] * R[s]["spr"]  # signed: an imbalance matters more when wide
    return F


def c_basis(R):
    """gapB (= basis/2) is already a feature for every symbol. Add its changes and a mid version."""
    F = base(R)
    g = F["gapB"]
    for L in (10, 60):
        F[f"gapB_d{L}"] = g - g.shift(L)
    F["gapM"] = (R["HPPO"]["mid"] - R["BNKY"]["mid"]) / 2 - R["BRRY"]["mid"]
    return F


def c_basis_ew(R):
    F = base(R)
    F["gapB_ew60"] = F["gapB"].ewm(halflife=60, adjust=False).mean()
    return F


def c_volnorm(R):
    F = base(R)
    for s in SYMS:
        v = rvol(R, s)
        for L in LAGS:
            F[f"{s}_r{L}"] = F[f"{s}_r{L}"] / v
    return F


def c_volnorm_add(R):
    """Keep the raw returns, add vol-normalised short returns (the model can choose)."""
    F = base(R)
    for s in SYMS:
        v = rvol(R, s)
        for L in (10, 30):
            F[f"{s}_n{L}"] = F[f"{s}_r{L}"] / v
    return F


def c_sqrt(R):
    F = base(R)
    for c in F.columns:
        F[c] = np.sign(F[c]) * np.sqrt(np.abs(F[c]))
    return F


CANDIDATES = {"baseline": base, "flow_size": c_flow_size, "flow_ewma": c_flow_ewma, "flow_ewma3": c_flow_ewma3,
              "imb": c_imb, "imb_spr": c_imb_spr, "basis_d": c_basis, "basis_ew": c_basis_ew,
              "volnorm": c_volnorm, "volnorm_add": c_volnorm_add, "sqrt": c_sqrt,
              "clip4": base, "clip3": base}  # clips are applied in cv() with training-set levels
CLIP = {"clip4": 4.0, "clip3": 3.0}


def load_all():
    if os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    R = {n: raw(n) for n in sessions()}
    pd.to_pickle(R, CACHE)
    return R


def stack(frames, ys, names, cols):
    X = np.vstack([frames[n][cols].to_numpy()[240:] for n in names])
    y = np.concatenate([ys[n][240:] for n in names])
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    return X[ok], y[ok]


def ridge(X, y):
    return np.linalg.solve(X.T @ X + 1e-3 * len(X) * np.eye(X.shape[1]), X.T @ y)


def cv(name, Rall, return_coefs=False):
    names = sessions()
    frames = {n: CANDIDATES[name](Rall[n]) for n in names}
    cols = list(frames[names[0]].columns)
    res, fold_r2, coefs = {}, {}, {}
    for s in SYMS:
        ys = {n: (Rall[n][s]["mid"].shift(-H) - Rall[n][s]["micro"]).to_numpy() for n in names}
        num = den = 0.0
        per = []
        for k, (_, span) in enumerate(FOLDS, start=1):
            a, z = span.split("-")
            test = [n for n in names if a <= n <= z]
            train = [n for n in names if n not in test]
            Xt, yt = stack(frames, ys, train, cols)
            Xe, ye = stack(frames, ys, test, cols)
            clip = None
            if name in CLIP:
                clip = CLIP[name] * np.sqrt((Xt ** 2).mean(0))
                Xt, Xe = np.clip(Xt, -clip, clip), np.clip(Xe, -clip, clip)
            b = ridge(Xt, yt)
            e = np.sum((ye - Xe @ b) ** 2)
            num += e
            den += np.sum(ye ** 2)
            per.append(1 - e / np.sum(ye ** 2))
            coefs.setdefault(k, {})[s] = (dict(zip(cols, b)), None if clip is None else dict(zip(cols, clip)))
        res[s] = 1 - num / den
        fold_r2[s] = per
    return (res, fold_r2, coefs) if return_coefs else (res, fold_r2)


if __name__ == "__main__":
    Rall = load_all()
    which = sys.argv[1:] or list(CANDIDATES)
    rows, folds = {}, {}
    for c in which:
        r, f = cv(c, Rall)
        rows[c] = r
        folds[c] = f
        print(c, {s: round(v, 4) for s, v in r.items()}, flush=True)
    df = pd.DataFrame(rows).T
    if "baseline" in df.index:
        d = df.sub(df.loc["baseline"], axis=1)
        df["n_better"] = (d > 0).sum(1)
        wins = {c: sum(folds[c][s][k] > folds["baseline"][s][k] for s in SYMS for k in range(3)) for c in folds}
        df["fold_wins/12"] = pd.Series(wins)
        print("\nOOS R^2 minus baseline (x1000)")
        print((d * 1000).round(2).to_string())
    print("\nOOS R^2")
    print(df.round(4).to_string())
