"""Fair-value model: in-sample vs out-of-sample R^2, coefficient stability, and regularised variants.

Run from ``research/`` with ``../.venv/bin/python overfit.py``.
OOS = the same 3 session folds as final_cv.py (fit on 16 sessions, score the other 8).
The ridge penalty for "std" variants is chosen by an inner 2-fold CV inside the 16 training
sessions only, so the held-out fold never influences any choice.
"""
import json
import sys

import numpy as np
import pandas as pd

from cv import FOLDS
from load import SYMS, sessions
from model import features

H = 120
ALL = None  # the 33 features, filled in main()
OWN = lambda s: [c for c in ALL if c.startswith(s + "_")]  # noqa: E731


def own_plus(s):
    """Own-symbol features plus the cross terms the data says matter (see NOTES.md)."""
    extra = {
        "HPPO": ["BRRY_f30", "BRRY_f120", "CRRT_f30", "BNKY_r10", "BNKY_r30"],
        "BNKY": ["HPPO_r10", "HPPO_f30", "BRRY_f30"],
        "CRRT": ["HPPO_r10", "HPPO_r30", "HPPO_f30", "HPPO_f120"],
        "BRRY": ["gapB", "HPPO_r10", "HPPO_r30", "HPPO_f120", "HPPO_um", "BNKY_um"],
    }[s]
    return OWN(s) + extra


def stack(names, data, s, cols):
    xs, ys = [], []
    for n in names:
        x, t = data[n]
        d = pd.concat([x[cols], t[(s, H)].rename("y")], axis=1).dropna()
        xs.append(d[cols].to_numpy())
        ys.append(d["y"].to_numpy())
    return np.vstack(xs), np.concatenate(ys)


def ridge(X, y, lam, standardise):
    """No intercept (as in alpha.py). If standardise, scale columns to unit RMS first, then map
    the coefficients back to raw units so alpha.py can use them unchanged."""
    n, k = X.shape
    scale = np.sqrt((X**2).mean(0)) if standardise else np.ones(k)
    Z = X / scale
    b = np.linalg.solve(Z.T @ Z + lam * n * np.eye(k), Z.T @ y)
    return b / scale


def r2(X, y, b):
    return 1 - np.mean((y - X @ b) ** 2) / np.mean(y**2)


def fit_spec(names, data, s, spec):
    cols = spec["cols"](s)
    lam = spec["lam"]
    if lam == "inner":  # choose lambda on the training sessions only (2-fold, by session halves)
        half = len(names) // 2
        a, b = names[:half], names[half:]
        best = None
        for cand in spec["grid"]:
            score = 0.0
            for tr, te in ((a, b), (b, a)):
                Xt, yt = stack(tr, data, s, cols)
                Xe, ye = stack(te, data, s, cols)
                score += r2(Xe, ye, ridge(Xt, yt, cand, True))
            if best is None or score > best[0]:
                best = (score, cand)
        lam = best[1]
    X, y = stack(names, data, s, cols)
    return cols, ridge(X, y, lam, spec["std"]), lam


def evaluate(spec, data, names):
    rows = {}
    coefs = {s: [] for s in SYMS}
    for s in SYMS:
        cols, b_all, lam_all = fit_spec(names, data, s, spec)
        X, y = stack(names, data, s, cols)
        ins = r2(X, y, b_all)
        num = den = 0.0
        lams = []
        for _, span in FOLDS:
            a, z = span.split("-")
            test = [n for n in names if a <= n <= z]
            train = [n for n in names if n not in test]
            cols, b, lam = fit_spec(train, data, s, spec)
            lams.append(lam)
            Xe, ye = stack(test, data, s, cols)
            num += np.sum((ye - Xe @ b) ** 2)
            den += np.sum(ye**2)
            coefs[s].append(pd.Series(b, index=cols))
        oos = 1 - num / den
        c = pd.concat(coefs[s], axis=1)
        same_sign = (np.sign(c).nunique(axis=1) == 1).mean()
        rows[s] = dict(IS=ins, OOS=oos, gap=ins - oos, k=len(cols), sign_stable=same_sign, lam=lams)
    return pd.DataFrame(rows).T


def specs():
    grid = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    return {
        "current: 33 feats, raw ridge 1e-3": dict(cols=lambda s: ALL, lam=1e-3, std=False),
        "33 feats, std ridge (inner CV)": dict(cols=lambda s: ALL, lam="inner", std=True, grid=grid),
        "own symbol only (8+gapB)": dict(cols=lambda s: OWN(s) + (["gapB"] if s == "BRRY" else []), lam="inner", std=True, grid=grid),
        "own + key cross (inner CV)": dict(cols=own_plus, lam="inner", std=True, grid=grid),
    }


def main():
    global ALL
    names = sessions()
    data = {n: features(n) for n in names}
    ALL = list(data[names[0]][0].columns)
    out = {}
    for label, spec in specs().items():
        res = evaluate(spec, data, names)
        out[label] = res
        print(f"\n{label}")
        print(res.round(3).to_string())
    summary = pd.DataFrame(
        {lab: {f"{s} {m}": r.loc[s, m] for s in SYMS for m in ("IS", "OOS")} for lab, r in out.items()}
    ).T
    print("\nIn-sample vs out-of-sample R^2 (target: mid 120 batches ahead minus micro)")
    print(summary.round(3).to_string())
    pd.to_pickle(out, "charts/data/overfit_r2.pkl")


if __name__ == "__main__":
    main()
