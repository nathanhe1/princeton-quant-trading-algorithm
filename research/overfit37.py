"""Is the 37-feature fair-value model (imb+clip3, the one in alpha.py) overfit?

    ../.venv/bin/python overfit37.py            (from research/)

Same 3 session folds as every other test (fit on 16 sessions, score the other 8). For each symbol:
1. R^2 in-sample vs out-of-sample for the full model and for smaller / shrunk ones:
   full37   the live model (ridge 1e-3, i.e. least squares)
   own10    only the symbol's own 9 features + the ETF gap
   top-k    the k features with the largest |t| in the training folds (chosen on training data only)
   ridge λ  features standardised, penalty λ * n (a grid; read as a curve, not a choice)
2. Coefficient stability: share of the prediction's weight carried by coefficients whose sign is
   the same in all 3 fold fits.
3. Writes fold{k}_<variant>.json for the trading backtest (evalx.py coefs="_<variant>").
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "variants"))
import feat_r2 as F  # noqa: E402

FOLDS = [("s01", "s08"), ("s09", "s16"), ("s17", "s24")]
RIDGE = [0.0, 0.01, 0.1, 1.0]


def fit(X, y, lam_std=None):
    """lam_std None: the live fit (1e-3 * n on raw features). Else ridge on standardised features."""
    if lam_std is None:
        return F.ridge(X, y)
    sd = X.std(0) + 1e-12
    Z = X / sd
    b = np.linalg.solve(Z.T @ Z + lam_std * len(Z) * np.eye(Z.shape[1]), Z.T @ y)
    return b / sd


def r2(X, y, b):
    return 1 - np.sum((y - X @ b) ** 2) / np.sum(y ** 2)


def main():
    Rall = F.load_all()
    names = F.sessions()
    build, clipk = F.builder("imb+clip3")
    frames = {n: build(Rall[n]) for n in names}
    cols = list(frames[names[0]].columns)
    print(f"{len(cols)} features, {len(names)} sessions\n")
    variants = {}  # name -> fold -> sym -> (coef dict, clip dict)
    table = {}
    stab = {}
    for s in F.SYMS:
        ys = {n: (Rall[n][s]["mid"].shift(-F.H) - Rall[n][s]["micro"]).to_numpy() for n in names}
        fold_b = []
        for k, (a, z) in enumerate(FOLDS, start=1):
            test = [n for n in names if a <= n <= z]
            train = [n for n in names if n not in test]
            Xt, yt = F.stack(frames, ys, train, cols)
            Xe, ye = F.stack(frames, ys, test, cols)
            clip = clipk * np.sqrt((Xt ** 2).mean(0))
            Xt, Xe = np.clip(Xt, -clip, clip), np.clip(Xe, -clip, clip)
            clipd = dict(zip(cols, clip))

            def record(vname, b, idx=None):
                full = np.zeros(len(cols))
                if idx is None:
                    full[:] = b
                else:
                    full[idx] = b
                table.setdefault((vname, s), []).append(
                    (r2(Xt, yt, full), r2(Xe, ye, full), int((full != 0).sum())))
                variants.setdefault(vname, {}).setdefault(k, {})[s] = (dict(zip(cols, full)), clipd)
                return full

            b_full = record("full37", fit(Xt, yt))
            fold_b.append(b_full)
            own = [i for i, c in enumerate(cols) if c.startswith(s + "_") or c == "gapB"]
            record("own10", fit(Xt[:, own], yt), own)
            # t-stats on the training data, with an overlap correction (target spans H batches)
            res = yt - Xt @ b_full
            n_eff = len(yt) / F.H
            cov = np.linalg.inv(Xt.T @ Xt) * res.var() * (len(yt) / n_eff)
            t = np.abs(b_full) / np.sqrt(np.diag(cov))
            order = np.argsort(-t)
            for kk in (5, 10, 20):
                idx = sorted(order[:kk])
                record(f"top{kk}", fit(Xt[:, idx], yt), idx)
            for lam in RIDGE:
                record(f"ridge{lam:g}", fit(Xt, yt, lam))
        B = np.array(fold_b)  # 3 x p
        same = (np.sign(B) == np.sign(B[0])).all(0)
        # weight = |coef| x feature RMS (its typical contribution to the prediction)
        Xa, _ = F.stack(frames, ys, names, cols)
        w = np.abs(B.mean(0)) * np.sqrt((np.clip(Xa, -clipk * np.sqrt((Xa ** 2).mean(0)), None) ** 2).mean(0))
        stab[s] = (same.mean(), (w * same).sum() / w.sum(),
                   [cols[i] for i in np.argsort(-w)[:6]], [cols[i] for i in range(len(cols)) if not same[i] and w[i] > 0.05 * w.max()])

    print("R^2 x100, mean over 3 folds: in-sample / out-of-sample (gap)")
    vnames = list(dict.fromkeys(v for v, _ in table))
    print(f"{'variant':10}" + "".join(f"{s:>22}" for s in F.SYMS))
    for v in vnames:
        row = f"{v:10}"
        for s in F.SYMS:
            xs = table[(v, s)]
            i, o = np.mean([x[0] for x in xs]) * 100, np.mean([x[1] for x in xs]) * 100
            row += f"   {i:5.2f} / {o:5.2f} ({i - o:+.2f})"
        print(row)
    print("\nper-fold OOS R^2 x100, full37 vs own10 vs top10:")
    for s in F.SYMS:
        print(f"  {s}: " + "  ".join(
            f"{v} " + ",".join(f"{x[1] * 100:.2f}" for x in table[(v, s)]) for v in ("full37", "own10", "top10")))
    print("\ncoefficient sign stability across the 3 fold fits:")
    for s, (frac, wfrac, top, unstable) in stab.items():
        print(f"  {s}: {frac:.0%} of coefs keep their sign; they carry {wfrac:.0%} of the prediction weight")
        print(f"      biggest: {', '.join(top)}")
        if unstable:
            print(f"      unstable but not small: {', '.join(unstable)}")
    for vname, folds in variants.items():
        for k, per in folds.items():
            out = {"H": F.H, "variant": vname, "coef": {s: {c: round(float(v), 6) for c, v in cb.items()} for s, (cb, _) in per.items()},
                   "clip": {c: round(float(v), 4) for c, v in next(iter(per.values()))[1].items()},
                   "lags": F.LAGS, "flows": F.FLOWS}
            json.dump(out, open(os.path.join(HERE, f"fold{k}_{vname}.json"), "w"))
    print("\nwrote fold{1,2,3}_<variant>.json")


if __name__ == "__main__":
    main()
