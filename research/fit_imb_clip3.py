"""Live fit of the imb+clip3 model (book imbalance at the touch, features winsorised at 3x their
training RMS) on all 24 sessions, with the same code that produced the cross-validated variant
(research/variants/feat_r2.py). Output: coef_all_imb_clip3.json, then
    ../.venv/bin/python bake.py coef_all_imb_clip3.json ../strategies/alpha.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "variants"))
import feat_r2 as F  # noqa: E402

if __name__ == "__main__":
    Rall = F.load_all()
    names = F.sessions()
    build, clipk = F.builder("imb+clip3")
    frames = {n: build(Rall[n]) for n in names}
    cols = list(frames[names[0]].columns)
    coef, clip = {}, None
    for s in F.SYMS:
        ys = {n: (Rall[n][s]["mid"].shift(-F.H) - Rall[n][s]["micro"]).to_numpy() for n in names}
        X, y = F.stack(frames, ys, names, cols)
        clip = clipk * np.sqrt((X ** 2).mean(0))
        b = F.ridge(np.clip(X, -clip, clip), y)
        coef[s] = {c: round(float(v), 6) for c, v in zip(cols, b)}
    out = {"H": F.H, "train": names, "lags": F.LAGS, "flows": F.FLOWS, "variant": "imb_clip3",
           "coef": coef, "clip": {c: round(float(v), 4) for c, v in zip(cols, clip)}}
    json.dump(out, open(os.path.join(HERE, "coef_all_imb_clip3.json"), "w"), indent=1)
    print("wrote coef_all_imb_clip3.json:", len(cols), "features,", len(names), "sessions")
