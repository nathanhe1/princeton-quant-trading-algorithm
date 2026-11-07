"""One-factor-at-a-time screen for the redesign (capacity, per-symbol edge, skew, quote size, ETF
projection, volatility-scaled edge). Every config runs on all 24 sessions with 3-fold out-of-sample
coefficients under base, realistic stress and stress without the look-ahead adverse rule.
Per-session results: charts/data/screen.csv (for nested selection in screen_nested.py).
    ../.venv/bin/python screen.py
"""
import sys

import pandas as pd

sys.path.insert(0, "../strategies")
from alpha import Alpha  # noqa: E402
from evalx import evaluate  # noqa: E402

ST, NA = "share=0.3+through=2+adverse", "share=0.3+through=2"
C, E, K, Q = Alpha.CAP, Alpha.EDGE_MULT, Alpha.SKEW, Alpha.QUOTE
CONFIGS = {
    "current": {},
    "cap BRRY 600": {"CAP": {**C, "BRRY": 600}},
    "cap HPPO 600": {"CAP": {**C, "HPPO": 600}},
    "cap HPPO+BRRY 600": {"CAP": {**C, "HPPO": 600, "BRRY": 600}},
    "edge BNKY 0.75": {"EDGE_MULT": {**E, "BNKY": 0.75}},
    "edge BNKY 1.0": {"EDGE_MULT": {**E, "BNKY": 1.0}},
    "edge CRRT 0.9": {"EDGE_MULT": {**E, "CRRT": 0.9}},
    "skew x1.5": {"SKEW": {s: v * 1.5 for s, v in K.items()}},
    "skew x0.67": {"SKEW": {s: v / 1.5 for s, v in K.items()}},
    "quote HPPO+BRRY x1.5": {"QUOTE": {**Q, "HPPO": int(Q["HPPO"] * 1.5), "BRRY": int(Q["BRRY"] * 1.5)}},
    "etf project 0.5": {"ETF_PROJECT": 0.5},
    "etf project 1.0": {"ETF_PROJECT": 1.0},
    "vol gamma 0.5": {"VOL_GAMMA": 0.5},
    "vol gamma 1.0": {"VOL_GAMMA": 1.0},
}

if __name__ == "__main__":
    only = sys.argv[1:]
    rows = []
    for name, attrs in CONFIGS.items():
        if only and name not in only:
            continue
        for scen in ("base", ST, NA):
            s, f = evaluate("../strategies/alpha.py", coefs="raw", scenario=scen, attrs=attrs)
            for _, r in s.iterrows():
                rows.append(dict(cfg=name, scen=scen, sess=r.sess, mk=r.mk120, shares=r.shares))
            print(f"{name:22} {scen[:22]:22} mean {s.mk120.mean():6.0f} worst {s.mk120.min():6.0f}", flush=True)
        pd.DataFrame(rows).to_csv("charts/data/screen.csv", index=False)
    d = pd.DataFrame(rows)
    cur = d[d.cfg == "current"].set_index(["scen", "sess"]).mk
    out = []
    for name, g in d.groupby("cfg", sort=False):
        row = {"cfg": name}
        for scen, lab in (("base", "base"), (ST, "stress"), (NA, "noadv")):
            x = g[g.scen == scen].set_index("sess").mk
            diff = x - cur.loc[scen]
            row[lab] = x.mean()
            row[f"d_{lab}"] = diff.mean()
            row[f"se_{lab}"] = diff.std() / len(diff) ** 0.5
        out.append(row)
    print(pd.DataFrame(out).round(0).to_string(index=False))
