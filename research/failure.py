"""Model-failure acceptance table (ROADMAP P0): how a strategy behaves when its forecast is right,
wrong-signed, absent or overconfident, under the backtest fill rule and realistic stress.

    ../.venv/bin/python failure.py [strategy.py ...]   (default: ../strategies/alpha.py)

COEF_MULT multiplies every out-of-sample coefficient: 1 = the live model, -0.5 = a wrong-signed
model (what the sandbox looked like), 0 = no model (fair = microprice), 2 = overconfident.
Rows: $/session mean and worst, maker/taker shares and c/share, all 3-fold out-of-sample.
"""
import sys

import pandas as pd

from evalx import evaluate

ST = "share=0.3+through=2+adverse"
NA = "share=0.3+through=2"  # stress without the look-ahead adverse rule
CASES = [("base", 1.0), (ST, 1.0), (NA, 1.0), ("base", -0.5), (ST, -0.5), ("base", -1.0), (ST, -1.0),
         ("base", 0.0), (ST, 0.0), ("base", 2.0), (ST, 2.0)]


def row(path, scen, mult, attrs=None, over=None):
    s, f = evaluate(path, coefs="raw", scenario=scen, attrs={**(attrs or {}), "COEF_MULT": mult}, over=over)
    out = dict(strategy=path.split("/")[-1], scenario="stress" if "adverse" in scen else ("stress-noadv" if scen == NA else scen), coef=mult,
               mean=s.mk120.mean(), worst=s.mk120.min())
    for liq in ("MAKER", "TAKER"):
        x = f[f.liq.str.contains(liq)]
        sh = x.q.abs().sum()
        out[f"{liq.lower()}_sh"] = sh / len(s)
        out[f"{liq.lower()}_c"] = x.mk120.sum() / sh if sh else float("nan")
    return out


def table(paths, attrs=None, over=None):
    rows = [row(p, sc, m, attrs, over) for sc, m in CASES for p in paths]
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    paths = sys.argv[1:] or ["../strategies/alpha.py"]
    df = table(paths)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
