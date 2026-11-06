"""Train vs test performance of the live model (raw ridge, 33 features, 120-batch target).

Two splits:
  folds : each of the 3 CV models (fold{k}.json) scored on its 16 training sessions (train) and
          its 8 held-out sessions (test)
  chrono: coef_train_s01_s16.json, trained on s01-s16, tested on s17-s24 (later sessions)
Reports R^2 per symbol and strategy markout ($/session) under the base fill rule and realistic stress.
Run from research/: ../.venv/bin/python train_test.py
"""
import json

import numpy as np
import pandas as pd

from cv import pick
from evalx import evaluate_pairs
from load import SYMS, sessions
from model import features

REAL = "share=0.3+through=2+adverse"
ALL = sessions()
SPLITS = [
    ("fold 1 (test s01-s08)", "fold1.json", [n for n in ALL if n > "s08"], pick("s01-s08")),
    ("fold 2 (test s09-s16)", "fold2.json", [n for n in ALL if not "s09" <= n <= "s16"], pick("s09-s16")),
    ("fold 3 (test s17-s24)", "fold3.json", [n for n in ALL if n < "s17"], pick("s17-s24")),
    ("chronological (train s01-s16, test s17-s24)", "coef_train_s01_s16.json", pick("s01-s16"), pick("s17-s24")),
]


def r2(data, names, coef, s):
    cols = list(coef[s])
    num = den = 0.0
    for n in names:
        x, t = data[n]
        d = pd.concat([x[cols], t[(s, 120)].rename("y")], axis=1).dropna()
        pred = d[cols].to_numpy() @ np.array([coef[s][c] for c in cols])
        num += np.sum((d.y.to_numpy() - pred) ** 2)
        den += np.sum(d.y.to_numpy() ** 2)
    return 1 - num / den


def main():
    data = {n: features(n) for n in ALL}
    rows = []
    for label, cf, train, test in SPLITS:
        coef = json.load(open(cf))["coef"]
        assert not set(train) & set(test)
        for part, names in (("train", train), ("test", test)):
            row = {"split": label, "part": part, "sessions": len(names)}
            for s in SYMS:
                row[f"R2 {s}"] = r2(data, names, coef, s)
            for scen, tag in (("base", "base"), (REAL, "realistic")):
                summ, fills = evaluate_pairs("../strategies/alpha.py", [(cf, n) for n in names], scenario=scen)
                mk = fills[fills.liq.str.contains("MAKER")]
                row[f"{tag} mean"] = summ.mk120.mean()
                row[f"{tag} worst"] = summ.mk120.min()
                row[f"{tag} maker c/sh"] = mk.mk120.sum() / mk.q.abs().sum()
            rows.append(row)
            print(label, part, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("charts/data/train_test.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
