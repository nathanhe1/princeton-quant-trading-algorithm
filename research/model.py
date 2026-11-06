"""Compare fair-value horizons with the same held-out folds used by final_cv.py.

Run from ``research/`` with ``../.venv/bin/python model.py [horizons...]``.
Each fold's coefficients are fit on its other 16 sessions; both model fit and
strategy markouts are evaluated only on the 8 held-out sessions.
"""
import sys

import numpy as np
import pandas as pd

from bt import load_cls, run
from cv import FOLDS, pick
from load import SYMS, sessions, tops, trades

LAGS = [10, 30, 60, 120, 240]
FLOWS = [30, 120]
STRATEGY = "../strategies/alpha.py"


def features(name):
    top = tops(name)
    trade = trades(name)
    mids = {s: top[s]["mid"].ffill() for s in SYMS}
    micros = {s: top[s]["micro"].ffill() for s in SYMS}
    trade["signed_qty"] = np.where(
        trade.aggressor == "BUY", trade.qty, -trade.qty
    )
    flow = (
        trade.pivot_table(
            index="batch",
            columns="symbol",
            values="signed_qty",
            aggfunc="sum",
        )
        .reindex(mids["HPPO"].index)
        .fillna(0)
    )
    x = pd.DataFrame(index=mids["HPPO"].index)
    for symbol in SYMS:
        for lag in LAGS:
            x[f"{symbol}_r{lag}"] = micros[symbol] - micros[symbol].shift(lag)
        for window in FLOWS:
            x[f"{symbol}_f{window}"] = (
                flow[symbol].rolling(window).sum() / 1000
            )
        x[f"{symbol}_um"] = micros[symbol] - mids[symbol]
    x["gapB"] = (
        (micros["HPPO"] - micros["BNKY"]) / 2 - micros["BRRY"]
    )
    targets = {
        (symbol, horizon): mids[symbol].shift(-horizon) - micros[symbol]
        for symbol in SYMS
        for horizon in (120, 180, 240)
    }
    return x, targets


def fit(train_names, horizon, data, columns):
    coef = {}
    for symbol in SYMS:
        matrices, outcomes = [], []
        for name in train_names:
            x, targets = data[name]
            sample = pd.concat(
                [x, targets[(symbol, horizon)].rename("target")], axis=1
            ).dropna()
            matrices.append(sample[columns].to_numpy())
            outcomes.append(sample["target"].to_numpy())
        matrix = np.vstack(matrices)
        target = np.concatenate(outcomes)
        ridge = 1e-3 * len(matrix)
        beta = np.linalg.solve(
            matrix.T @ matrix + ridge * np.eye(len(columns)),
            matrix.T @ target,
        )
        coef[symbol] = {
            column: round(float(value), 5)
            for column, value in zip(columns, beta)
        }
    return {
        "H": horizon,
        "train": train_names,
        "lags": LAGS,
        "flows": FLOWS,
        "coef": coef,
    }


def evaluate(horizon, cls, data):
    all_names = sessions()
    columns = list(data[all_names[0]][0].columns)
    predictions = {s: [] for s in SYMS}
    outcomes = {s: [] for s in SYMS}
    session_results, fill_results = [], []

    for _, session_range in FOLDS:
        test_names = pick(session_range)
        train_names = [name for name in all_names if name not in test_names]
        model = fit(train_names, horizon, data, columns)
        for name in test_names:
            x, targets = data[name]
            for symbol in SYMS:
                sample = pd.concat(
                    [x, targets[(symbol, horizon)].rename("target")], axis=1
                ).dropna()
                prediction = (
                    sample[columns].to_numpy()
                    @ np.array([model["coef"][symbol][c] for c in columns])
                )
                predictions[symbol].append(prediction)
                outcomes[symbol].append(sample["target"].to_numpy())

            summary, fills = run(cls, [name], model)
            session_results.append(summary)
            if not fills.empty:
                fill_results.append(fills)

    r2 = {}
    for symbol in SYMS:
        predicted = np.concatenate(predictions[symbol])
        target = np.concatenate(outcomes[symbol])
        r2[symbol] = 1 - np.mean((target - predicted) ** 2) / np.mean(target**2)
    return r2, pd.concat(session_results), (
        pd.concat(fill_results) if fill_results else pd.DataFrame()
    )


def main():
    horizons = [int(value) for value in sys.argv[1:]] or [120, 180, 240]
    data = {name: features(name) for name in sessions()}
    cls = load_cls(STRATEGY)
    rows = []
    for horizon in horizons:
        r2, summary, fills = evaluate(horizon, cls, data)
        row = {
            "H": horizon,
            **{f"R2_{s}": r2[s] for s in SYMS},
            "mean_$120": summary.mk120.mean(),
            "worst_$120": summary.mk120.min(),
            "positive": int((summary.mk120 > 0).sum()),
            "mean_$240": summary.mk240.mean(),
        }
        rows.append(row)
        print(f"\nH={horizon}:")
        print(pd.DataFrame([row]).round(3).to_string(index=False))
        if not fills.empty:
            shares = fills.groupby(["sym", "liq"]).q.apply(
                lambda q: q.abs().sum()
            )
            markout = fills.groupby(["sym", "liq"]).mk120.sum()
            by_type = pd.DataFrame(
                {"shares": shares, "c120_per_share": markout / shares}
            )
            print(by_type.round(3).to_string())
    print("\nHorizon comparison:")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
