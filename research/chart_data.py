"""Compute and cache the numbers behind research/charts.py.

Run from ``research/`` with ``../.venv/bin/python chart_data.py [stress] [versions] [horizons]``
(no argument = all three). Writes CSVs to ``charts/data/``. Every number is out of sample:
sessions are scored with the fold coefficients that never saw them (same as final_cv.py).
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from bt import load_cls, run
from cv import FOLDS, pick

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts", "data")
CURRENT = "../strategies/alpha.py"
V1 = "../strategies/archive/alpha_v1.py"


def _jobs():
    return [(coef, name) for coef, span in FOLDS for name in pick(span)]


def _one(args):
    strategy, scenario_name, coef, name = args
    import stress  # imported in the worker so the monkeypatch lives in this process

    scenario = next(s for s in stress.SCENARIOS if s.name == scenario_name)
    with open(coef) as file:
        model = json.load(file)
    original = stress._install(scenario, 2026)
    try:
        summary, fills = run(load_cls(strategy), [name], model)
    finally:
        stress._Replay.rest_fills = original
    summary["scenario"] = scenario_name
    if fills.empty:
        return summary, pd.DataFrame()
    fills["shares"] = fills.q.abs()
    agg = fills.groupby(["sess", "sym", "liq"], as_index=False).agg(
        shares=("shares", "sum"), mk120=("mk120", "sum"), mk240=("mk240", "sum")
    )
    agg["scenario"] = scenario_name
    return summary, agg


def _grid(strategies, scenarios):
    jobs = [(path, sc, coef, name) for path in strategies for sc in scenarios for coef, name in _jobs()]
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        out = list(pool.map(_one, jobs, chunksize=2))
    labels = [path for path, *_ in jobs]
    summaries = pd.concat([o[0].assign(strategy=lab) for o, lab in zip(out, labels)])
    fills = pd.concat([o[1].assign(strategy=lab) for o, lab in zip(out, labels) if len(o[1])])
    return summaries, fills


def stress_data():
    import stress

    names = [s.name for s in stress.SCENARIOS]
    summaries, fills = _grid([CURRENT], names)
    summaries.to_csv(f"{OUT}/stress_sessions.csv", index=False)
    fills.to_csv(f"{OUT}/stress_fills.csv", index=False)
    print("stress:", summaries.groupby("scenario", sort=False).mk120.agg(["mean", "min"]).round(0).to_string())


def versions_data():
    import stress

    summaries, _ = _grid([V1, CURRENT], ["base", stress.REALISTIC.name])
    summaries["strategy"] = summaries.strategy.map({V1: "alpha_v1", CURRENT: "alpha (current)"})
    summaries.to_csv(f"{OUT}/versions_sessions.csv", index=False)
    print("versions:", summaries.groupby(["strategy", "scenario"]).mk120.agg(["mean", "min"]).round(0).to_string())


def _horizon(horizon):
    import model
    from load import sessions

    data = {name: model.features(name) for name in sessions()}
    r2, summary, _ = model.evaluate(horizon, load_cls(CURRENT), data)
    summary["H"] = horizon
    return r2, summary


def horizons_data():
    horizons = [120, 180, 240]
    with ProcessPoolExecutor(len(horizons)) as pool:
        out = list(pool.map(_horizon, horizons))
    r2 = pd.DataFrame([{"H": h, **r} for h, (r, _) in zip(horizons, out)])
    sessions = pd.concat([s for _, s in out])
    r2.to_csv(f"{OUT}/horizon_r2.csv", index=False)
    sessions.to_csv(f"{OUT}/horizon_sessions.csv", index=False)
    print("horizons:\n", r2.round(3).to_string(index=False))
    print(sessions.groupby("H").mk120.agg(["mean", "min"]).round(0).to_string())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    parts = sys.argv[1:] or ["stress", "versions", "horizons"]
    for part in parts:
        {"stress": stress_data, "versions": versions_data, "horizons": horizons_data}[part]()
