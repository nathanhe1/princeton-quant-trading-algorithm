"""3-fold out-of-sample markouts for the feature variants vs strategies/alpha.py (coefs="raw").

    cd research && STRESS_SEED=2026 ../.venv/bin/python variants/eval_variants.py OUT.pkl [name ...]
Names: baseline (alpha.py + fold{k}.json), baseline_refit (alpha.py + variants/baseline_refit_fold{k}),
or any variant from make_variants.VARIANTS. Coef files are relative to research/ (evalx joins them).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cv import FOLDS, pick  # noqa: E402
from evalx import evaluate, evaluate_pairs  # noqa: E402

SCEN = os.environ.get("SCEN", "base,share=0.3+through=2+adverse").split(",")


def run(name, scen):
    if name == "baseline":
        return evaluate("../strategies/alpha.py", coefs="raw", scenario=scen)
    path = "../strategies/alpha.py" if name == "baseline_refit" else f"variants/alpha_{name}.py"
    pairs = [(f"variants/{name}_fold{k}.json", s) for k, (_, span) in enumerate(FOLDS, 1) for s in pick(span)]
    return evaluate_pairs(path, pairs, scenario=scen)


if __name__ == "__main__":
    out, names = sys.argv[1], sys.argv[2:]
    res = pd.read_pickle(out) if os.path.exists(out) else {}
    for n in names:
        for sc in SCEN:
            summ, fills = run(n, sc)
            mk = fills[fills.liq.str.contains("MAKER")]
            res[(n, sc)] = summ.set_index("sess")[["mk120", "shares", "ms_mean", "ms_max"]].assign(
                maker_c=mk.groupby("sess").mk120.sum() / mk.groupby("sess").q.apply(lambda q: q.abs().sum()))
            print(f"{n:16s} {sc:30s} mean ${summ.mk120.mean():7.0f} worst ${summ.mk120.min():6.0f} "
                  f"maker {mk.mk120.sum() / mk.q.abs().sum():.2f}c ms_max {summ.ms_max.max():.1f}", flush=True)
            pd.to_pickle(res, out)
