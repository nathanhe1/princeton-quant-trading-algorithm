"""Backtest cost and benefit of strategy switches (default: the out-of-range guards (VOL_GUARD, THROTTLE_FROM) on the historical
sessions: base, realistic stress, stress without adverse, and a wrong-signed model under stress.
Paired per session against the guards off.   ../.venv/bin/python guards_eval.py ['<configs as JSON>']
"""
import pandas as pd

from evalx import evaluate

ST, NA = "share=0.3+through=2+adverse", "share=0.3+through=2"
CONFIGS = {"off": {}, "vol 1.5": {"VOL_GUARD": 1.5}, "vol 2.0": {"VOL_GUARD": 2.0},
           "throttle 6": {"THROTTLE_FROM": 6}, "throttle 12": {"THROTTLE_FROM": 12},
           "vol 1.5 + thr 6": {"VOL_GUARD": 1.5, "THROTTLE_FROM": 6}}
CASES = [("base", 1.0), (ST, 1.0), (NA, 1.0), (ST, -1.0)]

if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) > 1:  # configs as JSON: '{"off": {}, "join": {"MAX_IMPROVE": 0}}'
        CONFIGS = json.loads(sys.argv[1])
    res = {}
    for name, attrs in CONFIGS.items():
        for scen, mult in CASES:
            s, _ = evaluate("../strategies/alpha.py", coefs="raw", scenario=scen, attrs={**attrs, "COEF_MULT": mult})
            res[(name, scen, mult)] = s.set_index("sess").mk120
            print(f"{name:16} {scen[:22]:22} x{mult}: {s.mk120.mean():6.0f} worst {s.mk120.min():6.0f}", flush=True)
    rows = []
    for name in CONFIGS:
        row = {"cfg": name}
        for scen, mult in CASES:
            lab = {"base": "base", ST: "stress", NA: "noadv"}[scen] + ("" if mult == 1 else f" x{mult:g}")
            x, o = res[(name, scen, mult)], res[(next(iter(CONFIGS)), scen, mult)]
            row[lab] = f"{x.mean():.0f} ({(x - o).mean():+.0f}±{(x - o).std() / len(x) ** .5:.0f})"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
