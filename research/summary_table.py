"""Before/after table for the overnight summary: v3.1 (start of the work, 33-feature model), v8 and
v10 (current), 3-fold out-of-sample, every scenario.   ../.venv/bin/python summary_table.py
"""
import pandas as pd

from evalx import evaluate

ST, NA, S1, H = "share=0.3+through=2+adverse", "share=0.3+through=2", "share=0.1", "all combined (share=0.1)"
VERSIONS = [("v3.1 (start)", "../strategies/archive/alpha_v3_1.py", "base33"),
            ("v8", "../strategies/archive/alpha_v8.py", "raw"),
            ("v10 (now)", "../strategies/alpha.py", "raw")]
CASES = [("base", "base", 1.0), ("realistic stress", ST, 1.0), ("stress, no adverse", NA, 1.0), ("10% share", S1, 1.0),
         ("harshest", H, 1.0), ("model wrong-signed", ST, -1.0), ("model overconfident x2", ST, 2.0), ("no model", ST, 0.0)]

if __name__ == "__main__":
    rows = []
    for label, scen, mult in CASES:
        row = {"case": label}
        for v, path, coefs in VERSIONS:
            s, _ = evaluate(path, coefs=coefs, scenario=scen, attrs={"COEF_MULT": mult})
            row[v] = f"{s.mk120.mean():,.0f} ({s.mk120.min():,.0f})"
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("charts/data/summary_table.csv", index=False)
    print(df.to_string(index=False))
