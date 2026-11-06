"""Final before/after for this iteration: alpha_v2 vs alpha.py (same coefficients; fair cap and
news rebase added). Run from research/: ../.venv/bin/python final_compare.py"""
import pandas as pd
from evalx import evaluate, line

V2, NEW = "../strategies/archive/alpha_v2.py", "../strategies/alpha.py"
CASES = [("base", None), ("share=0.3+through=2", None), ("share=0.3+through=2+adverse", None),
         ("base", {"D": 0}), ("base", {"D": 20}), ("base", {"D": 60}),
         ("share=0.3+through=2+adverse", {"D": 20})]


def main():
    rows = []
    for scen, news in CASES:
        for label, path in (("before (v2)", V2), ("after", NEW)):
            s, f = evaluate(path, coefs="raw", scenario=scen, news=news)
            rows.append({"scenario": scen, "news": "none" if news is None else f"reprice {news['D']} batches",
                         **line(label, s, f)})
            print(rows[-1]["scenario"], rows[-1]["news"], label, round(rows[-1]["mean"]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("charts/data/final_compare.csv", index=False)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
