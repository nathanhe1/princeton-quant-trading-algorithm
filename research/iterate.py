"""Before/after table for this iteration (no-news fill scenarios, then synthetic news).
Run from research/: ../.venv/bin/python iterate.py [nonews] [news]"""
import sys
import pandas as pd
from evalx import evaluate, line

V2 = "../strategies/archive/alpha_v2.py"
NEW = "../strategies/alpha.py"
REAL = "share=0.3+through=2+adverse"


def table(rows, title, out):
    df = pd.DataFrame(rows)
    print(f"\n{title}\n" + df.round(2).to_string(index=False), flush=True)
    df.to_csv(out, index=False)


def nonews():
    configs = [
        ("v2 (raw coefs)", V2, "raw", None),
        ("v2 code + scaled-ridge coefs", V2, "std", None),
        ("new: scaled ridge + fair cap", NEW, "std", None),
        ("new without fair cap", NEW, "std", {"FAIR_CAP": None}),
    ]
    rows = []
    for scen in ("base", "share=0.3+through=2", REAL):
        for label, path, coefs, attrs in configs:
            s, f = evaluate(path, coefs=coefs, scenario=scen, attrs=attrs)
            rows.append({"scenario": scen, **line(label, s, f)})
            print(rows[-1], flush=True)
    table(rows, "No news", "charts/data/iter_nonews.csv")


def news():
    configs = [
        ("v2 (raw coefs)", V2, "raw", None, None),
        ("new", NEW, "std", None, None),
        ("new, no rebase", NEW, "std", {"NEWS_REBASE": False}, None),
        ("new, no fair cap", NEW, "std", {"FAIR_CAP": None}, None),
        ("new, news_pause 30", NEW, "std", None, {"news_pause": 30}),
    ]
    rows = []
    for D in (0, 20, 60):
        for label, path, coefs, attrs, over in configs:
            s, f = evaluate(path, coefs=coefs, news={"D": D}, attrs=attrs, over=over)
            rows.append({"reprice_D": D, **line(label, s, f)})
            print(rows[-1], flush=True)
    table(rows, "Synthetic news (6 items/session, jumps 2x/4x the 60 s move), base fill rule", "charts/data/iter_news.csv")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["nonews", "news"]
    for p in parts:
        globals()[p]()
