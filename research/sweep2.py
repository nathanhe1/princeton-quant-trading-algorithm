from cv import cv, summarize
import pandas as pd
BIG = {"CAP": {"HPPO": 800, "BNKY": 2400, "CRRT": 2400, "BRRY": 580}}
configs = [
    ("take3", {"make_on": 0, "take_edge": 3}, None),
    ("take3 bigcap", {"make_on": 0, "take_edge": 3}, BIG),
    ("make2 only", {"take_on": 0, "make_edge": 2}, None),
    ("make4 only", {"take_on": 0, "make_edge": 4}, None),
    ("make6 only", {"take_on": 0, "make_edge": 6}, None),
    ("take3+make4", {"take_edge": 3, "make_edge": 4}, None),
    ("take3+make4 bigcap", {"take_edge": 3, "make_edge": 4}, BIG),
]
if __name__ == "__main__":
    rows = []
    for name, over, attrs in configs:
        s, f = cv("../strategies/alpha.py", over, attrs)
        g = summarize(f)
        rows.append(dict(cfg=name, mk120=s.mk120.mean(), mk240=s.mk240.mean(), pos=(s.mk120 > 0).sum(), worst=s.mk120.min(),
                         make_sh=g.xs("Liquidity.MAKER", level=1).shares.sum() if "Liquidity.MAKER" in g.index.get_level_values(1) else 0,
                         make_c=(g.xs("Liquidity.MAKER", level=1).mk120.sum() * 100 / g.xs("Liquidity.MAKER", level=1).shares.sum()) if "Liquidity.MAKER" in g.index.get_level_values(1) else 0,
                         take_sh=g.xs("Liquidity.TAKER", level=1).shares.sum() if "Liquidity.TAKER" in g.index.get_level_values(1) else 0,
                         rej=s.rejects.mean()))
        print(rows[-1], flush=True)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
