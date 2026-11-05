from cv import cv, summarize
import pandas as pd
BIG = {"CAP": {"HPPO": 800, "BNKY": 2400, "CRRT": 2400, "BRRY": 580}}
configs = [
    ("make1.5", {"take_on": 0, "make_edge": 1.5}, None),
    ("make2", {"take_on": 0, "make_edge": 2}, None),
    ("make2.5", {"take_on": 0, "make_edge": 2.5}, None),
    ("make2 sz1.5", {"take_on": 0, "make_edge": 2, "size_scale": 1.5}, None),
    ("make2 bigcap", {"take_on": 0, "make_edge": 2}, BIG),
    ("make2+take6", {"make_edge": 2, "take_edge": 6}, None),
    ("make2+take10", {"make_edge": 2, "take_edge": 10}, None),
]
if __name__ == "__main__":
    rows = []
    for name, over, attrs in configs:
        s, f = cv("../strategies/alpha.py", over, attrs)
        d = dict(cfg=name, mk120=s.mk120.mean(), mk240=s.mk240.mean(), pos=(s.mk120 > 0).sum(), worst=s.mk120.min(), sd=s.mk120.std())
        for liq in ("MAKER", "TAKER"):
            x = f[f.liq == liq]
            d[f"{liq}_sh"] = x.q.abs().sum() / 24; d[f"{liq}_c"] = x.mk120.sum() / max(1, x.q.abs().sum())
        for sym in ("HPPO", "BNKY", "CRRT", "BRRY"):
            d[sym] = f[f.sym == sym].mk120.sum() / 2400
        rows.append(d); print(name, flush=True)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
