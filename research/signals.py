import numpy as np, pandas as pd
from load import *
Hs = [2, 10, 30, 60, 120, 240]
out = []
for n in sessions():
    t = tops(n); tr = trades(n)
    M = {s: t[s]["mid"].ffill() for s in SYMS}; U = {s: t[s]["micro"].ffill() for s in SYMS}
    imp = (U["HPPO"] - U["BNKY"]) / 2
    gap = imp - U["BRRY"]
    tr["sq"] = np.where(tr.aggressor == "BUY", tr.qty, -tr.qty)
    flow = tr.pivot_table(index="batch", columns="symbol", values="sq", aggfunc="sum").reindex(M["HPPO"].index).fillna(0)
    for H in Hs:
        r = {}
        y = M["BRRY"].shift(-H) - U["BRRY"]
        r["brry_gap_beta"] = np.polyfit(gap[:-H].values, y[:-H].values, 1)[0]
        for s in SYMS:
            ys = M[s].shift(-H) - U[s]
            for k in (10, 60):
                f = flow[s].rolling(k).sum()
                d = pd.concat([f, ys], axis=1).dropna()
                r[f"{s}_flow{k}_corr"] = d.corr().iloc[0, 1]
            # lead lag: other symbols' past 30-batch return
            for o in SYMS:
                if o == s: continue
                pr = M[o].diff(30)
                d = pd.concat([pr, ys], axis=1).dropna()
                r[f"{s}<-{o}_corr"] = d.corr().iloc[0, 1]
            own = M[s].diff(30); d = pd.concat([own, ys], axis=1).dropna()
            r[f"{s}<-self_corr"] = d.corr().iloc[0, 1]
        out.append(dict(sess=n, H=H, **r))
df = pd.DataFrame(out).drop(columns="sess").groupby("H").mean().T
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(df.round(3).to_string())
