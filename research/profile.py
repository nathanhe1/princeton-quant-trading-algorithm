import numpy as np, pandas as pd
from load import *
rows = []
for n in sessions():
    t = tops(n); tr = trades(n)
    for s in SYMS:
        d = t[s]; m = d["mid"].ffill()
        r = m.diff()
        vol = tr[tr.symbol == s]
        rows.append(dict(sess=n, sym=s, px=m.iloc[0], spr=d.spr.median(), spr_mean=d.spr.mean(),
                         onesided=d.bid.isna().mean() + d.ask.isna().mean(),
                         sd1=r.std(), sd60s=m.diff(120).std(), drift=m.iloc[-1] - m.iloc[0],
                         ntr=len(vol), shares=vol.qty.sum(), selftr=(vol.self_trade).mean()))
    hp = t["HPPO"]["mid"]; bk = t["BNKY"]["mid"] + 2 * t["BRRY"]["mid"]
    rows.append(dict(sess=n, sym="BASIS", px=(hp - bk).mean(), spr=(hp - bk).std(),
                     sd1=(hp - bk).diff().std(), sd60s=(hp-bk).diff(120).std()))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(df.drop(columns="sess").groupby("sym").agg(["mean", "min", "max"]).T.round(2).to_string())
