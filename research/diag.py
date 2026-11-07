"""Diagnostics for the strategy redesign: per-session volatility vs result, score by symbol and
fill type, inventory at the cap. ../.venv/bin/python diag.py"""
import numpy as np
import pandas as pd

from evalx import evaluate
from load import SYMS, sessions, tops

ST = "share=0.3+through=2+adverse"
CAP = {"HPPO": 400, "BNKY": 1500, "CRRT": 1500, "BRRY": 400}

if __name__ == "__main__":
    vol = {}
    for n in sessions():
        t = tops(n)
        vol[n] = {s: float(t[s]["mid"].ffill().diff(120).std()) for s in SYMS}
        vol[n].update({f"spr_{s}": float((t[s]["ask"] - t[s]["bid"]).median()) if "ask" in t[s] else np.nan for s in SYMS})
    V = pd.DataFrame(vol).T
    print("60 s move sd by session (cents): min / median / max")
    for s in SYMS:
        print(f"  {s}: {V[s].min():.1f} / {V[s].median():.1f} / {V[s].max():.1f}  (max/min {V[s].max() / V[s].min():.1f}x)")
    for scen in ("base", ST):
        s, f = evaluate("../strategies/alpha.py", coefs="raw", scenario=scen)
        f["liq"] = f.liq.str.replace("Liquidity.", "")
        g = f.groupby(["sym", "liq"]).agg(sh=("q", lambda x: x.abs().sum()), mk=("mk120", "sum"))
        g["c_sh"] = g.mk / g.sh
        g["usd_sess"] = g.mk / 100 / 24
        print(f"\n{scen}: by symbol and fill type ($/session, c/share)")
        print(g[["usd_sess", "c_sh"]].round(2).to_string())
        # markout vs session volatility
        s = s.set_index("sess")
        z = (V[list(SYMS)] / V[list(SYMS)].median()).mean(1)
        print(f"corr(session result, relative volatility) = {np.corrcoef(s.mk120.loc[z.index], z)[0, 1]:+.2f}")
        q = pd.qcut(z, 3, labels=["low vol", "mid vol", "high vol"])
        print(s.mk120.groupby(q.loc[s.index]).mean().round(0).to_dict())
        # inventory at cap (reconstructed from fills)
        at = {}
        for sess, ff in f.groupby("sess"):
            for sym in SYMS:
                x = ff[ff.sym == sym].sort_values("batch")
                pos = x.q.cumsum().to_numpy()
                b = x.batch.to_numpy()
                if len(b) < 2:
                    continue
                dur = np.diff(np.append(b, 5400))
                at.setdefault(sym, []).append(float((dur * (np.abs(pos) >= 0.9 * CAP[sym])).sum() / 5400))
        print("share of session with |pos| >= 90% soft cap:", {k: f"{np.mean(v):.0%}" for k, v in at.items()})
