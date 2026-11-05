"""Backtest a strategy over sessions and score fills by markout (proxy for the official score).

usage: python bt.py ../strategies/alpha.py [--coef coef.json] [--sessions s17-s24] [--set k=v ...]
Markout per fill = q * (ref - price) - fee, with ref = mid 120 or 240 batches after the fill.
"""
import argparse, importlib.util, json, sys, os
import numpy as np, pandas as pd
from pqt_client.backtest_v2 import load_session
from pqt_client.v2 import backtest_strategy, Strategy
from load import DATA, tops, sessions

def load_cls(path):
    spec = importlib.util.spec_from_file_location("strat", path); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [v for v in vars(mod).values() if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy][0]

def run(cls, names, model=None, over=None, attrs=None, quiet=True):
    rows, fills_all = [], []
    for n in names:
        strat = cls()
        if model is not None: strat.model = model
        if over: strat.params_override = over
        for k, v in (attrs or {}).items(): setattr(strat, k, v)
        fills = []
        orig = strat.on_fill
        def rec(f, orig=orig, fills=fills):
            fills.append((f.batch, f.symbol, f.qty if f.side == "BUY" else -f.qty, f.price, f.fee_mills / 10, str(getattr(f.liquidity, "value", f.liquidity)), f.penalty_mills / 10))
            orig(f)
        strat.on_fill = rec
        import logging; logging.disable(logging.INFO if quiet else logging.NOTSET)
        res = backtest_strategy(load_session(DATA, n), strat)
        t = tops(n)
        df = pd.DataFrame(fills, columns=["batch", "sym", "q", "px", "fee", "liq", "pen"])
        if len(df):
            for H in (120, 240):
                ref = np.array([t[s]["mid"].ffill().iloc[min(b + H, 5399)] for b, s in zip(df.batch, df.sym)])
                df[f"mk{H}"] = df.q * (ref - df.px) - df.fee - df.pen
            df["sess"] = n
            fills_all.append(df)
        rows.append(dict(sess=n, pnl_est=res.pnl_est_mills / 1000, fills=len(df), shares=int(df.q.abs().sum()) if len(df) else 0,
                         mk120=df.mk120.sum() / 100 if len(df) else 0, mk240=df.mk240.sum() / 100 if len(df) else 0,
                         rejects=sum(res.rejects.values()) if hasattr(res, "rejects") and isinstance(res.rejects, dict) else getattr(res, "rejects", None)))
    return pd.DataFrame(rows), (pd.concat(fills_all) if fills_all else pd.DataFrame())

def pick(spec):
    ss = sessions()
    if not spec: return ss
    a, b = spec.split("-") if "-" in spec else (spec, spec)
    return [s for s in ss if a <= s <= b]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("strategy"); ap.add_argument("--coef"); ap.add_argument("--sessions")
    ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    cls = load_cls(a.strategy)
    model = json.load(open(a.coef)) if a.coef else None
    over = {k: float(v) for k, v in (x.split("=") for x in a.set)}
    summ, fills = run(cls, pick(a.sessions), model, over, quiet=not a.v)
    pd.set_option("display.width", 200)
    print(summ.round(2).to_string(index=False))
    print("TOTAL mk120 $%.2f  mk240 $%.2f  pnl_est $%.2f  per-session mk120 mean %.2f sd %.2f" % (
        summ.mk120.sum(), summ.mk240.sum(), summ.pnl_est.sum(), summ.mk120.mean(), summ.mk120.std()))
    if len(fills):
        g = fills.groupby(["sym", "liq"]).agg(n=("q", "size"), shares=("q", lambda x: x.abs().sum()), mk120=("mk120", "sum"), mk240=("mk240", "sum"))
        g["mk120"] /= 100; g["mk240"] /= 100; g["c_per_sh"] = g.mk120 * 100 / g.shares
        print(g.round(2).to_string())
