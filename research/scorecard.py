"""Standard scorecard for any strategy: run two strategies through this and compare the tables.

usage (from research/):
  ../.venv/bin/python scorecard.py ../strategies/alpha.py --label alpha [--coefs raw|none]

--coefs raw  : the strategy reads `self.model` (like alpha.py); each session gets the fold
               coefficients that never saw it (3-fold out of sample). Fair.
--coefs none : the strategy keeps its own built-in model. If that model was fitted on these 24
               sessions, its numbers are IN-SAMPLE and will look better than they are: say so.

Writes scorecards/<label>.md (tables) and scorecards/<label>_sessions.csv (every session x
scenario, for paired comparison). Same sessions, seeds, stress rules and synthetic news for every
strategy, so two scorecards are directly comparable. About 2-3 minutes on 10 cores.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from evalx import evaluate
from load import trades

SYMS = ["HPPO", "BNKY", "CRRT", "BRRY"]
REAL = "share=0.3+through=2+adverse"
CASES = [
    ("base (backtest fill rule)", "base", None),
    ("2 ticks through", "through=2", None),
    ("30% of trade size", "share=0.3", None),
    ("10% of trade size", "share=0.1", None),
    ("30% + 2 ticks", "share=0.3+through=2", None),
    ("realistic: 30% + 2 ticks + adverse", REAL, None),
    ("harshest: 10% + 2 ticks + adverse", "all combined (share=0.1)", None),
    ("news, instant repricing", "base", {"D": 0}),
    ("news, repricing over 10 s", "base", {"D": 20}),
    ("news, repricing over 30 s", "base", {"D": 60}),
    ("realistic + news (10 s)", REAL, {"D": 20}),
]


def liq_stats(f, liq):
    x = f[f.liq.str.contains(liq)]
    sh = x.q.abs().sum()
    return sh, (x.mk120.sum() / sh if sh else float("nan"))


def at_price_split(f):
    """Maker fills printed exactly at our price vs others (for the adverse look-ahead check)."""
    mk = f[f.liq.str.contains("MAKER")]
    at = set()
    for n in mk.sess.unique():
        t = trades(n)
        at |= set(zip([n] * len(t), t.batch, t.symbol, t.price))
    flag = np.array([(n, b, s, p) in at for n, b, s, p in zip(mk.sess, mk.batch, mk.sym, mk.px)])
    return mk[flag].mk120.sum() / 100 / 24, mk[~flag].mk120.sum() / 100 / 24


def money(v):
    return f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy")
    ap.add_argument("--label", required=True)
    ap.add_argument("--coefs", default="raw", choices=["raw", "std", "none"])
    a = ap.parse_args()
    os.makedirs("scorecards", exist_ok=True)

    rows, sess_rows, detail, ops = [], [], {}, None
    splits = {}
    for label, scen, news in CASES:
        s, f = evaluate(a.strategy, coefs=a.coefs, scenario=scen, news=news)
        n = len(s)
        msh, mc = liq_stats(f, "MAKER")
        tsh, tc = liq_stats(f, "TAKER")
        rows.append({
            "case": label, "mean": s.mk120.mean(), "sd": s.mk120.std(), "worst": s.mk120.min(),
            "best": s.mk120.max(), "positive": f"{(s.mk120 > 0).sum()}/{n}",
            "mean/sd": s.mk120.mean() / s.mk120.std(), "mk240_mean": s.mk240.mean(),
            "shares": s.shares.mean(), "maker_sh": msh / n, "maker_c": mc, "taker_sh": tsh / n, "taker_c": tc,
        })
        for _, r in s.iterrows():
            sess_rows.append({"case": label, "sess": r.sess, "mk120": r.mk120, "mk240": r.mk240, "shares": r.shares})
        if label.startswith("base") or label.startswith("realistic:"):
            g = f.assign(sh=f.q.abs(), liq=f.liq.str.replace("Liquidity.", "", regex=False)).groupby(["sym", "liq"]).agg(
                shares=("sh", "sum"), mk=("mk120", "sum"))
            g["shares/session"] = g.shares / n
            g["c/share"] = g.mk / g.shares
            g["$/session"] = g.mk / 100 / n
            detail[label] = g[["shares/session", "c/share", "$/session"]]
        if label.startswith("base"):
            rej = {}
            for r in s.rejects:
                try:
                    for k, v in json.loads(r).items():
                        rej[k] = rej.get(k, 0) + v
                except (ValueError, TypeError, AttributeError):
                    pass
            ops = {
                "messages per batch (mean)": f"{s.msgs_mean.mean():.2f}",
                "messages per batch (99th pct, worst session)": f"{s.msgs_p99.max():.0f}",
                "messages per batch (max)": f"{s.msgs_max.max():.0f} (limit 20)",
                "on_batch time (mean)": f"{s.ms_mean.mean():.2f} ms",
                "on_batch time (max)": f"{s.ms_max.max():.1f} ms (batch is 500 ms)",
                "rejects per session": f"{sum(rej.values()) / n:.1f} {json.dumps({k: round(v / n, 1) for k, v in rej.items()})}",
                "max |net exposure| (median session)": f"{s.max_abs_exposure.median():.0f} (limit 600)",
                **{f"max |position| {sym} (median session)": f"{s[f'maxpos_{sym}'].median():.0f}" for sym in SYMS},
            }
        if scen in ("share=0.3+through=2", REAL) and news is None:
            splits[scen] = at_price_split(f)
        print("done", label, round(s.mk120.mean()), flush=True)

    t = pd.DataFrame(rows)
    pd.DataFrame(sess_rows).to_csv(f"scorecards/{a.label}_sessions.csv", index=False)
    t.to_csv(f"scorecards/{a.label}_summary.csv", index=False)

    real = t.set_index("case").loc["realistic: 30% + 2 ticks + adverse", "mean"]
    leak = splits[REAL][1] - splits["share=0.3+through=2"][1]
    md = [f"# Scorecard: {a.label}", "",
          f"Strategy `{a.strategy}`, coefficients `{a.coefs}`"
          + (" (out of sample: fold coefficients)" if a.coefs != "none" else " (strategy's own model: IN-SAMPLE if fitted on these sessions)")
          + ". 24 sessions; 120-batch proxy markout in $ per session unless stated. Generated by `research/scorecard.py`.", "",
          "## Headline", "",
          "| metric | value |", "|---|---|",
          f"| base mean / worst session | {money(t.loc[0, 'mean'])} / {money(t.loc[0, 'worst'])} |",
          f"| realistic stress mean / worst | {money(real)} / {money(t.loc[5, 'worst'])} |",
          f"| realistic, adverse look-ahead removed (conservative) | {money(real - leak)} |",
          f"| harshest stress mean / worst | {money(t.loc[6, 'mean'])} / {money(t.loc[6, 'worst'])} |",
          f"| base maker edge | {t.loc[0, 'maker_c']:.2f} c/share on {t.loc[0, 'maker_sh']:,.0f} shares/session |",
          f"| realistic maker edge | {t.loc[5, 'maker_c']:.2f} c/share on {t.loc[5, 'maker_sh']:,.0f} shares/session |",
          f"| news (instant / 10 s / 30 s repricing) | {money(t.loc[7, 'mean'])} / {money(t.loc[8, 'mean'])} / {money(t.loc[9, 'mean'])} |",
          "", "## All cases", "",
          "| case | mean | sd | worst | best | positive | mean/sd | 240-batch mean | shares | maker c/sh | taker sh | taker c/sh |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        md.append(f"| {r.case} | {money(r['mean'])} | {money(r.sd)} | {money(r.worst)} | {money(r.best)} | {r.positive} | "
                  f"{r['mean/sd']:.2f} | {money(r.mk240_mean)} | {r.shares:,.0f} | {r.maker_c:.2f} | {r.taker_sh:,.0f} | {r.taker_c:.2f} |")
    md += ["", "Adverse look-ahead split (maker fills printed exactly at our price vs all others, $/session):", "",
           "| scenario | at-price fills | other fills |", "|---|---|---|"]
    for k, (atp, oth) in splits.items():
        md.append(f"| {k} | {money(atp)} | {money(oth)} |")
    for label, g in detail.items():
        md += ["", f"## By symbol and liquidity: {label}", "", "| symbol | liquidity | shares/session | c/share | $/session |",
               "|---|---|---|---|---|"]
        for (sym, liq), r in g.iterrows():
            md.append(f"| {sym} | {liq} | {r['shares/session']:,.0f} | {r['c/share']:.2f} | {money(r['$/session'])} |")
    md += ["", "## Operational (base case)", "", "| metric | value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in ops.items()]
    open(f"scorecards/{a.label}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
