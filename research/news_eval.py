"""Value and risk of news trading (alpha.py news_trade), on evalx's synthetic news: 6 items per
session, jumps of 2 or 4 x the symbol's typical 60 s move, repricing over D batches.
  off      news_trade 0 (today: pause and rebase only)
  on       news_trade 1, headlines read correctly
  misread  news_trade 1, every headline says the opposite of the move
3-fold out-of-sample, base and realistic stress.   ../.venv/bin/python news_eval.py
"""
import pandas as pd

from evalx import evaluate, line

ST = "share=0.3+through=2+adverse"
if __name__ == "__main__":
    rows = []
    for D in (0, 20, 60):
        for scen in ("base", ST):
            for label, on, flip in (("off", 0, False), ("on", 1, False), ("misread", 1, True)):
                s, f = evaluate("../strategies/alpha.py", coefs="raw", scenario=scen,
                                news={"D": D, "flip": flip}, over={"news_trade": on})
                tk = f[f.liq.str.contains("TAKER")]
                rows.append(dict(D=D, scen="stress" if scen == ST else scen, cfg=label, **line(label, s, f),
                                 taker_sh=tk.q.abs().sum() / len(s), taker_c=tk.mk120.sum() / max(1, tk.q.abs().sum())))
                r = rows[-1]
                print(f"D={D} {r['scen']} {label}: mean {r['mean']:.0f} worst {r['worst']:.0f} "
                      f"near-news {r['near_news']:.0f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("charts/data/news_eval.csv", index=False)
    print(df[["D", "scen", "cfg", "mean", "worst", "near_news", "taker_sh", "taker_c"]].round(1).to_string(index=False))
