"""Trading backtest of overfit37.py's model variants: 3-fold out-of-sample, all 24 sessions, paired
per session against the live model (fold{k}.json = full37).
    ../.venv/bin/python overfit37_trade.py
"""
import math
import sys

from evalx import evaluate

ST = "share=0.3+through=2+adverse"
VARIANTS = ["", "_full37", "_top10", "_top20", "_ridge0.1", "_ridge1", "_own10", "none"]
SCEN = ["base", ST, "all combined (share=0.1)"]

if __name__ == "__main__":
    only = sys.argv[1:]
    res = {}
    for v in VARIANTS:
        if only and v not in only and v != "":
            continue
        for sc in SCEN:
            summ, _ = evaluate("../strategies/alpha.py", coefs=v, scenario=sc, over={"kill_floor": 0})
            res[(v, sc)] = dict(zip(summ.sess, summ.mk120))
            print(f"{v or 'live (fold{k}.json)':22} {sc:28} {summ.mk120.mean():7.0f} worst {summ.mk120.min():6.0f}", flush=True)
    print("\npaired vs the live model, $/session (mean ± se, sessions better):")
    for v in dict.fromkeys(k[0] for k in res):
        row = f"{v or 'live':12}"
        for sc in SCEN:
            a, b = res[("", sc)], res[(v, sc)]
            d = [b[n] - a[n] for n in a]
            m = sum(d) / len(d)
            se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) / math.sqrt(len(d))
            row += f"   {sc[:12]:12} {sum(b.values()) / len(b):6.0f} ({m:+5.0f}±{se:3.0f}, {sum(x > 0 for x in d):2}/24)"
        print(row)
