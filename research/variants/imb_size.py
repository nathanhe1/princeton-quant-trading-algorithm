"""imb + flow-by-trade-size + clip3 on top of v10: fold fits, R², variant file, and paired markouts
against strategies/alpha.py (v10, fold{k}.json).   cd research && ../.venv/bin/python variants/imb_size.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import feat_r2 as F  # noqa: E402
import make_variants as MV  # noqa: E402


def c_imb_size(R):
    G = F.c_flow_size(R)
    for s in F.SYMS:
        bq, aq = R[s]["bq"], R[s]["aq"]
        G[f"{s}_imb"] = (bq - aq) / (bq + aq)
    return G


F.CANDIDATES["imb_size"] = c_imb_size

if __name__ == "__main__":
    import pandas as pd
    from cv import FOLDS, pick
    from evalx import evaluate_pairs
    Rall = F.load_all()
    for cand in ("imb+clip3", "imb_size+clip3"):
        r, _ = F.cv(cand, Rall)
        print(cand, "OOS R2", {s: round(v, 4) for s, v in r.items()}, flush=True)
    _, _, coefs = F.cv("imb_size+clip3", Rall, return_coefs=True)
    for k, per in coefs.items():
        clip = next(iter(per.values()))[1]
        out = {"H": F.H, "lags": F.LAGS, "flows": F.FLOWS, "variant": "imbsize_clip3", "big": F.BIG,
               "coef": {s: {c: round(float(v), 6) for c, v in b.items()} for s, (b, _) in per.items()},
               "clip": {c: round(float(v), 4) for c, v in clip.items()}}
        json.dump(out, open(os.path.join(HERE, f"imbsize_clip3_fold{k}.json"), "w"), indent=1)
    src = open(os.path.join(HERE, "..", "..", "strategies", "alpha.py")).read()
    open(os.path.join(HERE, "alpha_imbsize_clip3.py"), "w").write(MV.patch(src, {"size"}, "imbsize_clip3"))
    ST, NA, S1 = "share=0.3+through=2+adverse", "share=0.3+through=2", "share=0.1"
    rows = []
    for scen, mult in (("base", 1.0), (ST, 1.0), (NA, 1.0), (S1, 1.0), (ST, -1.0)):
        res = {}
        for lab, path, pat in (("v10", "../strategies/alpha.py", "fold{k}.json"),
                               ("imb+size", "variants/alpha_imbsize_clip3.py", "variants/imbsize_clip3_fold{k}.json")):
            pairs = [(pat.format(k=k), n) for k, (_, span) in enumerate(FOLDS, 1) for n in pick(span)]
            s, _ = evaluate_pairs(path, pairs, scenario=scen, attrs={"COEF_MULT": mult})
            res[lab] = s.set_index("sess").mk120
        d = res["imb+size"] - res["v10"]
        rows.append(dict(scen=scen[:22], coef=mult, v10=res["v10"].mean(), new=res["imb+size"].mean(),
                         diff=d.mean(), se=d.std() / len(d) ** 0.5, worst_v10=res["v10"].min(), worst_new=res["imb+size"].min()))
        print(rows[-1], flush=True)
    print(pd.DataFrame(rows).round(0).to_string(index=False))
