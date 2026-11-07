"""Write research/variants/alpha_<name>.py (a patched copy of strategies/alpha.py: feature code only)
and research/variants/<name>_fold{1,2,3}.json (fitted by feat_r2.cv on each fold's 16 training sessions).

    cd research && ../.venv/bin/python variants/make_variants.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import feat_r2 as F  # noqa: E402

SRC = os.path.join(HERE, "..", "..", "strategies", "alpha.py")

# name -> (feat_r2 candidate, patches)
VARIANTS = {
    "imb": ("imb", {"imb"}),
    "clip3": ("clip3", {"clip"}),
    "flowsize": ("flow_size", {"size"}),
    "flowsize_clip3": ("flow_size+clip3", {"size", "clip"}),
    "imb_clip3": ("imb+clip3", {"imb", "clip"}),
}


def sub(text, old, new):
    assert text.count(old) == 1, f"patch target not unique/found: {old[:60]!r}"
    return text.replace(old, new)


def patch(src, kinds, name):
    s = src
    s = sub(s, '"""Alpha: a fitted', f'"""[research variant {name}: feature patches {sorted(kinds)}; see research/variants/]\n\nAlpha: a fitted')
    s = sub(s, "class Alpha(Strategy):", f"class Alpha_{name}(Strategy):")
    if "size" in kinds:
        s = sub(s, "        self.f_hist = {s: deque(maxlen=max(self.flows)) for s in SYMS}\n",
                "        self.f_hist = {s: deque(maxlen=max(self.flows)) for s in SYMS}\n"
                "        self.big = m.get(\"big\", {})  # print size at or above which flow counts as big\n"
                "        self.fb_hist = {s: deque(maxlen=max(self.flows)) for s in SYMS}\n"
                "        self.fs_hist = {s: deque(maxlen=max(self.flows)) for s in SYMS}\n"
                "        self.own_taker_big: dict[str, int] = {s: 0 for s in SYMS}\n")
        s = sub(s, "            self.own_taker[fill.symbol] += fill.qty if fill.side == \"BUY\" else -fill.qty\n",
                "            self.own_taker[fill.symbol] += fill.qty if fill.side == \"BUY\" else -fill.qty\n"
                "            if fill.qty >= self.big.get(fill.symbol, 10**9):\n"
                "                self.own_taker_big[fill.symbol] += fill.qty if fill.side == \"BUY\" else -fill.qty\n")
        s = sub(s, """        flow = {s: 0 for s in SYMS}
        for tr in state.trades:
            if tr.symbol in flow:
                flow[tr.symbol] += tr.qty if tr.aggressor == "BUY" else -tr.qty
        if self.mode != "backtest":  # the live tape includes our own trades: leave them out
            for s in SYMS:
                flow[s] -= self.own_taker[s]
        self.own_taker = {s: 0 for s in SYMS}
""", """        flow = {s: 0 for s in SYMS}
        fbig = {s: 0 for s in SYMS}
        for tr in state.trades:
            if tr.symbol in flow:
                q = tr.qty if tr.aggressor == "BUY" else -tr.qty
                flow[tr.symbol] += q
                if tr.qty >= self.big.get(tr.symbol, 10**9):
                    fbig[tr.symbol] += q
        if self.mode != "backtest":  # the live tape includes our own trades: leave them out
            for s in SYMS:
                flow[s] -= self.own_taker[s]
                fbig[s] -= self.own_taker_big[s]
        self.own_taker = {s: 0 for s in SYMS}
        self.own_taker_big = {s: 0 for s in SYMS}
""")
        s = sub(s, "            self.f_hist[s].append(flow[s])\n",
                "            self.f_hist[s].append(flow[s])\n"
                "            self.fb_hist[s].append(fbig[s])\n"
                "            self.fs_hist[s].append(flow[s] - fbig[s])\n")
        s = sub(s, "            self.f_hist[s] = deque(f[: len(f) - m] + [0] * m, maxlen=self.f_hist[s].maxlen)\n",
                "            self.f_hist[s] = deque(f[: len(f) - m] + [0] * m, maxlen=self.f_hist[s].maxlen)\n"
                "            for hist in (self.fb_hist, self.fs_hist):\n"
                "                g = list(hist[s])\n"
                "                hist[s] = deque(g[: len(g) - m] + [0] * m, maxlen=hist[s].maxlen)\n")
    feat = ""
    if "size" in kinds:
        feat += ("            fb, fs = list(self.fb_hist[s]), list(self.fs_hist[s])\n"
                 "            for L in self.flows:\n"
                 "                x[f\"{s}_fb{L}\"] = sum(fb[-L:]) / 1000.0\n"
                 "                x[f\"{s}_fs{L}\"] = sum(fs[-L:]) / 1000.0\n")
    if "imb" in kinds:
        feat += ("            e = self.ext[s]\n"
                 "            x[f\"{s}_imb\"] = (e.bq - e.aq) / (e.bq + e.aq)\n")
    if feat:
        s = sub(s, "            x[f\"{s}_um\"] = u - self.ext[s].mid\n", "            x[f\"{s}_um\"] = u - self.ext[s].mid\n" + feat)
    if "clip" in kinds:
        s = sub(s, "        x[\"gapB\"] = (self.ext[\"HPPO\"].micro - self.ext[\"BNKY\"].micro) / 2 - self.ext[\"BRRY\"].micro\n",
                "        x[\"gapB\"] = (self.ext[\"HPPO\"].micro - self.ext[\"BNKY\"].micro) / 2 - self.ext[\"BRRY\"].micro\n"
                "        for k, lim in (self.model.get(\"clip\") or {}).items():  # winsorise at training-set levels\n"
                "            if k in x:\n"
                "                x[k] = max(-lim, min(lim, x[k]))\n")
    s = sub(s, "        for s in SYMS:\n            c = self.coef[s]\n",
            "        self.last_x = x  # research: feature parity check\n        for s in SYMS:\n            c = self.coef[s]\n")
    return s


def main():
    src = open(SRC).read()
    Rall = F.load_all()
    for name, (cand, kinds) in VARIANTS.items():
        open(os.path.join(HERE, f"alpha_{name}.py"), "w").write(patch(src, kinds, name))
        _, _, coefs = F.cv(cand, Rall, return_coefs=True)
        for k, per in coefs.items():
            clip = next(iter(per.values()))[1]
            out = {"H": F.H, "lags": F.LAGS, "flows": F.FLOWS, "variant": name, "candidate": cand,
                   "test_fold": k, "coef": {s: {c: round(float(v), 6) for c, v in b.items()} for s, (b, _) in per.items()}}
            if clip:
                out["clip"] = {c: round(float(v), 4) for c, v in clip.items()}
            if "size" in kinds:
                out["big"] = F.BIG
            json.dump(out, open(os.path.join(HERE, f"{name}_fold{k}.json"), "w"), indent=1)
        print("wrote", name)
    # baseline refit with the same code, for a parity check against fold{k}.json
    _, _, coefs = F.cv("baseline", Rall, return_coefs=True)
    for k, per in coefs.items():
        out = {"H": F.H, "lags": F.LAGS, "flows": F.FLOWS, "variant": "baseline_refit",
               "coef": {s: {c: round(float(v), 6) for c, v in b.items()} for s, (b, _) in per.items()}}
        json.dump(out, open(os.path.join(HERE, f"baseline_refit_fold{k}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
