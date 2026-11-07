"""One evaluation harness: 3-fold out-of-sample markouts, optionally under a stressed fill model
(stress.py) and with synthetic news. Parallel over sessions.

    from evalx import evaluate
    summ, fills = evaluate("../strategies/alpha.py", coefs="std", scenario="base", news=None)

coefs: "raw" -> fold{k}.json (old fit), "std" -> fold{k}_std.json (scaled ridge, nested-CV penalty).
attrs: class attributes to set on the strategy; the special key COEF_MULT multiplies every fitted
coefficient (the model-failure scenario: -0.5 wrong sign, 0 no model, 2 overconfident).
news:  None, or dict(D=<repricing batches>, n=<items per session>, mults=(2, 4), seed=...,
       flip=True for headlines that say the opposite of the move, neutral=True for no direction).

Synthetic news (the data has none). Each item moves one stock's true value by J = +/- mult x its
typical 60 s move, at batch t; HPPO moves with its basket (1 BNKY + 2 BRRY). The recorded book and
trades of every affected symbol are shifted by J * min(1, (b - t + 1) / (D + 1)) from batch t on,
so the market reprices over D batches and trades during the ramp can hit stale quotes. An item is
delivered to on_news in batch t. Markouts follow the official rule: the reference is the (shifted)
mid 120 batches after the fill, minus the jump of any item published after the fill and within those
120 batches (the score's Delta-N term). This is a sensitivity test with invented jump sizes and
repricing speeds, not a forecast.
"""
import dataclasses
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from pqt_client import backtest_v2 as bt
from pqt_client.backtest_v2 import load_session
from pqt_client.v2 import backtest_strategy
from pqt_client.v2 import backtest as v2bt
from pqt_protocol import v2 as wire

from bt import load_cls
from cv import FOLDS, pick
from load import DATA, SYMS

HERE = os.path.dirname(os.path.abspath(__file__))
H = 120
WEIGHT = {"HPPO": 1.0, "BNKY": 0.15, "CRRT": -0.29, "BRRY": 0.425}  # exposure weights (guide section 6)
# typical 60 s mid move (cents, std of mid(t+120) - mid(t) over all sessions); see sigma60()
SIGMA60 = {"HPPO": 12.6, "BNKY": 4.9, "CRRT": 4.9, "BRRY": 8.2}
PROPAGATE = {"BNKY": {"BNKY": 1, "HPPO": 1}, "BRRY": {"BRRY": 1, "HPPO": 2}, "CRRT": {"CRRT": 1}}


def sigma60():
    from load import sessions, tops

    out = {s: [] for s in SYMS}
    for n in sessions():
        t = tops(n)
        for s in SYMS:
            out[s].append(t[s]["mid"].ffill().diff(H).dropna())
    return {s: float(pd.concat(v).std()) for s, v in out.items()}


def make_events(name, cfg):
    rng = random.Random(f"{cfg.get('seed', 7)}-{name}")
    events = []
    for i in range(cfg.get("n", 6)):
        t = rng.randint(600, 5000)
        src = rng.choice(["BNKY", "BRRY", "CRRT"])
        j = rng.choice([-1, 1]) * rng.choice(cfg.get("mults", (2, 4))) * SIGMA60[src]
        jumps = {s: int(round(j * w)) for s, w in PROPAGATE[src].items()}
        events.append(dict(id=i + 1, t=t, src=src, jumps=jumps))
    return sorted(events, key=lambda e: e["t"])


def shift_of(events, D, n_batches):
    """shift[s][b] in cents for every symbol and batch."""
    out = {s: np.zeros(n_batches) for s in SYMS}
    b = np.arange(n_batches)
    for e in events:
        ramp = np.clip((b - e["t"] + 1) / (D + 1), 0, 1)
        for s, j in e["jumps"].items():
            out[s] += j * ramp
    return {s: np.round(v).astype(int) for s, v in out.items()}


def shifted_session(session, shift):
    tops = []
    for b, top in enumerate(session.tops):
        new = {}
        for s, t in top.items():
            d = int(shift[s][b]) if s in shift else 0
            new[s] = dataclasses.replace(
                t,
                bid_px=None if t.bid_px is None else t.bid_px + d,
                ask_px=None if t.ask_px is None else t.ask_px + d,
            )
        tops.append(new)
    trades = tuple(
        tuple(dataclasses.replace(tr, price=tr.price + int(shift[tr.symbol][tr.batch])) for tr in batch)
        for batch in session.trades
    )
    return dataclasses.replace(session, tops=tuple(tops), trades=trades)


def _mids(session):
    out = {}
    for s in SYMS:
        m = [
            (t[s].bid_px + t[s].ask_px) / 2 if s in t and t[s].bid_px is not None and t[s].ask_px is not None else np.nan
            for t in session.tops
        ]
        out[s] = pd.Series(m).ffill().to_numpy()
    return out


def _one(job):
    path, coef_file, name, scenario_name, over, attrs, news = job
    import stress

    session = load_session(DATA, name)
    events = make_events(name, news) if news else []
    if events:
        session = shifted_session(session, shift_of(events, news["D"], session.n_batches))
    by_batch = {}
    for e in events:
        up = sum(e["jumps"].values()) > 0
        if news.get("flip"):  # a misread: the headline says the opposite of what happens
            up = not up
        head = f"synthetic {e['src']}" if news.get("neutral") else (
            f"{e['src']} surges on record results" if up else f"{e['src']} plunges after recall")
        by_batch.setdefault(e["t"], []).append(wire.NewsItem(e["id"], e["t"], head, ""))

    original_send = v2bt._Feed.send
    msgs = []  # client messages (orders, cancels, replaces, conversions) answered to each batch

    def send(self, message):
        if by_batch and isinstance(message, wire.Batch) and message.batch in by_batch:
            message = dataclasses.replace(message, news=by_batch[message.batch])
        out = original_send(self, message)
        if isinstance(message, wire.Batch):
            msgs.append(len(out))
        return out

    scenario = next(s for s in stress.SCENARIOS if s.name == scenario_name)
    original_fills = stress._install(scenario, int(os.environ.get("STRESS_SEED", 2026))) if scenario_name != "base" else None
    v2bt._Feed.send = send
    try:
        strat = load_cls(path)()
        if coef_file:  # None = use the strategy's own built-in model (may be in-sample!)
            with open(os.path.join(HERE, coef_file)) as f:
                strat.model = json.load(f)
        attrs = dict(attrs or {})
        coef_mult = attrs.pop("COEF_MULT", None)  # model-failure scenario (failure.py)
        if coef_mult is not None:
            m = strat.model
            strat.model = {**m, "coef": {s: {k: v * coef_mult for k, v in c.items()} for s, c in m["coef"].items()}}
        if over:
            strat.params_override = over
        for k, v in (attrs or {}).items():
            setattr(strat, k, v)
        fills = []
        orig = strat.on_fill

        def rec(f, orig=orig):
            fills.append((f.batch, f.symbol, f.qty if f.side == "BUY" else -f.qty, f.price, f.fee_mills / 10,
                          str(getattr(f.liquidity, "value", f.liquidity)), f.penalty_mills / 10))
            orig(f)

        strat.on_fill = rec
        times = []
        orig_batch = strat.on_batch

        def timed(state, orig_batch=orig_batch):
            t0 = time.perf_counter()
            orig_batch(state)
            times.append(time.perf_counter() - t0)

        strat.on_batch = timed
        import logging

        logging.disable(logging.INFO)
        result = backtest_strategy(session, strat)
    finally:
        v2bt._Feed.send = original_send
        if original_fills is not None:
            stress._Replay.rest_fills = original_fills

    df = pd.DataFrame(fills, columns=["batch", "sym", "q", "px", "fee", "liq", "pen"])
    rej = getattr(result, "rejects", None)
    summary = dict(sess=name, fills=len(df), shares=0, mk120=0.0, mk240=0.0, news_fills_mk=0.0,
                   msgs_mean=float(np.mean(msgs)) if msgs else 0.0, msgs_max=max(msgs) if msgs else 0,
                   msgs_p99=float(np.percentile(msgs, 99)) if msgs else 0.0,
                   ms_mean=1000 * float(np.mean(times)) if times else 0.0,
                   ms_max=1000 * max(times) if times else 0.0,
                   rejects=json.dumps({str(k): v for k, v in rej.items()}) if isinstance(rej, dict) else str(rej),
                   max_abs_exposure=0.0, **{f"maxpos_{s}": 0 for s in SYMS})
    if len(df):
        mids = _mids(session)
        last = session.n_batches - 1
        ref = np.array([mids[s][min(b + H, last)] for b, s in zip(df.batch, df.sym)])
        dn = np.zeros(len(df))
        near = np.zeros(len(df), dtype=bool)
        for e in events:
            for s, j in e["jumps"].items():
                m = (df.sym == s).to_numpy()
                after = m & (df.batch.to_numpy() < e["t"]) & (e["t"] <= df.batch.to_numpy() + H)
                dn[after] += j
                near |= m & (df.batch.to_numpy() >= e["t"]) & (df.batch.to_numpy() < e["t"] + 240)
        df["mk120"] = df.q * (ref - dn - df.px) - df.fee - df.pen
        ref240 = np.array([mids[s][min(b + 240, last)] for b, s in zip(df.batch, df.sym)])
        dn240 = np.zeros(len(df))
        for e in events:
            for s, j in e["jumps"].items():
                m = (df.sym == s).to_numpy()
                dn240[m & (df.batch.to_numpy() < e["t"]) & (e["t"] <= df.batch.to_numpy() + 240)] += j
        df["mk240"] = df.q * (ref240 - dn240 - df.px) - df.fee - df.pen
        pos = {s: 0 for s in SYMS}
        expo = 0.0
        for sym, q in zip(df.sym, df.q):
            pos[sym] += q
            summary[f"maxpos_{sym}"] = max(summary[f"maxpos_{sym}"], abs(pos[sym]))
            expo = max(expo, abs(sum(WEIGHT[s] * pos[s] for s in SYMS)))
        summary["max_abs_exposure"] = expo
        df["near_news"] = near
        df["sess"] = name
        summary.update(shares=int(df.q.abs().sum()), mk120=df.mk120.sum() / 100, mk240=df.mk240.sum() / 100,
                       news_fills_mk=df.loc[df.near_news, "mk120"].sum() / 100)
    return summary, df


def evaluate(path, coefs="std", scenario="base", news=None, over=None, attrs=None, sessions_spec=None):
    suffix = "_std" if coefs == "std" else ""
    jobs = []
    for k, (coef_file, span) in enumerate(FOLDS, start=1):
        for name in pick(span):
            if sessions_spec and name not in pick(sessions_spec):
                continue
            cf = None if coefs == "none" else f"fold{k}{suffix}.json"
            jobs.append((os.path.abspath(path), cf, name, scenario, over, attrs, news))
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        out = list(pool.map(_one, jobs))
    summ = pd.DataFrame([o[0] for o in out])
    fills = pd.concat([o[1] for o in out if len(o[1])])
    return summ, fills


def line(label, summ, fills):
    mk = fills[fills.liq.str.contains("MAKER")]
    return dict(
        config=label,
        mean=summ.mk120.mean(),
        worst=summ.mk120.min(),
        sd=summ.mk120.std(),
        positive=int((summ.mk120 > 0).sum()),
        shares=summ.shares.mean(),
        maker_c=mk.mk120.sum() / max(1, mk.q.abs().sum()),
        near_news=summ.news_fills_mk.mean(),
    )


if __name__ == "__main__":
    print({k: round(v, 1) for k, v in sigma60().items()})


def evaluate_pairs(path, pairs, scenario="base", news=None, over=None, attrs=None):
    """Like evaluate(), for an explicit list of (coef_file, session) pairs."""
    jobs = [(os.path.abspath(path), c, n, scenario, over, attrs, news) for c, n in pairs]
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        out = list(pool.map(_one, jobs))
    return pd.DataFrame([o[0] for o in out]), pd.concat([o[1] for o in out if len(o[1])])
