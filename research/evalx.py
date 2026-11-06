"""One evaluation harness: 3-fold out-of-sample markouts, optionally under a stressed fill model
(stress.py) and with synthetic news. Parallel over sessions.

    from evalx import evaluate
    summ, fills = evaluate("../strategies/alpha.py", coefs="std", scenario="base", news=None)

coefs: "raw" -> fold{k}.json (old fit), "std" -> fold{k}_std.json (scaled ridge, nested-CV penalty).
news:  None, or dict(D=<repricing batches>, n=<items per session>, mults=(2, 4), seed=...).

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
        by_batch.setdefault(e["t"], []).append(wire.NewsItem(e["id"], e["t"], f"synthetic {e['src']}", ""))

    original_send = v2bt._Feed.send

    def send(self, message):
        if by_batch and isinstance(message, wire.Batch) and message.batch in by_batch:
            message = dataclasses.replace(message, news=by_batch[message.batch])
        return original_send(self, message)

    scenario = next(s for s in stress.SCENARIOS if s.name == scenario_name)
    original_fills = stress._install(scenario, int(os.environ.get("STRESS_SEED", 2026))) if scenario_name != "base" else None
    v2bt._Feed.send = send
    try:
        strat = load_cls(path)()
        with open(os.path.join(HERE, coef_file)) as f:
            strat.model = json.load(f)
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
        import logging

        logging.disable(logging.INFO)
        backtest_strategy(session, strat)
    finally:
        v2bt._Feed.send = original_send
        if original_fills is not None:
            stress._Replay.rest_fills = original_fills

    df = pd.DataFrame(fills, columns=["batch", "sym", "q", "px", "fee", "liq", "pen"])
    summary = dict(sess=name, fills=len(df), shares=0, mk120=0.0, news_fills_mk=0.0)
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
        df["near_news"] = near
        df["sess"] = name
        summary.update(shares=int(df.q.abs().sum()), mk120=df.mk120.sum() / 100,
                       news_fills_mk=df.loc[df.near_news, "mk120"].sum() / 100)
    return summary, df


def evaluate(path, coefs="std", scenario="base", news=None, over=None, attrs=None, sessions_spec=None):
    suffix = "_std" if coefs == "std" else ""
    jobs = []
    for k, (coef_file, span) in enumerate(FOLDS, start=1):
        for name in pick(span):
            if sessions_spec and name not in pick(sessions_spec):
                continue
            jobs.append((os.path.abspath(path), f"fold{k}{suffix}.json", name, scenario, over, attrs, news))
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
