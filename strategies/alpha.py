"""Alpha: a fitted short-horizon fair value for all four symbols, traded by quoting and taking.

    pqt backtest strategies/alpha.py --data data --name s01
    pqt run strategies/alpha.py --mode test --server https://sandbox.princeton-quant.com --key KEY

Fair value. The market's prices follow the hidden true value with a lag: a symbol's recent moves,
the recent signed aggressor flow, other symbols' moves (CRRT moves against HPPO; BRRY lags HPPO
and BNKY through the ETF identity HPPO = BNKY + 2 BRRY) all predict where its mid will be a minute
later. A linear model fitted on the historical sessions (research/fit.py, baked in below by
research/bake.py) gives, every batch,

    fair[s] = microprice[s] + sum_k coef[s][k] * feature_k

Scoring is the 60-second markout against the true value, so a fill is worth (fair - price) per
share bought, whatever happens to the position later. Two ways to collect it:

* make: keep one bid and one ask resting around fair (shaded against inventory), so others trade
  with us only at prices the model likes. Most of the edge.
* take: hit the touch only when fair is beyond it by a wide margin.

Safety. Every order is checked against the worst-case position and the team's net exposure
(|E| <= 600) before it is sent. Live, our own quotes are removed from the book before computing
features (otherwise the model would chase its own quotes). Quotes are pulled for a few seconds
after any news item and after a symbol comes back from a halt, when the model's fair is stale.
"""

import json
import math
from collections import deque

from pqt_client.v2 import BatchState, ParamKind, ParamSpec, Side, Strategy

SYMS = ("HPPO", "BNKY", "CRRT", "BRRY")
WEIGHT = {"HPPO": 1.0, "BNKY": 0.15, "CRRT": -0.29, "BRRY": 0.425}
EXPOSURE_CAP = 580.0  # the exchange's limit is 600; keep a margin
TAKER_FEE = 0.4  # cents per share
WARMUP = 30  # batches before trading
HALT_COOLDOWN = 20  # batches to stay out of a symbol after its halt ends
REPORT_EVERY = 120  # one status line a minute

# --- fitted coefficients (research/fit.py); replaced by research/bake.py, keep the markers ---
# BEGIN COEF
COEF_JSON = r"""{}"""
# END COEF
MODEL = json.loads(COEF_JSON)


class Ext:
    """The top of book without our own orders."""

    __slots__ = ("bid", "bq", "ask", "aq")

    def __init__(self, bid, bq, ask, aq):
        self.bid, self.bq, self.ask, self.aq = bid, bq, ask, aq

    @property
    def micro(self):
        return (self.bid * self.aq + self.ask * self.bq) / (self.bq + self.aq)

    @property
    def mid(self):
        return (self.bid + self.ask) / 2


class Alpha(Strategy):
    MODEL_PARAMS = (
        ParamSpec("make_edge", ParamKind.FLOAT, 2.5, 0.0, 50.0),  # cents from fair to quote
        ParamSpec("take_edge", ParamKind.FLOAT, 6.0, 0.0, 50.0),  # cents beyond the fee to take
        ParamSpec("size_scale", ParamKind.FLOAT, 1.5, 0.0, 3.0),  # scales clip and quote sizes
        ParamSpec("make_on", ParamKind.BOOL, 1, 0, 1),
        ParamSpec("take_on", ParamKind.BOOL, 1, 0, 1),
        ParamSpec("news_pause", ParamKind.INT, 10, 0, 240),  # batches without quotes after news
        ParamSpec("kill_floor", ParamKind.INT, 20000, 0, 1000000),  # $ realized loss; 0 = off
    )
    # per-symbol knobs (shares); the edges above are scaled per symbol by EDGE_MULT
    CAP = {"HPPO": 400, "BNKY": 1500, "CRRT": 1500, "BRRY": 400}
    CLIP = {"HPPO": 100, "BNKY": 300, "CRRT": 300, "BRRY": 60}
    QUOTE = {"HPPO": 50, "BNKY": 150, "CRRT": 150, "BRRY": 30}
    EDGE_MULT = {"HPPO": 1.0, "BNKY": 0.5, "CRRT": 0.6, "BRRY": 1.0}
    SKEW = {"HPPO": 3.0, "BNKY": 1.0, "CRRT": 1.5, "BRRY": 3.0}  # cents of lean at CAP
    model = MODEL
    params_override: dict | None = None  # for research sweeps

    # ------------------------------------------------------------------ lifecycle
    def on_start(self) -> None:
        m = self.model
        self.lags = m.get("lags", [10, 30, 60, 120, 240])
        self.flows = m.get("flows", [30, 120])
        self.coef = m["coef"]
        n = max(self.lags) + 1
        self.u_hist = {s: deque(maxlen=n) for s in SYMS}
        self.f_hist = {s: deque(maxlen=max(self.flows)) for s in SYMS}
        self.ext: dict[str, Ext] = {}
        self.own_taker: dict[str, int] = {s: 0 for s in SYMS}  # our aggressor shares this batch
        self.fair: dict[str, float] = {}
        self.pause_until = 0
        self.was_halted = {s: False for s in SYMS}
        self.cool_until = {s: 0 for s in SYMS}
        floor = int(self.param("kill_floor"))
        if floor > 0:
            self.set_kill_switch(-floor * 1000)

    def on_fill(self, fill) -> None:
        if fill.liquidity == "TAKER":
            self.own_taker[fill.symbol] += fill.qty if fill.side == "BUY" else -fill.qty

    def on_news(self, item) -> None:
        pause = int(self.param("news_pause"))
        if pause > 0:
            self.log.info("news at batch %d (%s): quotes off for %d batches", item.batch, item.headline, pause)
            self.pause_until = max(self.pause_until, item.batch + pause)
            self.cancel_all()

    def on_reconnect(self) -> None:
        self.pause_until = 0  # resting orders are still there; carry on

    def param(self, name):
        if self.params_override and name in self.params_override:
            return self.params_override[name]
        return self.model_params[name]

    # ------------------------------------------------------------------ book and features
    def external(self, s, top) -> Ext | None:
        """The touch without our own resting orders (live only: the recorded backtest book never
        contains them). If we are alone at the best price, the level behind us is unknown: use
        the last one we saw, or one tick behind ours."""
        if not top.two_sided:
            return None
        bid, bq, ask, aq = top.bid_px, top.bid_qty, top.ask_px, top.ask_qty
        if self.mode != "backtest":
            prev = self.ext.get(s)
            mb = sum(o.remaining for o in self.open_orders(s) if o.side == "BUY" and o.price == bid)
            if mb:
                if bq > mb:
                    bq -= mb
                elif prev is not None and prev.bid < bid:
                    bid, bq = prev.bid, prev.bq
                else:
                    bid, bq = bid - 1, 100
            ma = sum(o.remaining for o in self.open_orders(s) if o.side == "SELL" and o.price == ask)
            if ma:
                if aq > ma:
                    aq -= ma
                elif prev is not None and prev.ask > ask:
                    ask, aq = prev.ask, prev.aq
                else:
                    ask, aq = ask + 1, 100
            if bid >= ask:
                return prev
        return Ext(bid, bq, ask, aq)

    def update_features(self, state: BatchState) -> bool:
        flow = {s: 0 for s in SYMS}
        for tr in state.trades:
            if tr.symbol in flow:
                flow[tr.symbol] += tr.qty if tr.aggressor == "BUY" else -tr.qty
        if self.mode != "backtest":  # the live tape includes our own trades: leave them out
            for s in SYMS:
                flow[s] -= self.own_taker[s]
        self.own_taker = {s: 0 for s in SYMS}
        for s in SYMS:
            top = state.top[s]
            if top.halted:
                self.was_halted[s] = True
            else:
                if self.was_halted[s]:
                    self.was_halted[s] = False
                    self.cool_until[s] = state.batch + HALT_COOLDOWN
                e = self.external(s, top)
                if e is not None:
                    self.ext[s] = e
            if s in self.ext:
                self.u_hist[s].append(self.ext[s].micro)
            self.f_hist[s].append(flow[s])
        return all(s in self.ext for s in SYMS)

    def compute_fair(self) -> None:
        x = {}
        for s in SYMS:
            h = self.u_hist[s]
            u = h[-1]
            for L in self.lags:
                x[f"{s}_r{L}"] = u - h[-1 - L] if len(h) > L else u - h[0]
            fh = list(self.f_hist[s])
            for L in self.flows:
                x[f"{s}_f{L}"] = sum(fh[-L:]) / 1000.0
            x[f"{s}_um"] = u - self.ext[s].mid
        x["gapB"] = (self.ext["HPPO"].micro - self.ext["BNKY"].micro) / 2 - self.ext["BRRY"].micro
        for s in SYMS:
            c = self.coef[s]
            self.fair[s] = self.ext[s].micro + sum(c[k] * x[k] for k in c)

    # ------------------------------------------------------------------ risk
    def worst_case(self):
        """Worst-case long and short per symbol over positions and resting orders."""
        lw, sw = {}, {}
        for s in SYMS:
            p = self.position(s)
            buys = sum(o.remaining for o in self.open_orders(s) if o.side == "BUY")
            sells = sum(o.remaining for o in self.open_orders(s) if o.side == "SELL")
            lw[s], sw[s] = p + buys, p - sells
        return lw, sw

    def exposure_bounds(self, lw, sw):
        hi = sum(WEIGHT[s] * (lw[s] if WEIGHT[s] > 0 else sw[s]) for s in SYMS)
        lo = sum(WEIGHT[s] * (sw[s] if WEIGHT[s] > 0 else lw[s]) for s in SYMS)
        return hi, lo

    def room(self, s, side, lw, sw) -> int:
        """Shares we may add on `side` of `s` within the soft cap, the hard limit and exposure."""
        cap = min(self.CAP[s], self.symbol_info[s].position_limit)
        hi, lo = self.exposure_bounds(lw, sw)
        w = WEIGHT[s]
        if side == "BUY":
            r = cap - lw[s]
            r = min(r, (EXPOSURE_CAP - hi) / w if w > 0 else (lo + EXPOSURE_CAP) / -w)
        else:
            r = cap + sw[s]
            r = min(r, (lo + EXPOSURE_CAP) / w if w > 0 else (EXPOSURE_CAP - hi) / -w)
        return max(0, int(r))

    # ------------------------------------------------------------------ trading
    def on_batch(self, state: BatchState) -> None:
        ready = self.update_features(state)
        if (state.batch + 1) % REPORT_EVERY == 0:
            self.report(state)
        if not ready or state.batch < WARMUP or state.final or state.blocked:
            return
        self.compute_fair()
        quoting = bool(self.param("make_on")) and state.batch >= self.pause_until
        take_on = bool(self.param("take_on"))
        scale = float(self.param("size_scale"))
        lw, sw = self.worst_case()
        for s in SYMS:
            top = state.top[s]
            if top.halted or not top.two_sided or state.batch < self.cool_until[s]:
                if self.open_orders(s) and not top.halted:
                    self.cancel_all(s)
                continue
            x = self.ext[s]
            f = self.fair[s]
            mult = self.EDGE_MULT[s]
            pos = self.position(s)
            # --- take
            if take_on:
                take_edge = TAKER_FEE + float(self.param("take_edge")) * mult
                maxq = self.symbol_info[s].max_order_qty
                if f - x.ask > take_edge:
                    q = min(int(self.CLIP[s] * scale), x.aq, self.room(s, "BUY", lw, sw), maxq)
                    if q > 0:
                        self.buy(s, q, x.ask, ioc=True)
                        lw[s] += q
                elif x.bid - f > take_edge:
                    q = min(int(self.CLIP[s] * scale), x.bq, self.room(s, "SELL", lw, sw), maxq)
                    if q > 0:
                        self.sell(s, q, x.bid, ioc=True)
                        sw[s] -= q
            # --- make
            if not quoting:
                if self.open_orders(s):
                    self.cancel_all(s)
                continue
            centre = f - self.SKEW[s] * pos / self.CAP[s]
            e = float(self.param("make_edge")) * mult
            bid = min(math.floor(centre - e), x.ask - 1)  # never cross the (external) book
            ask = max(math.ceil(centre + e), x.bid + 1)
            if bid >= ask:
                continue
            size = max(1, int(self.QUOTE[s] * scale))
            # Never cross our own resting order: our messages are processed in the order sent,
            # so move the side that is getting out of the way first.
            cur_ask = min((o.price for o in self.open_orders(s) if o.side == "SELL"), default=None)
            if cur_ask is not None and bid >= cur_ask:
                self.keep_quote(s, Side.SELL, ask, size, lw, sw)
                self.keep_quote(s, Side.BUY, bid, size, lw, sw)
            else:
                self.keep_quote(s, Side.BUY, bid, size, lw, sw)
                self.keep_quote(s, Side.SELL, ask, size, lw, sw)

    def keep_quote(self, s, side, price, size, lw, sw) -> None:
        """Leave exactly one resting order on `side` at `price`, with as few messages as
        possible (a few UNKNOWN_ORDER rejects are expected when a quote fills as we move it)."""
        mine = [o for o in self.open_orders(s) if o.side == side]
        for extra in mine[1:]:
            self.cancel(order_id=extra.order_id)
            if side == Side.BUY:
                lw[s] -= extra.remaining
            else:
                sw[s] += extra.remaining
        cur = mine[0] if mine else None
        if cur is not None:  # room is computed as if our current quote were gone
            if side == Side.BUY:
                lw[s] -= cur.remaining
            else:
                sw[s] += cur.remaining
        q = min(size, self.room(s, side.value, lw, sw))
        if q <= 0:
            if cur is not None:
                self.cancel(order_id=cur.order_id)
            return
        if cur is None:
            (self.buy if side == Side.BUY else self.sell)(s, q, price)
        elif cur.price != price or cur.remaining > q:
            self.replace(order_id=cur.order_id, price=price, qty=q)
        else:
            q = cur.remaining
        if side == Side.BUY:
            lw[s] += q
        else:
            sw[s] -= q

    def report(self, state: BatchState) -> None:
        pos = " ".join(f"{s} {self.position(s):+d}" for s in SYMS)
        edge = " ".join(f"{s} {self.fair[s] - self.ext[s].micro:+.1f}" for s in SYMS if s in self.fair)
        self.log.info("minute %d: %s | fair-micro %s | est PnL $%.2f", (state.batch + 1) // REPORT_EVERY,
                      pos, edge, state.pnl_est_mills / 1000)
