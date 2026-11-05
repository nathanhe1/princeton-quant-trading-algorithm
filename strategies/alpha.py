"""Alpha: a fitted short-horizon fair value for all four symbols, traded by taking and quoting.

    pqt backtest strategies/alpha.py --data data --name s01

Fair value. The market's prices follow the hidden true value with a lag: a symbol's recent moves,
the recent signed aggressor flow, other symbols' moves (CRRT moves against HPPO; BRRY lags HPPO
and BNKY through the ETF identity HPPO = BNKY + 2 BRRY) all predict where its mid will be a minute
later. A linear model fitted on the historical sessions (research/fit.py) gives, every batch,

    fair[s] = microprice[s] + sum_k coef[s][k] * feature_k

Scoring is the 60-second markout against the true value, so a fill is worth (fair - price) per
share bought, whatever happens to the position later. Two ways to collect it:

* take: buy at the ask when fair is above it by more than the taker fee plus an edge (sell
  likewise);
* make: keep one bid and one ask resting around fair (shaded against inventory), so others trade
  with us only at prices our model likes.

Limits: every order is checked against the worst-case position and the team's net exposure
(|E| <= 600) before it is sent, so the exchange should never need to reject one.
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

# --- fitted coefficients (research/fit.py); replaced by tooling, keep the markers ---
# BEGIN COEF
COEF_JSON = r"""{}"""
# END COEF
MODEL = json.loads(COEF_JSON)


class Alpha(Strategy):
    MODEL_PARAMS = (
        ParamSpec("take_edge", ParamKind.FLOAT, 1.0, 0.0, 50.0),  # cents beyond the fee to take
        ParamSpec("make_edge", ParamKind.FLOAT, 2.0, 0.0, 50.0),  # cents from fair to quote
        ParamSpec("size_scale", ParamKind.FLOAT, 1.0, 0.0, 2.0),  # scales clip and quote sizes
        ParamSpec("make_on", ParamKind.BOOL, 1, 0, 1),
        ParamSpec("take_on", ParamKind.BOOL, 1, 0, 1),
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
        self.last_u: dict[str, float] = {}
        self.last_m: dict[str, float] = {}
        self.own_taker: dict[str, int] = {s: 0 for s in SYMS}  # our aggressor shares this batch
        self.fair: dict[str, float] = {}

    def on_fill(self, fill) -> None:
        if fill.liquidity == "TAKER":
            self.own_taker[fill.symbol] += fill.qty if fill.side == "BUY" else -fill.qty

    def param(self, name):
        if self.params_override and name in self.params_override:
            return self.params_override[name]
        return self.model_params[name]

    # ------------------------------------------------------------------ features
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
            if top.two_sided and not top.halted:
                self.last_u[s] = top.microprice
                self.last_m[s] = top.mid
            if s in self.last_u:
                self.u_hist[s].append(self.last_u[s])
            self.f_hist[s].append(flow[s])
        return all(s in self.last_u for s in SYMS)

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
            x[f"{s}_um"] = u - self.last_m[s]
        x["gapB"] = (self.last_u["HPPO"] - self.last_u["BNKY"]) / 2 - self.last_u["BRRY"]
        for s in SYMS:
            c = self.coef[s]
            self.fair[s] = self.last_u[s] + sum(c[k] * x[k] for k in c)

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
        lim = self.symbol_info[s].position_limit
        cap = min(self.CAP[s], lim)
        hi, lo = self.exposure_bounds(lw, sw)
        w = WEIGHT[s]
        if side == "BUY":
            r = cap - lw[s]
            if w > 0:
                r = min(r, (EXPOSURE_CAP - hi) / w)
            else:
                r = min(r, (lo + EXPOSURE_CAP) / -w)
        else:
            r = cap + sw[s]
            if w > 0:
                r = min(r, (lo + EXPOSURE_CAP) / w)
            else:
                r = min(r, (EXPOSURE_CAP - hi) / -w)
        return max(0, int(r))

    # ------------------------------------------------------------------ trading
    def on_batch(self, state: BatchState) -> None:
        ready = self.update_features(state)
        if not ready or state.batch < WARMUP or state.final:
            return
        self.compute_fair()
        take_on, make_on = self.param("take_on"), self.param("make_on")
        scale = float(self.param("size_scale"))
        lw, sw = self.worst_case()
        for s in SYMS:
            top = state.top[s]
            if top.halted or not top.two_sided:
                continue
            f = self.fair[s]
            mult = self.EDGE_MULT[s]
            take_edge = TAKER_FEE + float(self.param("take_edge")) * mult
            pos = self.position(s)
            # --- take
            if take_on:
                if f - top.ask_px > take_edge:
                    q = min(int(self.CLIP[s] * scale), top.ask_qty, self.room(s, "BUY", lw, sw),
                            self.symbol_info[s].max_order_qty)
                    if q > 0:
                        self.buy(s, q, top.ask_px, ioc=True)
                        lw[s] += q
                elif top.bid_px - f > take_edge:
                    q = min(int(self.CLIP[s] * scale), top.bid_qty, self.room(s, "SELL", lw, sw),
                            self.symbol_info[s].max_order_qty)
                    if q > 0:
                        self.sell(s, q, top.bid_px, ioc=True)
                        sw[s] -= q
            # --- make
            if make_on:
                cap = self.CAP[s]
                centre = f - self.SKEW[s] * pos / cap
                e = float(self.param("make_edge")) * mult
                bid = min(math.floor(centre - e), top.ask_px - 1)
                ask = max(math.ceil(centre + e), top.bid_px + 1)
                size = max(1, int(self.QUOTE[s] * scale))
                self.keep_quote(s, Side.BUY, bid, size, lw, sw)
                self.keep_quote(s, Side.SELL, ask, size, lw, sw)

    def keep_quote(self, s, side, price, size, lw, sw) -> None:
        mine = [o for o in self.open_orders(s) if o.side is side]
        for extra in mine[1:]:
            self.cancel(order_id=extra.order_id)
        cur = mine[0] if mine else None
        # room excluding our current quote on this side (it gets replaced)
        if cur is not None:
            if side is Side.BUY:
                lw[s] -= cur.remaining
            else:
                sw[s] += cur.remaining
        q = min(size, self.room(s, side.value, lw, sw))
        if q <= 0:
            if cur is not None:
                self.cancel(order_id=cur.order_id)
            return
        if cur is None:
            (self.buy if side is Side.BUY else self.sell)(s, q, price)
        elif cur.price != price or cur.remaining > q:
            self.replace(order_id=cur.order_id, price=price, qty=q)
        else:
            q = cur.remaining
        if side is Side.BUY:
            lw[s] += q
        else:
            sw[s] -= q
