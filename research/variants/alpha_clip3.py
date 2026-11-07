"""[research variant clip3: feature patches ['clip']; see research/variants/]

Alpha: a fitted short-horizon fair value for all four symbols, traded by quoting and taking.

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

v4 (ported from the PQT strategy; v3 is in strategies/archive/):
* quotes also lean against net exposure (`expo_lean`), freeing exposure capacity;
* ETF arbitrage: IOC all three legs when HPPO and BNKY + 2 BRRY cross by more than the cost
  (`arb_on`, `arb_min`), and matched HPPO vs basket inventory is converted away (`recycle_on`);
* live controls: `enabled` (kill), `pause_SYM` (stand aside on one symbol), `bias_SYM` (shift
  fair value by a human view, in cents; with taking on, a big enough bias also takes). Reset
  bias to 0 afterwards.

v5 (v4 is in strategies/archive/): the model's adjustment is scaled per symbol by its live
calibration (`calibrate()`, GUI switch `calib_on`), so a model that stops predicting fades toward
quoting around the micro; the arbitrage never sizes off a guessed level (`Ext.guessed_*`) and counts
BRRY's split orders against the message budget. research/ROADMAP.md has the numbers.
"""

import json
import math
from collections import deque

from pqt_client.v2 import BatchState, ParamKind, ParamSpec, Side, Strategy

SYMS = ("HPPO", "BNKY", "CRRT", "BRRY")
WEIGHT = {"HPPO": 1.0, "BNKY": 0.15, "CRRT": -0.29, "BRRY": 0.425}
EXPOSURE_CAP = 580.0  # the exchange's limit is 600; keep a margin
TAKER_FEE = 0.4  # cents per share
WARMUP = 30  # batches this program must have seen before trading (also after a late start)
HALT_COOLDOWN = 20  # batches to stay out of a symbol after its halt ends
REPORT_EVERY = 120  # one status line a minute

# --- fitted coefficients (research/fit.py); replaced by research/bake.py, keep the markers ---
# BEGIN COEF
COEF_JSON = r"""{"H":120,"train":["s01","s02","s03","s04","s05","s06","s07","s08","s09","s10","s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21","s22","s23","s24"],"lags":[10,30,60,120,240],"flows":[30,120],"coef":{"HPPO":{"HPPO_r10":0.29776,"HPPO_r30":0.29611,"HPPO_r60":0.08317,"HPPO_r120":0.03809,"HPPO_r240":-0.01317,"HPPO_f30":1.01824,"HPPO_f120":1.21654,"HPPO_um":-0.94338,"BNKY_r10":0.0951,"BNKY_r30":0.12543,"BNKY_r60":-0.04674,"BNKY_r120":-0.01938,"BNKY_r240":0.00901,"BNKY_f30":0.29431,"BNKY_f120":0.07168,"BNKY_um":0.11446,"CRRT_r10":-0.15548,"CRRT_r30":-0.14718,"CRRT_r60":-0.06774,"CRRT_r120":0.09139,"CRRT_r240":-0.01869,"CRRT_f30":-0.36192,"CRRT_f120":-0.08668,"CRRT_um":0.23261,"BRRY_r10":0.01893,"BRRY_r30":0.01999,"BRRY_r60":-0.00288,"BRRY_r120":-0.00773,"BRRY_r240":0.01411,"BRRY_f30":0.64006,"BRRY_f120":0.56239,"BRRY_um":-0.07342,"gapB":-0.08381},"BNKY":{"HPPO_r10":0.08218,"HPPO_r30":0.04072,"HPPO_r60":-0.00652,"HPPO_r120":0.0211,"HPPO_r240":0.01108,"HPPO_f30":0.19284,"HPPO_f120":0.08884,"HPPO_um":-0.12963,"BNKY_r10":-0.00411,"BNKY_r30":0.33548,"BNKY_r60":0.1839,"BNKY_r120":-0.05279,"BNKY_r240":-0.01037,"BNKY_f30":-0.00246,"BNKY_f120":0.17949,"BNKY_um":-0.7473,"CRRT_r10":-0.01467,"CRRT_r30":0.00972,"CRRT_r60":0.01116,"CRRT_r120":-0.03516,"CRRT_r240":0.03542,"CRRT_f30":-0.17546,"CRRT_f120":0.0439,"CRRT_um":0.00058,"BRRY_r10":-0.00354,"BRRY_r30":-0.00929,"BRRY_r60":0.00658,"BRRY_r120":0.00029,"BRRY_r240":-0.01145,"BRRY_f30":0.19803,"BRRY_f120":-0.0748,"BRRY_um":0.01189,"gapB":0.00237},"CRRT":{"HPPO_r10":-0.0916,"HPPO_r30":-0.08193,"HPPO_r60":-0.04037,"HPPO_r120":0.01718,"HPPO_r240":-0.01318,"HPPO_f30":-0.18995,"HPPO_f120":-0.23693,"HPPO_um":0.11516,"BNKY_r10":-0.01188,"BNKY_r30":0.02577,"BNKY_r60":0.04552,"BNKY_r120":-0.01575,"BNKY_r240":0.0522,"BNKY_f30":-0.01538,"BNKY_f120":-0.01761,"BNKY_um":-0.09346,"CRRT_r10":-0.01172,"CRRT_r30":0.1345,"CRRT_r60":0.14508,"CRRT_r120":0.01751,"CRRT_r240":-0.08778,"CRRT_f30":0.11991,"CRRT_f120":0.21263,"CRRT_um":-0.87628,"BRRY_r10":-0.01167,"BRRY_r30":-0.01957,"BRRY_r60":-0.01145,"BRRY_r120":-0.02094,"BRRY_r240":-0.02976,"BRRY_f30":0.27298,"BRRY_f120":-0.14415,"BRRY_um":0.00626,"gapB":-0.06986},"BRRY":{"HPPO_r10":0.14402,"HPPO_r30":0.15499,"HPPO_r60":0.06589,"HPPO_r120":0.00925,"HPPO_r240":0.00818,"HPPO_f30":0.12778,"HPPO_f120":0.46266,"HPPO_um":-0.33903,"BNKY_r10":0.02605,"BNKY_r30":-0.11479,"BNKY_r60":-0.13689,"BNKY_r120":0.04959,"BNKY_r240":-0.02398,"BNKY_f30":0.01986,"BNKY_f120":-0.07098,"BNKY_um":0.2565,"CRRT_r10":-0.04783,"CRRT_r30":-0.08478,"CRRT_r60":-0.05371,"CRRT_r120":-0.03075,"CRRT_r240":0.02791,"CRRT_f30":0.05595,"CRRT_f120":-0.14174,"CRRT_um":0.11464,"BRRY_r10":-0.04834,"BRRY_r30":0.01364,"BRRY_r60":0.00674,"BRRY_r120":-0.04095,"BRRY_r240":-0.00016,"BRRY_f30":-0.12768,"BRRY_f120":0.44946,"BRRY_um":-0.26444,"gapB":0.49616}}}"""
# END COEF
MODEL = json.loads(COEF_JSON)


class Ext:
    """The top of book without our own orders. `guessed_bid` / `guessed_ask`: that side is not
    observed (we were alone at the best price, so the level behind us is a guess)."""

    __slots__ = ("bid", "bq", "ask", "aq", "guessed_bid", "guessed_ask")

    def __init__(self, bid, bq, ask, aq, guessed_bid=False, guessed_ask=False):
        self.bid, self.bq, self.ask, self.aq = bid, bq, ask, aq
        self.guessed_bid, self.guessed_ask = guessed_bid, guessed_ask

    @property
    def micro(self):
        return (self.bid * self.aq + self.ask * self.bq) / (self.bq + self.aq)

    @property
    def mid(self):
        return (self.bid + self.ask) / 2


class Alpha_clip3(Strategy):
    MODEL_PARAMS = (
        ParamSpec("make_edge", ParamKind.FLOAT, 2.5, 0.0, 50.0),  # cents from fair to quote
        ParamSpec("take_edge", ParamKind.FLOAT, 6.0, 0.0, 50.0),  # cents beyond the fee to take
        ParamSpec("size_scale", ParamKind.FLOAT, 1.5, 0.0, 3.0),  # scales clip and quote sizes
        ParamSpec("make_on", ParamKind.BOOL, 1, 0, 1),
        ParamSpec("take_on", ParamKind.BOOL, 1, 0, 1),
        ParamSpec("news_pause", ParamKind.INT, 10, 0, 240),  # batches without quotes after news
        ParamSpec("kill_floor", ParamKind.INT, 20000, 0, 1000000),  # $ realized loss; 0 = off
        # v4, ported from the PQT strategy and checked by nested CV (ANALYSIS.md appendix D)
        ParamSpec("enabled", ParamKind.BOOL, 1, 0, 1),  # 0: cancel everything and stop
        ParamSpec("calib_on", ParamKind.BOOL, 1, 0, 1),  # scale the model by its live calibration
        ParamSpec("expo_lean", ParamKind.FLOAT, 3.0, 0.0, 20.0),  # c of lean against net exposure at the cap
        ParamSpec("arb_on", ParamKind.BOOL, 1, 0, 1),  # ETF vs basket arbitrage
        ParamSpec("arb_min", ParamKind.FLOAT, 1.0, -2.0, 30.0),  # c/unit beyond 4 taker fees + 2c conversion
        ParamSpec("recycle_on", ParamKind.BOOL, 1, 0, 1),  # convert matched HPPO vs basket inventory
        ParamSpec("bias_HPPO", ParamKind.FLOAT, 0.0, -200.0, 200.0),  # human fair-value shift, cents
        ParamSpec("bias_BNKY", ParamKind.FLOAT, 0.0, -200.0, 200.0),
        ParamSpec("bias_CRRT", ParamKind.FLOAT, 0.0, -200.0, 200.0),
        ParamSpec("bias_BRRY", ParamKind.FLOAT, 0.0, -200.0, 200.0),
        ParamSpec("pause_HPPO", ParamKind.BOOL, 0, 0, 1),  # 1: no orders on this symbol
        ParamSpec("pause_BNKY", ParamKind.BOOL, 0, 0, 1),
        ParamSpec("pause_CRRT", ParamKind.BOOL, 0, 0, 1),
        ParamSpec("pause_BRRY", ParamKind.BOOL, 0, 0, 1),
    )
    CONV_BUSY = 0.6  # convert once some ETF leg uses this share of its soft cap
    CONV_MIN = 40  # smallest conversion worth a message
    # per-symbol knobs (shares); the edges above are scaled per symbol by EDGE_MULT
    CAP = {"HPPO": 400, "BNKY": 1500, "CRRT": 1500, "BRRY": 400}
    CLIP = {"HPPO": 100, "BNKY": 300, "CRRT": 300, "BRRY": 60}
    QUOTE = {"HPPO": 50, "BNKY": 150, "CRRT": 150, "BRRY": 30}
    EDGE_MULT = {"HPPO": 1.0, "BNKY": 0.5, "CRRT": 0.6, "BRRY": 1.0}
    SKEW = {"HPPO": 4.5, "BNKY": 1.5, "CRRT": 2.25, "BRRY": 4.5}  # cents of lean at CAP
    # |fair - micro| never exceeds this (cents): ~4x each symbol's typical predicted 60 s move.
    # A fixed guard against feature outliers (news jumps, a bad print), not a tuned value.
    FAIR_CAP = {"HPPO": 20.0, "BNKY": 6.5, "CRRT": 8.5, "BRRY": 20.0}
    # After a news pause, remove the news jump from the price history so it is not read as momentum.
    NEWS_REBASE = True
    # Online calibration: per symbol, the decayed regression slope of the realized target
    # (mid H batches later minus micro now) on the model's adjustment, shrunk toward 1 by a prior
    # worth CALIB_PRIOR batches. The adjustment is scaled by that slope, clipped to [0, 1]: a model
    # that stops working live (wrong sign, overconfident) fades toward quoting around the micro.
    CALIB_ON = True
    CALIB_HALFLIFE = 2400  # batches (20 min)
    CALIB_PRIOR = 1200  # batches of prior weight at slope 1
    CALIB_FULL = 1.0  # estimated slope at which the model gets full weight (k = slope / this)
    # One-sided quoting (research, off by default): when |fair - micro| exceeds ONE_SIDE_K times the
    # symbol's quoting edge, the side the forecast says will be picked off is widened to
    # ONE_SIDE_WIDEN times the edge, or pulled when ONE_SIDE_WIDEN is 0.
    ONE_SIDE_K = 0.0  # 0 = off
    ONE_SIDE_WIDEN = 0.0
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
        self.news_batch = None  # first news batch of a pending rebase
        self.news_ref: dict[str, float] = {}  # micro just before that news
        self.rebase_at = None
        self.seen = 0  # batches seen by this program (a late start has no history yet)
        self.horizon = int(m.get("H", 120))
        self.adj_hist = {s: deque(maxlen=self.horizon + 1) for s in SYMS}  # (adj, micro) per batch
        self.cal = {s: [0.0, 0.0, 0.0] for s in SYMS}  # decayed sums: p*y, p*p, count
        self.cal_decay = 0.5 ** (1.0 / self.CALIB_HALFLIFE)
        self.k = {s: 1.0 for s in SYMS}
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
        for h in self.adj_hist.values():  # a news jump is not the model's error: drop pending samples
            h.clear()
        if self.NEWS_REBASE:
            if self.rebase_at is None:  # keep the reference from before the first of several items
                self.news_batch = item.batch
                self.news_ref = {s: e.micro for s, e in self.ext.items()}
            self.rebase_at = max(self.rebase_at or 0, item.batch + max(pause, 1))

    def on_reconnect(self) -> None:
        pass  # resting orders are still there and a news pause stays in force: carry on

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
        gb = ga = False
        if self.mode != "backtest":
            prev = self.ext.get(s)
            mb = sum(o.remaining for o in self.open_orders(s) if o.side == "BUY" and o.price == bid)
            if mb:
                if bq > mb:
                    bq -= mb
                elif prev is not None and prev.bid < bid:
                    bid, bq, gb = prev.bid, prev.bq, True
                else:
                    bid, bq, gb = bid - 1, 100, True
            ma = sum(o.remaining for o in self.open_orders(s) if o.side == "SELL" and o.price == ask)
            if ma:
                if aq > ma:
                    aq -= ma
                elif prev is not None and prev.ask > ask:
                    ask, aq, ga = prev.ask, prev.aq, True
                else:
                    ask, aq, ga = ask + 1, 100, True
            if bid >= ask:
                return prev
        return Ext(bid, bq, ask, aq, gb, ga)

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
        if self.rebase_at is not None and state.batch >= self.rebase_at:
            self.rebase(state.batch)
        return all(s in self.ext for s in SYMS)

    def rebase(self, batch: int) -> None:
        """Shift pre-news prices by the move since the news and flatten the pause window, so the
        return features measure only what happened after the market repriced. Flow traded during
        the pause (news-driven) is dropped too."""
        k = batch - self.news_batch + 1  # history entries from the news batch on
        for s in SYMS:
            h = self.u_hist[s]
            if not h or s not in self.news_ref:
                continue
            now = h[-1]
            d = now - self.news_ref[s]
            n = min(k, len(h))
            vals = [u + d for u in list(h)[: len(h) - n]] + [now] * n
            self.u_hist[s] = deque(vals, maxlen=h.maxlen)
            f = list(self.f_hist[s])
            m = min(k, len(f))
            self.f_hist[s] = deque(f[: len(f) - m] + [0] * m, maxlen=self.f_hist[s].maxlen)
        self.news_batch, self.news_ref, self.rebase_at = None, {}, None

    def calibrate(self, s, batch, adj) -> None:
        """Score the adjustment made H batches ago against what happened, then queue this one."""
        h, c = self.adj_hist[s], self.cal[s]
        while h and h[0][0] <= batch - self.horizon:
            _, p, u0 = h.popleft()
            y = self.ext[s].mid - u0
            d = self.cal_decay
            c[0], c[1], c[2] = c[0] * d + p * y, c[1] * d + p * p, c[2] * d + 1
        h.append((batch, adj, self.ext[s].micro))
        if c[2] > 0 and c[1] > 0:
            prior = self.CALIB_PRIOR * c[1] / c[2]  # prior weight in units of the mean p*p
            slope = (c[0] + prior) / (c[1] + prior)
            self.k[s] = max(0.0, min(1.0, slope / self.CALIB_FULL))

    def compute_fair(self, batch=None) -> None:
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
        for k, lim in (self.model.get("clip") or {}).items():  # winsorise at training-set levels
            if k in x:
                x[k] = max(-lim, min(lim, x[k]))
        self.last_x = x  # research: feature parity check
        for s in SYMS:
            c = self.coef[s]
            adj = sum(c[k] * x[k] for k in c)
            cap = self.FAIR_CAP.get(s) if self.FAIR_CAP else None
            if cap is not None:
                adj = max(-cap, min(cap, adj))
            if self.CALIB_ON and batch is not None:
                self.calibrate(s, batch, adj)  # keeps learning while switched off in the GUI
                if self.param("calib_on"):
                    adj *= self.k[s]
            # a human bias (GUI) is added after calibration: it is a view, not the model's error
            self.fair[s] = self.ext[s].micro + adj + float(self.param(f"bias_{s}"))

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

    # ------------------------------------------------------------------ ETF arbitrage (from PQT)
    def trade_arb(self, state: BatchState, lw, sw) -> set:
        """HPPO's true value is exactly BNKY + 2 BRRY, so when the books cross by more than the cost
        (4 taker fees + the 2c conversion that later recycles the legs) take all three legs: a
        completed set scores its edge whatever the true value does."""
        legs = ("HPPO", "BNKY", "BRRY")
        for s in legs:
            t = state.top[s]
            if (t.halted or not t.two_sided or state.batch < self.cool_until[s]
                    or self.param(f"pause_{s}")):
                return set()
        h, b, r = (self.ext[s] for s in legs)
        need = 4 * TAKER_FEE + 2.0 + float(self.param("arb_min"))
        sell_etf = h.bid - (b.ask + 2 * r.ask)  # sell HPPO, buy the basket
        buy_etf = (b.bid + 2 * r.bid) - h.ask  # buy HPPO, sell the basket
        if sell_etf >= need:
            d, q = -1, min(h.bq, b.aq, r.aq // 2)
            guessed = h.guessed_bid or b.guessed_ask or r.guessed_ask
        elif buy_etf >= need:
            d, q = 1, min(h.aq, b.bq, r.bq // 2)
            guessed = h.guessed_ask or b.guessed_bid or r.guessed_bid
        else:
            return set()
        if guessed:  # a leg's price or size is our guess, not the book: IOCs could miss it
            return set()
        want = {"HPPO": d, "BNKY": -d, "BRRY": -d}  # +1 buy, -1 sell
        mult = {"HPPO": 1, "BNKY": 1, "BRRY": 2}
        for s in legs:
            q = min(q, self.room(s, "BUY" if want[s] > 0 else "SELL", lw, sw) // mult[s])
        q = min(q, self.symbol_info["HPPO"].max_order_qty, self.symbol_info["BNKY"].max_order_qty)
        orders = sum(-(-q * mult[s] // self.symbol_info[s].max_order_qty) for s in legs)
        if q < 5 or self.messages_left() < orders + sum(len(self.open_orders(s)) for s in legs):
            return set()
        for s in legs:  # never trade with ourselves: pull our quotes on the legs first
            for o in self.open_orders(s):
                self.cancel(order_id=o.order_id)
                if o.side == "BUY":
                    lw[s] -= o.remaining
                else:
                    sw[s] += o.remaining
        for s in legs:
            e = self.ext[s]
            n = q * mult[s]
            while n > 0:  # BRRY's leg may need two orders (max order 120)
                k = min(n, self.symbol_info[s].max_order_qty)
                if want[s] > 0:
                    self.buy(s, k, e.ask, ioc=True)
                    lw[s] += k
                else:
                    self.sell(s, k, e.bid, ioc=True)
                    sw[s] -= k
                n -= k
        self.log.info("arb b%d %s %d units, gap %.1fc", state.batch, "sellETF" if d < 0 else "buyETF",
                      q, sell_etf if d < 0 else buy_etf)
        return set(legs)

    def recycle(self, state: BatchState) -> None:
        """Convert matched HPPO vs basket inventory (arbitrage leftovers, mostly) to free position
        room. Exposure-neutral; costs 2c per HPPO share."""
        if any(state.top[s].halted for s in ("HPPO", "BNKY", "BRRY")) or self.messages_left() < 2:
            return
        h, b, r = self.position("HPPO"), self.position("BNKY"), self.position("BRRY")
        busy = max(abs(h) / self.CAP["HPPO"], abs(b) / self.CAP["BNKY"], abs(r) / self.CAP["BRRY"])
        if busy < self.CONV_BUSY:
            return
        if h < 0 and b > 0 and r > 1:
            q = min(-h, b, r // 2)
            if q >= self.CONV_MIN:
                self.create(q)
        elif h > 0 and b < 0 and r < -1:
            q = min(h, -b, -r // 2)
            if q >= self.CONV_MIN:
                self.redeem(q)

    # ------------------------------------------------------------------ trading
    def on_batch(self, state: BatchState) -> None:
        ready = self.update_features(state)
        if (state.batch + 1) % REPORT_EVERY == 0:
            self.report(state)
        if not self.param("enabled"):
            if any(self.open_orders(s) for s in SYMS):
                self.cancel_all()
            return
        self.seen += 1
        if not ready or self.seen <= WARMUP or state.final or state.blocked:
            return
        self.compute_fair(state.batch)
        quoting = bool(self.param("make_on")) and state.batch >= self.pause_until
        take_on = bool(self.param("take_on"))
        scale = float(self.param("size_scale"))
        lean = float(self.param("expo_lean"))
        lw, sw = self.worst_case()
        arb = self.trade_arb(state, lw, sw) if self.param("arb_on") else set()
        if self.param("recycle_on") and not arb:
            self.recycle(state)
        E = sum(WEIGHT[s] * self.position(s) for s in SYMS)  # net exposure of positions
        for s in SYMS:
            if s in arb:  # its quotes were just pulled for the arbitrage
                continue
            top = state.top[s]
            if self.param(f"pause_{s}"):
                if self.open_orders(s):
                    self.cancel_all(s)
                continue
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
                # x is the book without our orders, so a take at x's price would first hit any of
                # our own quotes at or through it (a self-trade, penalised): skip until it moves.
                if f - x.ask > take_edge and not self.own_at_or_through(s, "SELL", x.ask):
                    q = min(int(self.CLIP[s] * scale), x.aq, self.room(s, "BUY", lw, sw), maxq)
                    if q > 0:
                        self.buy(s, q, x.ask, ioc=True)
                        lw[s] += q
                elif x.bid - f > take_edge and not self.own_at_or_through(s, "BUY", x.bid):
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
            # lean against net exposure too (from PQT): the strategy is exposure-capped, so steering
            # E back toward 0 frees capacity on both sides
            centre -= lean * E / EXPOSURE_CAP * (1 if WEIGHT[s] > 0 else -1) * min(1.0, abs(WEIGHT[s]) * 2)
            e = float(self.param("make_edge")) * mult
            bid = min(math.floor(centre - e), x.ask - 1)  # never cross the (external) book
            ask = max(math.ceil(centre + e), x.bid + 1)
            if bid >= ask:
                continue
            size = max(1, int(self.QUOTE[s] * scale))
            bsize = asize = size
            sig = f - x.micro
            if self.ONE_SIDE_K and abs(sig) > self.ONE_SIDE_K * e:
                if sig > 0:  # price expected up: our ask is the quote that gets picked off
                    if self.ONE_SIDE_WIDEN:
                        ask = max(math.ceil(centre + e * self.ONE_SIDE_WIDEN), x.bid + 1)
                    else:
                        asize = 0
                else:
                    if self.ONE_SIDE_WIDEN:
                        bid = min(math.floor(centre - e * self.ONE_SIDE_WIDEN), x.ask - 1)
                    else:
                        bsize = 0
            # Never cross our own resting order: our messages are processed in the order sent,
            # so move the side that is getting out of the way first.
            cur_ask = min((o.price for o in self.open_orders(s) if o.side == "SELL"), default=None)
            if cur_ask is not None and bid >= cur_ask:
                self.keep_quote(s, Side.SELL, ask, asize, lw, sw)
                self.keep_quote(s, Side.BUY, bid, bsize, lw, sw)
            else:
                self.keep_quote(s, Side.BUY, bid, bsize, lw, sw)
                self.keep_quote(s, Side.SELL, ask, asize, lw, sw)

    def own_at_or_through(self, s, side, price) -> bool:
        """Do we rest an order on `side` that a take at `price` would trade against?"""
        if side == "SELL":
            return any(o.side == "SELL" and o.price <= price for o in self.open_orders(s))
        return any(o.side == "BUY" and o.price >= price for o in self.open_orders(s))

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
        k = " ".join(f"{s} {self.k[s]:.2f}" for s in SYMS)
        self.log.info("minute %d: %s | fair-micro %s | model weight %s | est PnL $%.2f",
                      (state.batch + 1) // REPORT_EVERY, pos, edge, k, state.pnl_est_mills / 1000)
