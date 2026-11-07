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

v4 (ported from the PQT strategy; v3 is in strategies/archive/):
* quotes also lean against net exposure (`expo_lean`), freeing exposure capacity;
* ETF arbitrage: IOC all three legs when HPPO and BNKY + 2 BRRY cross by more than the cost
  (`arb_on`, `arb_min`), and matched HPPO vs basket inventory is converted away (`recycle_on`);
* live controls: `enabled` (kill), `pause_SYM` (stand aside on one symbol), `bias_SYM` (shift
  fair value by a human view, in cents; with taking on, a big enough bias also takes). Reset
  bias to 0 afterwards.

v10: make_edge 2.5 -> 1.5. With the quote cap, narrower edges add volume without the far-inside
fills (nested CV picked 0.5; 1.5 keeps ~60% of the gain and loses less on the sandbox recording, a
market outside the historical range). Held out: realistic stress $2,707 -> $3,202 per session.

v9: a quote never sits more than MAX_IMPROVE (2) ticks better than the external best price on its
side. Before, a strong forecast or skew could pull a bid up to the ask minus one tick, where its fills
were poor value. Held out: base $6,382 -> $8,335, realistic stress $1,661 -> $2,707, wrong-signed
model $69 -> $1,208 per session. v8c (the same code without the limit) is in the archive.

v8: the ETF arbitrage defaults to off (arb_on 0). On the sandbox most arbitrages filled only some
legs, and even the backtest, which has no leg risk, scores it below zero. v7 is in the archive.

v7: book imbalance at the touch is a feature, and every feature is winsorised at 3x its training
RMS (`clip` in the model JSON; research/fit_imb_clip3.py). v6 = v5 + hardening and news (off).

v5 (v4 is in strategies/archive/): the model's adjustment is scaled per symbol by its live
calibration (`calibrate()`, GUI switch `calib_on`), so a model that stops predicting fades toward
quoting around the micro; the arbitrage never sizes off a guessed level (`Ext.guessed_*`) and counts
BRRY's split orders against the message budget. research/ROADMAP.md has the numbers.
"""

import json
import math
import re
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
COEF_JSON = r"""{"H":120,"train":["s01","s02","s03","s04","s05","s06","s07","s08","s09","s10","s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21","s22","s23","s24"],"lags":[10,30,60,120,240],"flows":[30,120],"variant":"imb_clip3","coef":{"HPPO":{"HPPO_r10":0.304485,"HPPO_r30":0.306479,"HPPO_r60":0.076776,"HPPO_r120":0.030637,"HPPO_r240":-0.010934,"HPPO_f30":0.833143,"HPPO_f120":1.527321,"HPPO_um":-1.909842,"BNKY_r10":0.101891,"BNKY_r30":0.102371,"BNKY_r60":-0.043344,"BNKY_r120":-0.048941,"BNKY_r240":0.021313,"BNKY_f30":0.317553,"BNKY_f120":0.154316,"BNKY_um":-0.091524,"CRRT_r10":-0.163197,"CRRT_r30":-0.143142,"CRRT_r60":-0.054839,"CRRT_r120":0.110936,"CRRT_r240":-0.017691,"CRRT_f30":-0.28733,"CRRT_f120":-0.260607,"CRRT_um":0.397844,"BRRY_r10":0.024347,"BRRY_r30":0.020346,"BRRY_r60":-0.003252,"BRRY_r120":-0.005869,"BRRY_r240":0.011651,"BRRY_f30":0.401127,"BRRY_f120":0.757924,"BRRY_um":-0.115166,"gapB":-0.078474,"HPPO_imb":2.798434,"BNKY_imb":0.446577,"CRRT_imb":-0.519143,"BRRY_imb":0.224295},"BNKY":{"HPPO_r10":0.080027,"HPPO_r30":0.042316,"HPPO_r60":-0.009068,"HPPO_r120":0.019155,"HPPO_r240":0.015074,"HPPO_f30":0.101517,"HPPO_f120":0.188488,"HPPO_um":-0.201932,"BNKY_r10":0.013647,"BNKY_r30":0.334018,"BNKY_r60":0.192436,"BNKY_r120":-0.076476,"BNKY_r240":-0.01396,"BNKY_f30":-0.053671,"BNKY_f120":0.300977,"BNKY_um":-1.350232,"CRRT_r10":-0.013819,"CRRT_r30":0.007973,"CRRT_r60":0.014033,"CRRT_r120":-0.035837,"CRRT_r240":0.035035,"CRRT_f30":-0.187373,"CRRT_f120":0.053942,"CRRT_um":-0.022194,"BRRY_r10":-0.004253,"BRRY_r30":-0.010487,"BRRY_r60":0.006295,"BRRY_r120":-0.003217,"BRRY_r240":-0.016271,"BRRY_f30":0.133041,"BRRY_f120":-0.04183,"BRRY_um":-0.061171,"gapB":-0.007769,"HPPO_imb":0.21956,"BNKY_imb":1.183268,"CRRT_imb":0.061279,"BRRY_imb":0.458466},"CRRT":{"HPPO_r10":-0.094757,"HPPO_r30":-0.080882,"HPPO_r60":-0.038469,"HPPO_r120":0.019938,"HPPO_r240":-0.013888,"HPPO_f30":-0.157962,"HPPO_f120":-0.309,"HPPO_um":0.344758,"BNKY_r10":-0.009072,"BNKY_r30":0.027738,"BNKY_r60":0.048606,"BNKY_r120":-0.016202,"BNKY_r240":0.051852,"BNKY_f30":-0.024394,"BNKY_f120":-0.021171,"BNKY_um":-0.204387,"CRRT_r10":0.001725,"CRRT_r30":0.142849,"CRRT_r60":0.148148,"CRRT_r120":0.014453,"CRRT_r240":-0.08588,"CRRT_f30":0.071885,"CRRT_f120":0.29675,"CRRT_um":-1.290029,"BRRY_r10":-0.013588,"BRRY_r30":-0.020556,"BRRY_r60":-0.010801,"BRRY_r120":-0.021613,"BRRY_r240":-0.029819,"BRRY_f30":0.387688,"BRRY_f120":-0.196479,"BRRY_um":0.023783,"gapB":-0.072031,"HPPO_imb":-0.67815,"BNKY_imb":0.210394,"CRRT_imb":1.07597,"BRRY_imb":-0.091985},"BRRY":{"HPPO_r10":0.148188,"HPPO_r30":0.161643,"HPPO_r60":0.063191,"HPPO_r120":0.006675,"HPPO_r240":0.004145,"HPPO_f30":0.016524,"HPPO_f120":0.610197,"HPPO_um":-0.632797,"BNKY_r10":0.019872,"BNKY_r30":-0.120629,"BNKY_r60":-0.144329,"BNKY_r120":0.051844,"BNKY_r240":-0.010599,"BNKY_f30":0.047657,"BNKY_f120":-0.117089,"BNKY_um":0.404803,"CRRT_r10":-0.046751,"CRRT_r30":-0.075814,"CRRT_r60":-0.048687,"CRRT_r120":-0.032637,"CRRT_r240":0.024544,"CRRT_f30":0.032604,"CRRT_f120":-0.194307,"CRRT_um":0.181159,"BRRY_r10":-0.045186,"BRRY_r30":0.014804,"BRRY_r60":0.004724,"BRRY_r120":-0.038419,"BRRY_r240":0.001945,"BRRY_f30":-0.256977,"BRRY_f120":0.586894,"BRRY_um":-0.342071,"gapB":0.501307,"HPPO_imb":0.865987,"BNKY_imb":-0.28344,"CRRT_imb":-0.212167,"BRRY_imb":0.425433}},"clip":{"HPPO_r10":7.5595,"HPPO_r30":14.3311,"HPPO_r60":23.3574,"HPPO_r120":37.9004,"HPPO_r240":59.3508,"HPPO_f30":2.2881,"HPPO_f120":6.0988,"HPPO_um":3.8007,"BNKY_r10":3.9462,"BNKY_r30":6.1183,"BNKY_r60":9.4213,"BNKY_r120":14.9043,"BNKY_r240":22.6209,"BNKY_f30":4.0898,"BNKY_f120":11.6924,"BNKY_um":2.6956,"CRRT_r10":5.28,"CRRT_r30":7.1627,"CRRT_r60":10.0702,"CRRT_r120":15.3222,"CRRT_r240":23.0849,"CRRT_f30":4.153,"CRRT_f120":12.4553,"CRRT_um":3.7647,"BRRY_r10":12.0227,"BRRY_r30":15.2076,"BRRY_r60":19.6124,"BRRY_r120":26.6568,"BRRY_r240":35.8208,"BRRY_f30":1.9318,"BRRY_f120":6.2105,"BRRY_um":7.7141,"gapB":20.2242,"HPPO_imb":1.4528,"BNKY_imb":1.4745,"CRRT_imb":1.4913,"BRRY_imb":1.3135}}"""
# END COEF
MODEL = json.loads(COEF_JSON)

# --- news reading (rules; switched on from the GUI with news_trade) ----------------------------
# Typical 60 s move per symbol (cents, research/evalx.py SIGMA60): a news jump is sized in these.
SIGMA60 = {"HPPO": 12.6, "BNKY": 4.9, "CRRT": 4.9, "BRRY": 8.2}
TICKERS = re.compile(r"\b(HPPO|BNKY|BRRY|CRRT)\b")
NEWS_UP = ("beat", "beats", "record", "surge", "surges", "soar", "soars", "jump", "jumps", "upgrade",
           "upgraded", "raises", "raised", "approval", "approved", "wins", "won", "strong", "growth",
           "profit", "rally", "rallies", "outperform", "exceeds", "boost", "boosts", "gain", "gains",
           "buyback", "breakthrough", "awarded", "tops", "exceeded", "surpasses", "upgrades", "soared",
           "surged", "jumped", "beat")
NEWS_DOWN = ("miss", "misses", "missed", "recall", "recalls", "lawsuit", "sued", "downgrade",
             "downgraded", "cuts", "cut", "loss", "losses", "fraud", "investigation", "probe",
             "resigns", "delay", "delayed", "warning", "warns", "plunge", "plunges", "fall", "falls",
             "drop", "drops", "fine", "fined", "bankruptcy", "default", "weak", "slump", "halt",
             # R1 lesson (CRRT, 14:40: "CEO was caught diverting company funds ... restate two quarters")
             "caught", "diverting", "diverted", "divert", "embezzlement", "embezzled", "misconduct",
             "scandal", "restate", "restates", "restatement", "accused", "indicted", "arrested",
             "charged", "subpoena", "ousted", "fired", "resign", "resigned", "breach", "hacked",
             "outage", "shortfall", "writedown", "impairment", "insolvency", "suspended", "suspends",
             "layoffs", "lowers", "lowered", "slashes", "plummets", "tumbles", "misled", "falsified")
NEGATIONS = ("no", "not", "denies", "denied", "without", "despite")
WORD = re.compile(r"[a-z]+")


def read_news(headline: str, body: str = "") -> dict:
    """{symbol: (direction +1/-1, confidence 0..1)} for a confident read; {} when unclear.
    One ticker named, keywords on one side only, no negation next to them. News on BNKY or BRRY
    also moves HPPO (HPPO = BNKY + 2 BRRY); news naming HPPO alone or several stocks is unclear."""
    text = f"{headline} {body}"
    named = set(TICKERS.findall(text))
    if len(named) != 1 or "HPPO" in named:
        return {}
    words = WORD.findall(text.lower())
    up = down = 0
    for i, w in enumerate(words):
        hit = 1 if w in NEWS_UP else -1 if w in NEWS_DOWN else 0
        if hit and any(n in words[max(0, i - 3):i] for n in NEGATIONS):
            return {}  # "not a recall", "denies fraud": leave it to a human
        up += hit > 0
        down += hit < 0
    if up and down or not (up or down):
        return {}
    d = 1 if up else -1
    conf = min(1.0, 0.5 + 0.25 * (up + down))
    (sym,) = named
    out = {sym: (d, conf)}
    if sym in ("BNKY", "BRRY"):
        out["HPPO"] = (d, conf)
    return out


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


class Alpha(Strategy):
    MODEL_PARAMS = (
        ParamSpec("make_edge", ParamKind.FLOAT, 1.5, 0.0, 50.0),  # cents from fair to quote (v10: was 2.5)
        ParamSpec("take_edge", ParamKind.FLOAT, 6.0, 0.0, 50.0),  # cents beyond the fee to take
        ParamSpec("size_scale", ParamKind.FLOAT, 1.5, 0.0, 3.0),  # scales clip and quote sizes
        ParamSpec("make_on", ParamKind.BOOL, 1, 0, 1),
        ParamSpec("take_on", ParamKind.BOOL, 1, 0, 1),
        ParamSpec("news_pause", ParamKind.INT, 10, 0, 240),  # batches without quotes after news
        ParamSpec("news_trade", ParamKind.BOOL, 0, 0, 1),  # trade a confident read of the news
        ParamSpec("news_move", ParamKind.FLOAT, 2.0, 0.0, 10.0),  # assumed jump, x typical 60 s move
        ParamSpec("news_hold", ParamKind.INT, 60, 0, 600),  # batches the news target stays in fair
        ParamSpec("kill_floor", ParamKind.INT, 2000, 0, 1000000),  # $ realized loss; 0 = off (see RUNBOOK)
        # v4, ported from the PQT strategy and checked by nested CV (ANALYSIS.md appendix D)
        ParamSpec("enabled", ParamKind.BOOL, 1, 0, 1),  # 0: cancel everything and stop
        ParamSpec("calib_on", ParamKind.BOOL, 1, 0, 1),  # scale the model by its live calibration
        ParamSpec("expo_lean", ParamKind.FLOAT, 3.0, 0.0, 20.0),  # c of lean against net exposure at the cap
        ParamSpec("arb_on", ParamKind.BOOL, 0, 0, 1),  # ETF vs basket arbitrage (off: legs rarely all fill live)
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
    # v11 (after R1): news on a symbol stops its quoting for NEWS_SYM_PAUSE batches, whatever the read
    # (R1: CRRT repriced 230c over ~90 s after a fraud headline while we kept quoting it). With
    # news_trade on and a confident read, the strategy also trades the news: it moves the position to
    # direction x conf x CAP with IOC orders up to NEWS_SLIP ticks through the touch (closing a
    # position on the wrong side first), for NEWS_BET_HOLD batches. Scoring credits fills after the
    # headline with the jump, and prices took over a minute to catch up in R1.
    NEWS_SYM_PAUSE = 120  # any news naming the symbol (R1's routine items moved nothing)
    NEWS_DIR_PAUSE = 360  # a confident read: R1's CRRT fraud item was still repricing at 120 s (-232c)
    NEWS_BET_HOLD = 360
    NEWS_BET_FRAC = 1.0  # of the exchange position limit (exposure still checked)
    NEWS_SLIP = {"HPPO": 30, "BNKY": 15, "CRRT": 15, "BRRY": 25}  # R1: CRRT moved -232c, so paying through is cheap
    NEWS_BURST = 4  # IOC orders per batch on a news bet (max order qty each): be filled before the reprice
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
    # ETF-consistent fair values (research, 0 = off): HPPO's true value is exactly BNKY + 2 BRRY, so
    # the three fairs are projected toward agreeing, each moving in proportion to its variance.
    ETF_PROJECT = 0.0
    # Volatility-scaled quoting edge (research, 0 = off): edge x (live 5 s move sd / historical)^gamma.
    VOL_GAMMA = 0.0
    VOL_REF = {"HPPO": 2.157, "BNKY": 0.968, "CRRT": 1.068, "BRRY": 2.317}  # median sd of 10-batch mid moves
    VOL_HALFLIFE = 1200
    # Volatility guard (0 = off): once a symbol's live 5 s move sd exceeds VOL_GUARD x its historical
    # median, edge x g and size / g with g = sd / (VOL_GUARD x median). Meant to act only outside the
    # historical range (session maxima: median ~1.1x, highest 1.7-2.7x).
    VOL_GUARD = 0.0
    # Auto-throttle (0 = off): every minute from minute THROTTLE_FROM, our fills' markout over the last
    # 10 minutes (vs the external mid 60 s later). Negative at two checks running: level +1, at most 2
    # (1: size x 0.5, no taking; 2: also edge x 1.6). Positive at three checks running: level -1.
    THROTTLE_FROM = 0
    # Size learner (research, off): per-symbol size factor starting at SIZE_START; every minute from
    # batch SIZE_FROM, x SIZE_UP if the symbol's 10-minute own markout > SIZE_UP_AT c/share, x SIZE_DOWN
    # if < 0, kept within [SIZE_MIN, SIZE_MAX]; needs SIZE_MIN_FILLS quote sizes of marked shares.
    SIZE_LEARN = False
    SIZE_START, SIZE_MIN, SIZE_MAX = 0.5, 0.33, 1.0
    SIZE_UP, SIZE_DOWN, SIZE_UP_AT = 1.2, 0.7, 0.5
    SIZE_FROM, SIZE_MIN_FILLS = 360, 5
    # Never quote more than MAX_IMPROVE ticks better than the external best price on our side
    # (None = no limit; 0 = only join or sit behind). Stops a quote reaching across a narrow spread,
    # where it is picked off first and can execute as a take when the book moves before our slot.
    MAX_IMPROVE = 2  # nested CV: every fold picked 2 over 3-6 and no limit (research/nested_improve.py)
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
        self.news_target = {}  # symbol -> (target price, direction, last batch) from a news read
        self.news_sym_until = {s: 0 for s in SYMS}  # no quoting on a symbol named in news until then
        self.news_bet = {}  # symbol -> (direction, confidence, last batch)
        self.mid10 = {s: deque(maxlen=11) for s in SYMS}
        self.var10 = {s: None for s in SYMS}
        self.n10 = {s: 0 for s in SYMS}
        self.own_fills = deque()  # (batch, symbol, signed qty, price) waiting 120 batches
        self.own_marked = deque()  # (fill batch, symbol, markout cents x shares, shares)
        self.size_k = {s: self.SIZE_START for s in SYMS}
        self.thr_level, self.thr_neg, self.thr_pos = 0, 0, 0
        self.last_qty = {}  # (symbol, side) -> last real external size at the touch
        self.gap_rebase = set()  # symbols whose next price is across a gap (halt end, reconnect)
        self.seen = 0  # batches seen by this program (a late start has no history yet)
        self.horizon = int(m.get("H", 120))
        self.adj_hist = {s: deque(maxlen=self.horizon + 1) for s in SYMS}  # (adj, micro) per batch
        self.cal = {s: [0.0, 0.0, 0.0] for s in SYMS}  # decayed sums: p*y, p*p, count
        self.cal_decay = 0.5 ** (1.0 / self.CALIB_HALFLIFE)
        self.k = {s: 1.0 for s in SYMS}
        self.kill_armed = False  # set in the first batch, once realized PnL is known

    def on_fill(self, fill) -> None:
        self.own_fills.append((fill.batch, fill.symbol, fill.qty if fill.side == "BUY" else -fill.qty, fill.price))
        if fill.liquidity == "TAKER":
            self.own_taker[fill.symbol] += fill.qty if fill.side == "BUY" else -fill.qty

    def on_news(self, item) -> None:
        named = set(TICKERS.findall(f"{item.headline} {item.body}"))
        if named & {"BNKY", "BRRY"}:
            named.add("HPPO")  # the ETF moves with its legs
        for s in named:
            self.news_sym_until[s] = max(self.news_sym_until[s], item.batch + self.NEWS_SYM_PAUSE)
        if named:
            self.log.info("news names %s: no quotes on them for %d batches", sorted(named), self.NEWS_SYM_PAUSE)
        if self.param("news_trade"):
            read = read_news(item.headline, item.body)
            for s, (d, conf) in read.items():
                self.news_bet[s] = (d, conf, item.batch + self.NEWS_BET_HOLD)
                self.news_sym_until[s] = max(self.news_sym_until[s], item.batch + self.NEWS_DIR_PAUSE)
            if read:
                self.log.warning("NEWS TRADE %s: %s", item.headline, {s: ("UP" if v[0] > 0 else "DOWN") for s, v in read.items()})
            move = float(self.param("news_move"))
            for s, (d, conf) in read.items():
                if s not in self.ext:
                    continue
                ref = self.news_ref.get(s, self.ext[s].micro) if self.rebase_at is not None else self.ext[s].micro
                j = d * conf * move * (SIGMA60[s] if s != "HPPO" else
                                       SIGMA60["BNKY" if "BNKY" in read else "BRRY"] * (1 if "BNKY" in read else 2))
                self.news_target[s] = (ref + j, d, item.batch + int(self.param("news_hold")))
            if read:
                self.log.info("news read %s: %s", item.headline, {s: v[0] for s, v in read.items()})
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

    def arm_kill_switch(self, state) -> None:
        """Set the realized-PnL floor, unless we are already below it (a restart after a loss must
        not block the whole team at once: a human decides)."""
        self.kill_armed = True
        floor = int(self.param("kill_floor"))
        realized = state.portfolio.realized_pnl_mills
        if floor <= 0:
            self.set_kill_switch(None)
        elif realized > -floor * 1000:
            self.set_kill_switch(-floor * 1000)
        else:
            self.log.warning("kill switch NOT set: realized PnL $%.0f is already below the -$%d floor; "
                             "set it in the GUI if wanted", realized / 1000, floor)

    def on_model_params(self, changed) -> None:
        if "kill_floor" in changed and self.state is not None:
            self.arm_kill_switch(self.state)  # a new floor from the GUI takes effect at once

    def on_reconnect(self) -> None:
        # resting orders are still there and a news pause stays in force. Batches (and possibly news
        # or a halt) were missed: rebase the price history across the gap and cool down every symbol.
        self.gap_rebase.update(SYMS)
        b = self.state.batch if self.state is not None else 0
        for s in SYMS:
            self.cool_until[s] = max(self.cool_until[s], b + HALT_COOLDOWN)

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
                    bid, bq, gb = bid - 1, self.last_qty.get((s, "bid"), 100), True
            ma = sum(o.remaining for o in self.open_orders(s) if o.side == "SELL" and o.price == ask)
            if ma:
                if aq > ma:
                    aq -= ma
                elif prev is not None and prev.ask > ask:
                    ask, aq, ga = prev.ask, prev.aq, True
                else:
                    ask, aq, ga = ask + 1, self.last_qty.get((s, "ask"), 100), True
            if bid >= ask:
                return prev
            if not gb:
                self.last_qty[(s, "bid")] = bq
            if not ga:
                self.last_qty[(s, "ask")] = aq
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
                    self.gap_rebase.add(s)
                e = self.external(s, top)
                if e is not None:
                    self.ext[s] = e
            if s in self.ext:
                self.u_hist[s].append(self.ext[s].micro)
                if s in self.gap_rebase and top.two_sided and not top.halted:
                    self.gap_rebase.discard(s)
                    h = self.u_hist[s]
                    if len(h) >= 2:  # shift the history before the gap so the jump reads as no move
                        jump = h[-1] - h[-2]
                        self.u_hist[s] = deque([u + jump for u in list(h)[:-1]] + [h[-1]], maxlen=h.maxlen)
                    self.adj_hist[s].clear()  # calibration samples across the gap are not the model's error
                h10 = self.mid10[s]
                h10.append(self.ext[s].mid)
                if len(h10) == 11:
                    d2 = (h10[-1] - h10[0]) ** 2
                    n = self.n10[s] = self.n10[s] + 1
                    v = self.var10[s] or 0.0
                    if n <= 240:  # plain mean while warming up, then exponential
                        self.var10[s] = v + (d2 - v) / n
                    else:
                        self.var10[s] = v + (1 - 0.5 ** (1 / self.VOL_HALFLIFE)) * (d2 - v)
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
            e = self.ext[s]
            x[f"{s}_imb"] = (e.bq - e.aq) / (e.bq + e.aq)  # book imbalance at the (external) touch
        x["gapB"] = (self.ext["HPPO"].micro - self.ext["BNKY"].micro) / 2 - self.ext["BRRY"].micro
        for k, lim in (self.model.get("clip") or {}).items():  # winsorise at training-set levels
            if k in x:
                x[k] = max(-lim, min(lim, x[k]))
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
            t = self.news_target.get(s)
            if t is not None:  # until the market reaches the news target, fair sits at least there
                target, d, until = t
                if batch is not None and batch > until:
                    del self.news_target[s]
                elif d > 0:
                    self.fair[s] = max(self.fair[s], target)
                else:
                    self.fair[s] = min(self.fair[s], target)

    def project_etf(self) -> None:
        """Move HPPO, BNKY and BRRY fairs toward HPPO = BNKY + 2 BRRY (weighted least squares)."""
        f = self.fair
        r = f["HPPO"] - f["BNKY"] - 2 * f["BRRY"]
        v = {s: SIGMA60[s] ** 2 for s in ("HPPO", "BNKY", "BRRY")}
        lam = self.ETF_PROJECT * r / (v["HPPO"] + v["BNKY"] + 4 * v["BRRY"])
        f["HPPO"] -= v["HPPO"] * lam
        f["BNKY"] += v["BNKY"] * lam
        f["BRRY"] += 2 * v["BRRY"] * lam

    def vol_guard(self, s) -> float:
        if not self.VOL_GUARD or self.n10[s] < 120:
            return 1.0
        return max(1.0, math.sqrt(self.var10[s]) / (self.VOL_GUARD * self.VOL_REF[s]))

    def throttle(self, batch) -> None:
        """Mark our fills 60 s on; once a minute move the throttle level on the 10-minute markout."""
        f, m = self.own_fills, self.own_marked
        while f and f[0][0] <= batch - self.horizon:
            b, s, q, px = f.popleft()
            if s in self.ext:
                m.append((b, s, q * (self.ext[s].mid - px), abs(q)))
        while m and m[0][0] < batch - self.horizon - 1200:
            m.popleft()
        if self.SIZE_LEARN and self.seen >= self.SIZE_FROM and self.seen % 120 == 0:
            self.learn_size(m)
        if not self.THROTTLE_FROM or self.seen < self.THROTTLE_FROM * 120 or self.seen % 120:
            return
        mk = sum(x[2] for x in m)
        self.thr_neg, self.thr_pos = (self.thr_neg + 1, 0) if mk < 0 else (0, self.thr_pos + 1)
        if self.thr_neg >= 2 and self.thr_level < 2:
            self.thr_level, self.thr_neg = self.thr_level + 1, 0
            self.log.info("throttle up to %d: 10-min markout $%.0f", self.thr_level, mk / 100)
        elif self.thr_pos >= 3 and self.thr_level > 0:
            self.thr_level, self.thr_pos = self.thr_level - 1, 0
            self.log.info("throttle down to %d: 10-min markout $%.0f", self.thr_level, mk / 100)

    def learn_size(self, marked) -> None:
        """Per symbol: grow the size factor while our last 10 minutes of fills mark out well, shrink
        it when they mark out badly (vs the mid 60 s later)."""
        agg = {}
        for _, s, mk, q in marked:
            a = agg.setdefault(s, [0.0, 0])
            a[0] += mk
            a[1] += q
        for s in SYMS:
            mk, q = agg.get(s, (0.0, 0))
            if q < self.SIZE_MIN_FILLS * self.QUOTE[s]:
                continue
            c = mk / q
            k = self.size_k[s]
            if c > self.SIZE_UP_AT:
                k *= self.SIZE_UP
            elif c < 0:
                k *= self.SIZE_DOWN
            self.size_k[s] = max(self.SIZE_MIN, min(self.SIZE_MAX, k))

    def vol_factor(self, s) -> float:
        if not self.VOL_GAMMA or self.var10[s] is None:
            return 1.0
        z = math.sqrt(self.var10[s]) / self.VOL_REF[s]
        return max(0.5, min(3.0, z)) ** self.VOL_GAMMA

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

    def room(self, s, side, lw, sw, cap=None) -> int:
        """Shares we may add on `side` of `s` within the soft cap, the hard limit and exposure."""
        cap = min(self.CAP[s] if cap is None else cap, self.symbol_info[s].position_limit)
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
        if not self.kill_armed:
            self.arm_kill_switch(state)
        if self.kill_switch.engaged or state.blocked:  # the exchange refuses everything: stop sending
            if not getattr(self, "said_blocked", False):
                self.said_blocked = True
                self.log.warning("trading stopped: %s", "kill switch engaged" if self.kill_switch.engaged else "team blocked")
            return
        self.said_blocked = False
        if not ready or self.seen <= WARMUP or state.final:
            return
        self.compute_fair(state.batch)
        if self.ETF_PROJECT:
            self.project_etf()
        quoting = bool(self.param("make_on")) and state.batch >= self.pause_until
        self.throttle(state.batch)
        take_on = bool(self.param("take_on")) and self.thr_level == 0
        scale = float(self.param("size_scale")) * (0.5 if self.thr_level else 1.0)
        lean = float(self.param("expo_lean"))
        lw, sw = self.worst_case()
        arb = self.trade_arb(state, lw, sw) if self.param("arb_on") else set()
        if self.param("recycle_on") and not arb:
            self.recycle(state)
        E = sum(WEIGHT[s] * self.position(s) for s in SYMS)  # net exposure of positions
        self.trade_news(state, lw, sw)
        for s in SYMS:
            if s in arb:  # its quotes were just pulled for the arbitrage
                continue
            top = state.top[s]
            if self.param(f"pause_{s}"):
                if self.open_orders(s):
                    self.cancel_all(s)
                continue
            if state.batch < self.news_sym_until[s]:  # named in recent news: stand aside
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
                take_edge = TAKER_FEE + float(self.param("take_edge")) * mult * self.vol_guard(s)
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
            g = self.vol_guard(s)
            e = float(self.param("make_edge")) * mult * self.vol_factor(s) * g * (1.6 if self.thr_level >= 2 else 1.0)
            bid = min(math.floor(centre - e), x.ask - 1)  # never cross the (external) book
            ask = max(math.ceil(centre + e), x.bid + 1)
            cap_ticks = self.MAX_IMPROVE.get(s) if isinstance(self.MAX_IMPROVE, dict) else self.MAX_IMPROVE
            if cap_ticks is not None:
                bid = min(bid, x.bid + cap_ticks)
                ask = max(ask, x.ask - cap_ticks)
            if bid >= ask:
                continue
            size = max(1, int(self.QUOTE[s] * scale / g * (self.size_k[s] if self.SIZE_LEARN else 1.0)))
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

    def trade_news(self, state: BatchState, lw, sw) -> None:
        """Move each symbol with a live news bet toward direction x conf x CAP: one IOC a batch, up to
        NEWS_SLIP ticks through the external touch. A position on the wrong side is closed first (the
        same order keeps going past zero). Our own quotes on it are cancelled first (no self-trade)."""
        for s in list(self.news_bet):
            d, conf, until = self.news_bet[s]
            if state.batch > until:
                del self.news_bet[s]
                continue
            x, top = self.ext.get(s), state.top[s]
            if x is None or top.halted or not top.two_sided or self.param(f"pause_{s}"):
                continue
            lim = self.symbol_info[s].position_limit
            target = int(d * conf * self.NEWS_BET_FRAC * lim)
            need = target - self.position(s)
            if need * d <= 0:
                continue
            if self.open_orders(s):
                self.cancel_all(s)
            maxq = self.symbol_info[s].max_order_qty
            cap = int(self.NEWS_BET_FRAC * lim)
            if d > 0:
                q = min(need, self.room(s, "BUY", lw, sw, cap=cap))
                for _ in range(self.NEWS_BURST):
                    c = min(q, maxq)
                    if c <= 0:
                        break
                    self.buy(s, c, x.ask + self.NEWS_SLIP[s], ioc=True)
                    lw[s] += c
                    q -= c
            else:
                q = min(-need, self.room(s, "SELL", lw, sw, cap=cap))
                for _ in range(self.NEWS_BURST):
                    c = min(q, maxq)
                    if c <= 0:
                        break
                    self.sell(s, c, x.bid - self.NEWS_SLIP[s], ioc=True)
                    sw[s] -= c
                    q -= c

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
