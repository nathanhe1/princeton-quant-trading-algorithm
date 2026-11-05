# Prompts for Claude Code in VS Code

Paste them in order, one per chat (or keep one chat going for 1–4). Each one is self-contained.
Rules that apply to all of them are in prompt 0: paste it at the start of every new chat.

---

## 0. Context (start of every new chat)

```
You are my senior quant researcher and developer for the PQT trading competition. Read
research/NOTES.md, STRATEGY_GUIDE.md sections 1, 6, 7 and 11, and strategies/alpha.py before doing
anything. Ground rules:
- The score is the 60-second markout of every fill against the hidden true value, NOT P&L. Judge
  every change with research/final_cv.py (3-fold out-of-sample proxy markout), never with the
  "estimated PnL" that pqt backtest prints.
- Use the venv: . .venv/bin/activate (Python 3.12). Research code goes in research/, the strategy
  stays one self-contained file. Regenerate coefficients with research/fit.py + research/bake.py.
- Prefer per-share edge and worst-session results over the mean: the backtest overstates our
  volume. Report results as a before/after table. Do not tune on the sessions you evaluate on.
- Before you change strategies/alpha.py, copy the current version to strategies/archive/ with a
  version number, so we can always go back.
```

## 1. Stress-test the fill model (most important)

```
The backtest fills our resting quotes whenever a recorded trade prints strictly through them
(pqt_client/backtest_v2.py, _Replay.rest_fills), and no other team competes. Live, 60-70 teams
will. Build research/stress.py that monkeypatches _Replay.rest_fills with knobs:
  through=N     a trade must print at least N ticks through our price
  share=p       only a fraction p of each trade's size is available to us (seeded random rounding)
  adverse=True  also fill us on trades AT our price, but only when the mid 120 batches later moves
                against us (a deliberately pessimistic pick-off model)
Run final_cv-style 3-fold evaluation of alpha.py under: base; through=2; share=0.3; share=0.1;
share=0.3+through=2; adverse; all combined. Show $/session, cents/share by symbol and maker/taker,
worst session. Then re-tune make_edge, size_scale, EDGE_MULT and SKEW per symbol to maximise the
WORST scenario we consider realistic (share=0.3+through=2+adverse), not the base case. Update the
defaults in alpha.py if they improve that scenario without hurting base by more than 10%.
```

## 2. A better fair value

```
Improve the fair-value model in research/fit.py, keeping it fast enough to compute every 500 ms in
pure Python. Try, one at a time, keeping only what improves out-of-sample R^2 under leave-sessions-
out CV AND improves final_cv.py markouts:
- target horizon 120 vs 180 vs 240 batches (the mid keeps drifting after 60 s; V may lead further)
- flow split by trade size (big prints vs small) and by count; flow decay (EWMA) instead of windows
- volatility-normalised features and a per-session volatility regime scaling of edges
- the ETF basis (HPPO vs BNKY+2 BRRY) as a feature for every symbol, not just BRRY
- order-book imbalance at the touch (bid_qty vs ask_qty) and spread
- separate coefficients by time in session (first 5 min vs rest)
- small nonlinearity: sign*sqrt or clipping of extreme features
Report a table: feature set, OOS R^2 per symbol, final_cv $/session, worst session. Bake the winner.
```

## 3. Smarter quoting and capacity

```
Our capacity is limited by position limits and |net exposure| <= 600 (HPPO weight 1, BNKY 0.15,
BRRY 0.425, CRRT -0.29). Improve how alpha.py uses it, testing each idea with final_cv.py AND the
pessimistic scenario from research/stress.py:
- quote only the side the signal favours when |fair - micro| is large; widen the other side
- size quotes by expected edge (bigger when fair is far from our price)
- two price levels per side instead of one
- inventory skew that targets the model's predicted direction instead of zero
- use conversion (create/redeem, $0.02 per HPPO share, does not change exposure) to free HPPO and
  BRRY position capacity when we are stuck at a limit
- allocate exposure across symbols by edge per unit of exposure (CRRT's negative weight is a hedge)
Keep the message budget (20/batch, team-wide) and log how close we get to it.
```

## 4. ETF arbitrage leg

```
Measure in the data how often HPPO's touch and the basket's touch (BNKY + 2 BRRY) cross by more
than fees, how long it lasts and how much size is there. Add an arbitrage leg to alpha.py only if it
adds markout under the stress scenarios: legs that fill score their price difference even without
converting, but leg risk is real live (an IOC can fill partly). Prefer passive legs on BRRY (wide
spread) hedged with aggressive HPPO/BNKY. Show the result with and without it.
```

## 5. News (the data has none; live will)

```
The live market publishes free-text news (headline, body) that moves true values. Fills made after
a news item and before the market reprices are credited (STRATEGY_GUIDE section 7). Right now
alpha.py only pulls quotes for news_pause batches. Build a news module:
1. a rules-based parser that finds which symbols an item names (including the ETF: news on BNKY or
   BRRY moves HPPO) and the direction and rough size from keywords;
2. on a confident read, take liquidity in that direction within the next batch, sized by confidence,
   and shift fair value for the affected symbols until the market has moved;
3. on an unclear read, pull quotes on the affected symbols only (not all four).
Write unit tests with made-up news items covering positive, negative, ETF-propagated and ambiguous
cases. Add a model parameter to switch news trading off from the GUI panel. I will paste real
sandbox news items here when I see them so we can tune the parser.
```

## 6. Sandbox validation (needs our key in PQT_API_KEY)

```
Help me validate alpha.py on the sandbox. First run pqt check --server
https://sandbox.princeton-quant.com and fix anything it flags. Then run alpha.py in test mode for
10 minutes, logging to a file, and check: no exceptions; rejects by code (expect only a few
UNKNOWN_ORDER); messages per batch stay under 20; on_batch time per batch; that external() removes
our own quotes from the book correctly (log the raw and external touch when we are at the best
price); that news and halts trigger the pause; that the model-parameter panel shows our params.
Fix what you find and summarise what the sandbox could not tell us.
```

## 7. Competition-day hardening and runbook

```
Review alpha.py for anything that could cost us a session: an exception path that stops trading,
state that breaks after a reconnect or exchange restart (all orders cancelled with reason SYSTEM),
halts on HPPO (conversion refused), the kill switch level, behaviour when a book is one-sided,
and what happens if we start late (features need history). Add tests in research/ that replay a
session with injected halts, a disconnect, a restart and late start. Then write RUNBOOK.md: the
exact commands for R1 and R2, which key runs where, which model parameters to change in the GUI
and when (e.g. widen make_edge if markouts look bad), and what to do if something breaks. R2 counts
double, so include anything we learn from R1's report.
```

## 8. Final review

```
/code-review high strategies/alpha.py
```
Then: `Act on the confirmed findings, re-run research/final_cv.py and research/stress.py, and show
before/after.`
