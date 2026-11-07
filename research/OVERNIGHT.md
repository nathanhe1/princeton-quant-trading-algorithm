# Overnight work, 2026-11-07: v8 → v10

**Conclusion:** `strategies/alpha.py` is v10. It makes money in every scenario the backtest can
model, worst sessions included. Most of its edge now comes from how it quotes, not from the
forecast, so it no longer depends on the model being right. All numbers are 3-fold out-of-sample on
the 24 historical sessions, $/session (worst session in brackets).

| case | v3.1 (start of this work) | v8 (start of tonight) | **v10 (now)** |
|---|---|---|---|
| base (backtest fill rule) | 5,602 (3,382) | 6,382 (3,912) | **9,315 (6,017)** |
| realistic stress (30% share, 2 ticks through, pick-offs) | 1,540 (473) | 1,661 (471) | **3,202 (1,342)** |
| stress without the look-ahead pick-off rule | 1,713 (665) | 1,765 (688) | **3,182 (1,423)** |
| 10% share | 1,365 (533) | 1,411 (693) | **2,070 (1,106)** |
| harshest (10% + 2 ticks + pick-offs) | 789 (−4) | 769 (88) | **1,413 (389)** |
| forecast wrong-signed, stress | −1,543 (−2,858) | 69 (−567) | **1,798 (634)** |
| forecast overconfident ×2, stress | 741 (−259) | 1,339 (430) | **2,840 (1,448)** |
| no forecast at all, stress | 186 (−236) | 402 (−130) | **2,158 (899)** |

v10 vs v9 under stress: +$491 ± 25 per session, better in 24/24 sessions on each of three seeds.

## What changed

1. **Quote cap (v9):** a quote never sits more than 2 ticks better than the best price on its side.
   - **Why:** before, a strong forecast could pull a bid up to the ask minus one tick, and those
     fills were poor value.
   - **Evidence:** nested CV picked 2 on every fold. Maker ¢/share rose on every symbol under
     every fill model.
2. **Narrower edge (v10):** make_edge 2.5 → 1.5. Once the cap exists, a tight quote cannot reach
   across the spread, so narrower edges add volume without the bad fills.
   - **Nested CV picked 0.5** (the end of the grid).
   - **I kept 1.5 because of the sandbox recording:** narrow edges lost more there (a market
     outside the historical range).
   - **It is a GUI lever:** the RUNBOOK widens it if the live markout turns negative.
3. **Kill switch −$2,000** (was $20,000, which never fires). The deepest intra-session realized PnL
   in any historical scenario is −$411; the v8 live disaster on the sandbox hit −$1,954 by minute
   12. It is armed in the first batch, never when we are already below the floor (a restart after a
   loss must not lock the team out). Panel changes apply at once.
4. **Fixes from two code reviews,** with tests:
   - halt reopenings and reconnects no longer read the price gap as momentum (a 30¢ halt gap caused
     3–11 mistaken takes before);
   - a reconnect cools all symbols down;
   - the level guessed behind a lone quote uses the last real size;
   - the logger passes GUI parameter changes on to the strategy.

   Checks: `tests/test_alpha_live.py` (8) and `research/hardening.py` (28 replays: halts, a halt with
   a price gap, one-sided books, dropout, late start, news) all pass.

## Tested and rejected (`research/HANDOFF.md` has the numbers)

- **Live learning:** a markout guard, an automatic throttle and a per-symbol size learner. All three
  cost money; 45 min holds too few independent outcomes to learn more than one number per symbol
  (the forecast weight, which stays).
- **Settings:** per-symbol caps; narrower per-symbol edges (+$118, the same direction as a lower
  make_edge, held back for risk); larger quotes; larger inventory caps; skew; volatility-scaled
  edges and a volatility guard.
- **Features and arbitrage:** flow-by-trade-size features (better R², no better markout); ETF
  arbitrage (legs rarely all fill live, and below zero even in the backtest).

## The sandbox

- **What its market is like:** its participants and prices are placeholders (its own news says
  "nothing learned on the sandbox transfers"). Its 5 s moves are 2–4× the historical range on BNKY
  and CRRT.
- **What we did there:** a clean recording of one session showed our stress model predicts its
  losses (pick-offs of about 7¢/share). So it is used for mechanics and as a held-out sanity check,
  never for tuning.
- **Live results tonight:** see below.

## Live sandbox results tonight

(filled in below as sessions finish)

## Your decisions

1. **Push to GitHub?** Nothing is pushed. The auto-committer committed a 37 MB cache
   (`research/variants/raw_cache.pkl`) into local history; say whether to strip it first.
2. **Agree with the team, before R1:**
   - the parameter list is frozen (no change to `MODEL_PARAMS`, defaults included, during a
     session);
   - only `strategies/alpha.py` runs, through the logger command in RUNBOOK.md;
   - no manual resting orders while it runs;
   - the kill switch is −$2,000.
3. **After R1:** `research/r1_report.py <R1 log>` gives markout curves (compare with the official
   report: does the true value lead the mid?), our share of market volume, and each news item with
   how fast prices repriced. Those decide `news_trade`, size and the edge for R2.
