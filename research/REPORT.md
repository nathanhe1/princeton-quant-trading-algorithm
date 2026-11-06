# Research report: overfitting, stress fills and news (2026-11-05)

What changed this iteration, what was tested and rejected, and what is still open. Every dollar
figure is the **120-batch proxy markout per session, out of sample**: each session is scored with
coefficients fitted without it (3 folds of 8 sessions, as in `final_cv.py`). The proxy is the mid
120 batches (60 s) after each fill, not the hidden true value the competition scores against.

## Summary

| | before (v2) | after | verdict |
|---|---|---|---|
| Base backtest fill rule | $5,600 (worst $3,382) | $5,600 (worst $3,382) | unchanged |
| Realistic stress (30% of size, 2 ticks through, adverse) | $1,540 (worst $473) | $1,540 (worst $473) | unchanged; **about $1,000 after removing look-ahead** (section 2) |
| Synthetic news, market reprices instantly | $5,137 (worst $3,063) | **$5,387** (worst $3,115) | +$249 |
| Synthetic news, reprices over 10 s | $5,391 (worst $3,441) | **$5,455** (worst $3,492) | +$64 |
| Synthetic news, reprices over 30 s | $5,500 (worst $3,474) | $5,461 (worst $3,301) | −$39 |
| Realistic stress + news (10 s) | $1,644 (worst $673) | **$1,670** (worst $807) | +$26 |

1. **The fair-value model overfits only a little.** In-sample R² is about 0.02 above out-of-sample
   for every symbol. A better-regularised fit raised out-of-sample R² but **lost money under realistic
   stress**, so the coefficients are unchanged.
2. **The quoting knobs were not badly overfit.** Nested CV, where the choice never sees the
   evaluation sessions, picks `make_edge` 2.0 in all three folds. Out of sample that gains only $9
   under realistic stress, which is noise, so the defaults are unchanged.
3. **The stress test was not harsh enough.** Its "adverse" option made more money than the base
   case. Fixed: realistic-scenario markout fell from $1,755 to $1,540 per session. Even fixed, it
   leaks future prices into our inventory. Without that leak the realistic figure is **about
   $1,000 per session**, and the harshest scenario's worst session is −$4.
4. **News is now handled better:** jumps no longer pass for momentum, and the forecast is capped.
   This changes nothing without news and adds up to $249 per session with it. It loses a little
   when the market reprices slowly (see open issues).

Changes to `strategies/alpha.py` (previous version saved as `strategies/archive/alpha_v2.py`):
`FAIR_CAP`, `NEWS_REBASE` and `rebase()`. The coefficients and every default are the same as v2.

---

## 1. R², in sample vs out of sample

Target: mid 120 batches ahead minus the current microprice. "In sample" = fitted and scored on all
24 sessions; "out of sample" = 3-fold CV, the same folds as `final_cv.py`. `research/overfit.py`.

| model | HPPO IS / OOS | BNKY IS / OOS | CRRT IS / OOS | BRRY IS / OOS |
|---|---|---|---|---|
| **current**: 33 features, ridge 1e-3 on raw features | 0.143 / 0.118 | 0.104 / 0.078 | 0.178 / 0.158 | 0.309 / 0.286 |
| 33 features, scaled ridge, penalty by inner CV | 0.138 / 0.123 | 0.098 / 0.081 | 0.176 / 0.159 | 0.307 / 0.288 |
| own-symbol features only (8, + `gapB` for BRRY) | 0.121 / 0.111 | 0.078 / 0.072 | 0.111 / 0.103 | 0.258 / 0.250 |
| own + hand-picked cross terms (11–14) | 0.136 / 0.126 | 0.088 / 0.082 | 0.153 / 0.145 | 0.297 / 0.284 |

- The gap between in-sample and out-of-sample R² is 0.020–0.026: about 15–25% of the out-of-sample
  figure. That is modest. The features are few and sessions are long; most of the overfitting risk
  lies elsewhere.
- **Cross-symbol terms are real**, especially for CRRT: dropping them cuts its out-of-sample R² from
  0.158 to 0.103. HPPO flow and the ETF gap carry information.
- **Coefficient signs are not stable across folds.** Only 61–82% of the current model's
  coefficients keep the same sign in all three folds (the scaled ridge: 61–91%). The small ones are
  noise, but they are also small in effect.
- **BNKY is the most fragile.** A model fitted on 6 sessions can have negative R² on another
  6-session block. It is fine with 16 or more sessions.

**Tested and rejected: scaled ridge.** Features were scaled to unit RMS before the ridge, with the
penalty chosen per symbol by an inner 2-fold CV inside each fold's training sessions. Penalties
chosen: 0.3 (HPPO, BNKY), 0.1–0.3 (CRRT, BRRY). Out-of-sample R² rose by 0.001–0.005 and base
markout by $83 per session. Under realistic stress it **lost $56–58 per session in all three
seeds tried** (paired standard error about $30); it was better in only 8–10 of 24 sessions, and
the worst session fell from $473 to $258. Prompt 2's rule is to keep only what improves both, so
it was dropped. Higher R² does not guarantee better fills. `research/fit.py` keeps it behind
`--scaled`; the default is still the baked fit.

## 2. Stress test fix

`research/stress.py` had `adverse=True` fill a resting quote on a trade at our price whenever
the mid 120 batches later had moved *at all* against us. Most such fills still made money (the
mid moved less than our distance to it), so "adverse" alone scored **more** than base ($6,752 vs
$5,600). Now the fill counts only if the mid 120 batches later is *past our price*: a real
pick-off, always a losing fill.

alpha.py (identical to v2 without news), 24 out-of-sample sessions, `research/chart_data.py`:

| scenario | $/session | worst session | shares/session | maker ¢/share |
|---|---|---|---|---|
| base (backtest rule) | $5,600 | $3,382 | 287k | 1.93 |
| trade must print 2 ticks through | $2,871 | $1,442 | 142k | 1.96 |
| we get 30% of each trade's size | $2,865 | $1,640 | 136k | 2.02 |
| we get 10% | $1,364 | $533 | 56k | 2.11 |
| 30% + 2 ticks through | $1,710 | $665 | 74k | 2.13 |
| adverse at-price fills only | $6,049 | $3,498 | 352k | 1.69 |
| **realistic: 30% + 2 ticks + adverse** | **$1,540** | **$473** | 104k | 1.30 |
| harshest: 10% + 2 ticks + adverse | $788 | **−$4** | 44k | 1.25 |

(Before the fix, realistic was $1,755 and adverse alone $6,752.)

**Even fixed, "adverse" flatters the strategy, because it leaks the future.** The rule decides a
pick-off by looking at the mid 120 batches ahead. Every forced fill therefore tells the strategy
where the price is going, and the inventory skew then moves its quotes the right way. The fills
split as follows (`at price` = filled on a print exactly at our price):

| realistic scenario, per session | at-price fills | all other fills | total |
|---|---|---|---|
| without adverse (30% + 2 ticks) | $67 (1.50¢/sh) | $1,452 (2.17¢/sh) | $1,710 |
| with adverse | **−$693** (−2.49¢/sh, 28k sh) | $2,000 (2.74¢/sh) | $1,540 |

The pick-offs cost what they should, but the other fills gain $548 per session from information
no real market would give. **A conservative realistic estimate is about $1,000 per session**
($1,540 − $548). It is still positive, but the worst sessions would be near zero. Being picked off
does carry information in a real market too, so the truth lies between the two figures. The rule
cannot be made leak-free without making it not adverse; read "adverse" results as optimistic.

## 3. Strategy knobs: nested CV

`research/tune_nested.py`. Grid: `make_edge` {2, 2.5, 3, 3.5} × `size_scale` {1, 1.5} ×
`take_edge` {6, 10}. For each held-out fold, every setting is scored on the other 16 sessions. Each
of those sessions is scored with coefficients fitted on the remaining single fold (`only{k}.json`),
so neither the coefficients nor the choice see the held-out fold. Selection rule (from prompt 1):
the best realistic-stress mean among settings whose base mean is at least 90% of the default's.

| fold | picked | base: default → picked | realistic: default → picked | realistic worst |
|---|---|---|---|---|
| s01–s08 | make 2.0, size 1.5, take 6 | $5,039 → $5,183 | $1,281 → $1,301 | $473 → $440 |
| s09–s16 | make 2.0, size 1.5, take 6 | $6,001 → $6,069 | $1,779 → $1,776 | $1,125 → $1,107 |
| s17–s24 | make 2.0, size 1.5, take 6 | $5,761 → $5,793 | $1,561 → $1,571 | $905 → $906 |
| **mean** | | **+$82** | **+$9** | worse in 2 of 3 folds |

- The pick is stable: the same setting in every fold.
- On the 16 tuning sessions the pick looked +$31 better under realistic stress; on held-out sessions
  that shrank to +$9. That gap is the optimism of choosing on the data you score.
- **Kept `make_edge` at 2.5.** +$9 is below the seed noise, and a tighter quote is the riskier side
  live, where adverse selection will be worse than any replay.
- `size_scale` 1.5 and `take_edge` 6 were also confirmed.

## 4. News

The historical data has no news; the live market will. The score credits fills made after an item
and before the market reprices, and removes the jump of any item published after a fill (the ΔN
term, STRATEGY_GUIDE section 7).

**The problem.** v2 pulled quotes for `news_pause` (10) batches. After that, the news jump was still
inside every return feature (lags up to 240 batches) and flow feature, so the model read a one-off
repricing as momentum. It then quoted and took in the direction of the jump for up to 2 minutes.

**The fix in `alpha.py`.**
- `NEWS_REBASE`: on news, record each symbol's microprice. When the pause ends, `rebase()` shifts
  the pre-news price history by the move since the news and flattens the pause window. It also
  zeroes the flow traded during the pause. The features then measure only what happened after
  the repricing. If several items arrive close together, the reference is taken from before the
  first.
- `FAIR_CAP`: |fair − micro| is capped at about 4× each symbol's typical predicted 60 s move (HPPO
  20¢, BNKY 6.5¢, CRRT 8.5¢, BRRY 20¢). This is a fixed guard against outlier features (news, a bad
  print), not a tuned value. On the historical data it almost never binds: the no-news results are
  identical.

**How it was tested.** `research/evalx.py` adds synthetic news to the replay; this is a
sensitivity test with invented sizes and speeds:
- Each session gets 6 items, each on BNKY, BRRY or CRRT, with a jump of ±2× or ±4× that symbol's
  typical 60 s move (HPPO 12.6¢, BNKY 4.9¢, CRRT 4.9¢, BRRY 8.2¢).
- Jumps propagate through the ETF: BNKY → HPPO ×1, BRRY → HPPO ×2.
- The recorded book and trades reprice linearly over D batches, so stale quotes can be hit during
  the ramp.
- Items reach `on_news` exactly as live.
- Markouts apply the ΔN rule.

| repricing | v2 | + scaled ridge, cap, rebase | − rebase | − cap | pause 30 |
|---|---|---|---|---|---|
| instant (D = 0) | $5,137 | $5,462 | $5,303 | $5,440 | $5,386 |
| 10 s (D = 20) | $5,391 | $5,552 | $5,494 | $5,549 | $5,628 |
| 30 s (D = 60) | $5,500 | $5,556 | $5,588 | $5,555 | $5,508 |

(The ablation columns use the scaled-ridge coefficients and were run before those were rejected;
the final before/after in the summary uses the original coefficients.)

- The rebase is worth +$160 per session with instant repricing and +$58 at 10 s; it costs $32 at
  30 s. With slow repricing, part of the post-pause move is real continuation, and the rebase drops
  some of it.
- The cap adds $0–20: insurance, not alpha.
- A 30-batch pause is better at 10 s repricing and worse at the other two speeds, so `news_pause`
  stays at 10. It is a GUI knob: widen it if live repricing looks slow.

## 5. Charts

`research/charts/`, drawn by `research/charts.py` from CSVs cached by `research/chart_data.py`
(about 3 minutes to regenerate). The stress charts use the fixed adverse rule.

| file | shows |
|---|---|
| `01_stress_scenarios.png` | every session under each fill scenario, with mean and worst |
| `02_stress_by_symbol.png` | maker ¢/share and $/session by symbol × scenario |
| `03_stress_volume_vs_edge.png` | shares filled vs maker ¢/share per scenario |
| `04_v1_vs_current.png` | change per session from the earlier skew change (v1 → v2), base and realistic |
| `05_horizons.png` | held-out R² and markout for 120 / 180 / 240-batch targets (120 is best) |

## 6. Files

| file | what |
|---|---|
| `strategies/alpha.py` | + `FAIR_CAP`, `NEWS_REBASE`, `rebase()`; coefficients and defaults as in v2 |
| `strategies/archive/alpha_v2.py` | the version before this iteration |
| `research/overfit.py` | in-sample vs out-of-sample R², sign stability, regularised variants |
| `research/fit.py` | default = the baked fit; `--scaled [--lam S=v,...]` = the rejected scaled ridge |
| `research/stress.py` | adverse fills fixed to real pick-offs |
| `research/evalx.py` | parallel 3-fold evaluation: any strategy, raw/scaled coefficients, any stress scenario, synthetic news; `STRESS_SEED` env var |
| `research/iterate.py` | ablations (coefficients, cap, rebase, pause) without and with news |
| `research/tune_nested.py` | nested CV of `make_edge` / `size_scale` / `take_edge` |
| `research/final_compare.py` | the summary before/after table |
| `research/fold{k}_std.json`, `only{k}.json`, `only{k}_std.json`, `coef_all_std.json` | coefficient files for the above (`only{k}` = fitted on fold k alone) |
| `research/charts/data/*.csv` | every number in this report |

Reproduce (from `research/`, with `../.venv/bin/python`): `overfit.py`, `iterate.py`,
`tune_nested.py` (about 15 minutes), `final_compare.py`, `chart_data.py` then `charts.py`.

## 7. Open issues, in priority order

1. **Volume share and adverse selection.** Realistic stress keeps about 18–28% of the base
   dollars ($1,000–$1,540 of $5,600). Pick-offs cost about 2.5¢ per share, more than the whole
   maker edge. Live fill share and pick-off rate are the biggest unknowns. Check maker ¢/share in
   R1's report first, before anything else. If it is below about 1¢, widen `make_edge`.
2. **Rebase timing vs repricing speed.** The rebase loses a little when the market takes about 30 s
   to reprice. Once real news is seen on the sandbox, measure how long repricing takes and set
   `news_pause` to cover it.
3. **News trading (prompt 5)** is still not built: the strategy only steps aside and protects its
   features. Reading direction from headlines and taking liquidity before the market reprices is
   the credited alpha.
4. **Proxy vs true value.** Everything here uses the mid 60 s later. Earlier notes show the 240-batch
   markout above the 120-batch one, so the true value may lead the mid; the horizon choice is
   biased toward the proxy.
5. **ETF identity is unused for HPPO and BNKY.** Their forecasts are not forced to agree with
   1 BNKY + 2 BRRY (prompts 2 and 4).
6. **Stress-scenario randomness.** The size haircut uses seeded rounding that depends on the fill
   path, so differences under about $60 per session between configurations are not meaningful
   without several seeds.
