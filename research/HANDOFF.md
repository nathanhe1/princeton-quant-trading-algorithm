# Handoff: PQT strategy, where it stands and what to build next

You're picking up `strategies/alpha.py`, our market maker for the PQT competition. It quotes all
four symbols around a 60-second fair-value forecast. The full write-up of the last iteration is
[`research/REPORT.md`](REPORT.md); the task list is [`PROMPTS.md`](../PROMPTS.md) (prompts 0–8).
Read `research/NOTES.md` and STRATEGY_GUIDE.md sections 1, 6, 7 and 11 first.

## State in one paragraph

The forecast is a linear model (ridge, 33 features: price changes, signed trade flow, micro−mid,
and BRRY's gap to its ETF value) predicting the mid 120 batches ahead. Held-out R² is 0.08–0.29 by
symbol, only about 0.02 below in-sample. The strategy earns **$5,600/session** out of sample under
the backtest's fill rule. Under a realistic fill model (30% of trade size, 2 ticks through, real
pick-offs) that falls to **$1,540, and about $1,000** once you remove a look-ahead in the stress
test. **Pick-offs cost about 2.5¢/share, more than our 1.9¢ maker edge.** Live fill rate and
adverse selection matter more than model accuracy. The default settings survived a nested-CV
re-tune. News handling only steps aside and protects the features; it does not trade news.

## How to work

```bash
. .venv/bin/activate          # Python 3.12; run research scripts from research/
cd research
python final_cv.py            # base 3-fold out-of-sample score (should print $5,600 mean)
python final_compare.py       # before/after table: base, stress, synthetic news (~2 min)
```

- **Evaluate with `research/evalx.py`.** `evaluate(path, coefs="raw", scenario=..., news=...)`
  covers any strategy file, any stress scenario from `stress.py`, and synthetic news (`news={"D":
  repricing_batches}`). It runs on all cores, about 10 s for 24 sessions. `coefs="raw"` is the live
  model; the `*_std.json` files are a rejected variant.
- **Score markouts, never the backtest's "estimated PnL".** Report mean, worst session, maker
  ¢/share and shares, under **base and `share=0.3+through=2+adverse`**. Accept a change only if it
  helps the stressed case without costing base more than 10%.
- **Tune only with nested CV** (`tune_nested.py` is the template). Choosing on the sessions you
  report on inflated the last pick's gain by 3×.
- **Noise:** stress scenarios use seeded random rounding that depends on the fill path. Differences
  under about $60/session need paired, per-session comparison over several seeds (set the
  `STRESS_SEED` environment variable).
- **Before editing `alpha.py`, copy it to `strategies/archive/alpha_vN.py`** (v2 is the latest).
  Coefficients go in through a fit script → `bake.py`. Since v7 the live fit is
  `research/fit_imb_clip3.py` (fit.py's 33-feature model is the older one; its fold files are
  `fold{k}_base33.json`).

## Gotchas

- **macOS multiprocessing:** any script that calls `evaluate()` or `cv()` needs an `if __name__ ==
  "__main__":` guard, or it crashes with `BrokenProcessPool`.
- **Units:** prices are integer cents. Flow features are in thousands of shares, so compare
  coefficients by effect, not size.
- **The `adverse` stress rule looks ahead.** It decides pick-offs from the mid 120 batches later,
  so it leaks the future into our inventory and flatters any inventory-driven logic. Check
  per-share results on at-price vs other fills (see REPORT.md section 2) before believing a gain
  under it.
- **The backtest has no news, halts or competitors.** `evalx.py` injects synthetic news with
  invented jump sizes and repricing speeds; treat it as a sensitivity test.
- **HPPO = 1 BNKY + 2 BRRY** exactly (true value). Conversion costs $0.02/share and scores nothing
  itself.
- **The message budget is 20 per batch for the whole team.**

## What to build next, in order

1. **Sandbox validation (prompt 6).** It needs `PQT_API_KEY`. Above all, record real news items,
   how long the market takes to reprice after them, and our live fill rate. Those three numbers
   decide `news_pause`, whether the news rebase helps (it loses about $39/session if repricing
   takes 30 s), and how wide `make_edge` should be.
2. **News trading (prompt 5).** This is the largest untapped source of credited markout. Hook
   points in `alpha.py`:
   - `on_news` for the read;
   - `rebase()`, which already removes the jump from the features;
   - `FAIR_CAP`: a news-driven fair shift must bypass it or raise it.

   Test with `evalx` news runs, plus unit tests on made-up headlines. Add a GUI switch to turn it
   off.
3. **Adverse-selection defence (prompt 3).** Pick-offs cost more than our edge. Try quoting only the
   side the forecast favours when |fair − micro| is large; widen the other side. Accept a change on
   maker ¢/share under stress, not just dollars.
4. **ETF identity (prompts 2 and 4).** HPPO's and BNKY's forecasts don't use 1 BNKY + 2 BRRY;
   tying the three together, or an arbitrage leg, is untested.
5. **Hardening and runbook (prompt 7)** before R1, then the final review (prompt 8).

## Rejected, don't redo without a new reason

- **Scaled ridge with nested-CV penalty:** +0.003 R², but −$57/session under stress.
- **180 or 240-batch targets:** lower R² and lower markout.
- **`make_edge` 2.0:** +$9 out of sample under stress, which is noise.
- **`news_pause` 30:** mixed results across repricing speeds.
- **Own-fill markout guard** (widen edge and shrink size when a symbol's own maker markout turns
  negative; code in `strategies/archive/alpha_v6_guard_rejected.py`): −$15 base, −$23 realistic
  stress, −$6 stress without adverse, worse worst sessions in the failure cases. Online calibration
  already covers it. Revisit only if R1's report shows maker markout negative while the forecast
  still calibrates.

- **One-sided quoting** (prompt 3; `ONE_SIDE_K`/`ONE_SIDE_WIDEN` in `alpha.py`, left at 0 = off;
  `research/nested_oneside.py`): widening or pulling the side the forecast says gets picked off
  raises maker ¢/share by up to 0.13¢, but loses more volume than that. Every outer fold picked off;
  the best setting (K=3, pull) is −$68 base and −$15 realistic stress.
- **Redesign screen, 2026-11-07** (`research/screen.py`, per-session results in
  `charts/data/screen.csv`; paired vs v8, $/session, base / stress / stress without adverse):
  - **Capacity:** caps 400→600 for BRRY, HPPO or both: +12…−148 / −14…−20 / −2…+7. No effect; the
    net-exposure limit binds first.
  - **Wider BNKY edge** (×0.75, ×1.0): −475…−893 / −112…−177 / −121…−187. **Wider CRRT edge** ×0.9:
    −695 / −211 / −194.
  - **Skew** ×1.5: +38 under adverse stress but −72 without it (look-ahead flattery). ×0.67 is the
    mirror image.
  - **Quotes** ×1.5 on HPPO and BRRY: +203 base, flat under stress.
  - **ETF-consistent fair projection** (`ETF_PROJECT`) 0.5 / 1.0: flat, then worse.
  - **Volatility-scaled edge** (`VOL_GAMMA`) 0.5 / 1.0: worse.
  - **Conclusion:** v8's parameters sit at the backtest optimum. Further gains need live
    information (R1). `ETF_PROJECT` and `VOL_GAMMA` stay in `alpha.py` at 0 (off).
- **Per-symbol size learner** (`SIZE_LEARN`, off; `research/guards_eval.py`, 2026-11-07). Grows or
  shrinks each symbol's size on its own 10-minute markout. Paired vs v9 per session, base / stress /
  stress without adverse / model ×−1: starting at 0.5: −235 / −67 / −37 / −154; starting at 1.0:
  −10 / −44 / −8 / −128. Ten minutes of one symbol's fills is too noisy, so it shrinks on bad luck.
  Live learning beyond one number per symbol (calibration) has not paid: the markout guard, the
  throttle and this learner were all rejected.
- **Auto-throttle** (`THROTTLE_FROM`, off): costs nothing in healthy sessions, but −104 to −135
  with a wrong model: it cuts size after calibration has already recovered.
- **Around v10, 2026-11-07** (`research/nested_generic.py`, `charts/data/nested_v10_round.csv`):
  - **Cap 1 tick:** base $12,431 but realistic stress $1,181 (vs $3,202), model ×−1 −$9. A through-rule
    artefact in base.
  - **Cap 3:** worse everywhere.
  - **Narrower per-symbol multipliers** (HPPO/BRRY 0.75, CRRT 0.45): won every fold, +$118 stress,
    +$117 without adverse, +$38 at 10% share. Not adopted: it is the same direction as a lower
    make_edge, held at 1.5 because narrow edges lost more on the sandbox recording (2.5: −$320,
    1.0: −$520, 0.5: −$631 per 13 min under stress).
  - **make_edge:** nested CV picked 0.5, the bottom of the grid; 1.5 was adopted as the risk
    trade-off (v10).
