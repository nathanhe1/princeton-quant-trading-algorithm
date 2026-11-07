# Roadmap: everything to fix, improve and add before R1 (2026-11-06)

Written after the first sandbox run of `alpha.py` (session sandbox-0196, batches 1215–2416, 10
minutes, log `research/sandbox_runs/20261107T025307Z.jsonl`, summary from
`research/sandbox_report.py`). Work top to bottom. Each item says why, what to do, and how to
accept it. Tick items off here and record the result in one line, so the next session knows where
things stand.

## What the sandbox run showed

| | sandbox (10 min) | backtest |
|---|---|---|
| markout, all fills, vs mid 60 s later | **−2.4¢/share, −$2,022** | +1.9¢ maker, +6.1¢ taker (24-session OOS) |
| maker shares / markout | 63k / −0.8¢ | 286k per 45-min session / +1.93¢ |
| taker shares / markout | **28.7k / −5.9¢** | **1.6k per 45-min session** / +6.1¢ |
| forecast vs next 60 s move, correlation | **−0.32 to −0.64**, all four symbols | +0.23 to +0.55, BNKY ≈ 0 (s05) |
| typical 60 s move (sd) | HPPO 12.5¢, BNKY 14¢, CRRT 12.6¢, BRRY 7¢ | HPPO 7¢, BNKY 3¢, CRRT 1.6¢, BRRY 5.6¢ (s05) |
| time with \|position\| ≥ 90% of soft cap | HPPO 32%, BNKY 38%, CRRT 10%, BRRY 3% | not measured yet |
| self-trades | 1 (10 shares, $1 penalty): **fixed** | 0 |

**What carries over and what doesn't.** The sandbox's prices and participants are placeholders
(STRATEGY_GUIDE section 9, and its own news item says so), so the negative correlation says nothing
about our model in the real market. What does carry over is how the strategy **behaves when its model
is wrong**:

1. **It never noticed.** It kept quoting and taking at full size for 10 minutes at −2.4¢/share.
   Each competition session draws fresh market parameters, so some sessions will be off our
   historical fit. Nothing in the code detects that.
2. **Taking scales with the size of the error.** A wrong, large `fair − micro` produces more takes:
   18× a whole backtest session's taker volume in 10 minutes, losing 6¢ a share on average. Takes earn about
   $100/session in the backtest, which is little upside for that downside.
3. **Inventory sits at the cap.** Skew is fixed cents (BNKY 1.5¢ at cap), so with larger moves the
   position runs to ±cap and we quote one side only, which is the side the market is hitting.

Mechanics were clean: on_batch 0.33 ms median (13 ms worst), at most 9 of 20 messages per batch,
no missed batches or reconnects, `external()` correct on all 3,160 at-touch checks, the news pause
and rebase fired, and the probe exited with no orders left. Rejects: 78 UNKNOWN_ORDER (expected)
and 7 EXPOSURE_LIMIT. No halts happened, so halt handling is still untested live.

## Progress log

- **2026-11-06, v5 = Rushil's v4 + online calibration.** Model-failure harness:
  `research/failure.py` (`COEF_MULT` in `evalx.py`); nested choice of the calibration threshold:
  `research/nested_calib.py` (every fold picked `CALIB_FULL` = 1.0). Tests: `tests/test_alpha_live.py`.
  3-fold OOS $/session (stress = realistic, seed 2026; paired over seeds 2026/7/99 where marked):

  | case | v3.1 (pre-v4) | v4 (Rushil) | **v5** |
  |---|---|---|---|
  | base, model right | 5,602 | 6,058 | **6,232** (vs v4 +174 ± 51, 20/24) |
  | stress, model right | 1,540 | 1,514 | 1,451 (vs v4 **−67 ± 25**, all 3 seeds) |
  | base / stress, model ×−1 | −304 / −1,543 | −371 / −1,626 | **3,482 / −72** |
  | stress, model ×−0.5 | −767 | −664 | **116** |
  | base / stress, model ×2 | 2,269 / 741 | 2,471 / 763 | **4,447 / 1,217** |
  | base / stress, no model | 4,707 / 186 | 4,270 / 256 | 4,270 / 256 |

  Decision: accept v5. It pays 4% under stress when the model is exactly right, and gains
  $450–1,550/session under stress when the model is wrong or overconfident. `calib_on` turns
  it off live.
- **About v4's arbitrage:** ~14k IOC shares/session, about $0 under own-mid marking (+$13 with the
  model, −$60 without). Rushil's +$123 stress figure marks HPPO against the basket mid
  (ANALYSIS.md appendix D). That is defensible, because V(HPPO) = V(BNKY) + 2 V(BRRY) exactly, but
  our harness does not do it yet. Open: decide on one marking rule for HPPO and use it everywhere.
  v4 also loses $436/session vs v3.1 when the model is absent (expo lean + arb); understand why.
- **Fixed in v5 (live-only, invisible to the backtest):**
  - the arbitrage sized legs off `external()`'s guessed level (100 shares invented when we are alone
    at the touch);
  - its message check ignored BRRY's split orders.

- **2026-11-06, hardening (prompt 7).**
  - **Fixed:** a late start or restart mid-session traded at once on no history, because `WARMUP`
    compared the exchange's batch number instead of batches seen. It now counts batches seen.
  - **Fixed:** `on_reconnect` lifted an active news pause.
  - **Tests:** `research/hardening.py` replays 4 sessions with injected halts (the data has none),
    one-sided books, a 60-batch dropout, a late start and a news burst. All 24 pass: no exception,
    nothing sent into a visible halt or its cooldown, no RATE_LIMIT, at most 12 msgs/batch.
  - **Runbook:** `RUNBOOK.md`. Base OOS unchanged at $6,232.
- **Rejected: own-fill markout guard** (see HANDOFF.md "Rejected").
- **2026-11-06, news trading (prompt 5), built and kept off.**
  - **Code:** `read_news()` in `alpha.py` is a keyword parser. It acts only on one stock named,
    one-sided keywords and no negation, and propagates BNKY/BRRY news to HPPO. A confident read sets
    a fair-value target (pre-news micro + `news_move` × the symbol's typical 60 s move) for
    `news_hold` batches. GUI: `news_trade` (default **0**), `news_move`, `news_hold`.
  - **Evaluation:** `research/news_eval.py` on evalx synthetic news (directional headlines; `flip`
    = misread). $/session gain vs off, base / stress:

    | repricing | read right | misread |
    |---|---|---|
    | instant | −31 / −1 | −388 / −197 |
    | 10 s | +436 / +489 | −1,676 / −1,414 |
    | 30 s | +599 / +786 | −2,046 / −1,690 |

  - **Decision:** break-even needs the parser right on about 70–75% of the items it acts on, and the
    real news format is unknown (the sandbox shows only "routine notice"). Turn it on for R2 only if
    R1's news reads correctly and repricing takes ≥ 10 s. Paste R1's real items into a test.

## P0: robustness, before R1 (independent of the sandbox)

Accept each P0 change under the existing rules: nested CV, stress `share=0.3+through=2+adverse` as
the primary case, and base not worse by more than 10%. Also add a **model-failure scenario** to
`evalx.py` and require P0 changes to improve it. To build it, run the strategy with its
coefficients multiplied by −0.5, 0 (no model) and +2 (overconfident), and with prices rescaled to 2×
volatility. Today the strategy has no defence in that scenario, and that is the sandbox's lesson.

1. **Live skill monitor with auto-throttle** (largest item). *Part done in v5: the forecast is
   scaled by its live calibration. Still open: own-fill markout monitoring and the defensive state.*
   - **Measure, per symbol and online:**
     - the rolling markout of our own fills against the mid 10, 30 and 60 s later (we know these
       mids live);
     - the rolling correlation of `fair − micro` with the realized 60 s move.
   - **Three states:** *normal*; *defensive* (`make_edge` ×2, size ×0.5, taking off); *model off*
     (quote symmetric around micro with a wide edge, or stop quoting the symbol).
   - **Rules:** state changes need hysteresis and a minimum sample. Log every change.
   - **GUI:** show the state with `self.record()`, and add a model param to force a state from the
     GUI.
   - **Accept:** it cuts the model-failure scenario's loss by more than half without costing base
     more than 10%.
2. **Gate taking hard, or default it off.**
   - **Options:**
     - make `take_on` default 0 until R1's report shows positive taker markout;
     - or require a volatility-normalised signal (`|fair − micro|` / rolling sd of 60 s moves), a
       healthy skill monitor, a taker-share budget per minute and a cooldown after each take.
   - **Accept:** paired per-session comparison. The expected cost is at most about $100/session in
     base.
3. **Volatility-scaled parameters.** Scale `make_edge`, `take_edge`, `FAIR_CAP` and `SKEW` by a
   rolling estimate of each symbol's volatility, measured as the sd of 60 s mid moves or the
   average spread, relative to its historical median. Market parameters change every session, so
   fixed cents are tuned to an average session. Check robustness across the historical sessions
   bucketed by volatility.
4. **Stronger inventory control.**
   - **Changes:** nonlinear skew (gentle near 0, steep past 50% of cap), quote size shrinking with
     \|position\|, and stopping the side that adds to the position before the cap, not at it.
   - **Measure in the backtest:** time at ≥90% cap and markout split by inventory bucket.
   - **Why:** score isn't charged for holding, but a pinned position means one-sided quoting.
5. **Self-trade guard: done, unit test added** (`own_at_or_through` in `alpha.py`; archive `alpha_v3.py`). Base
   OOS $5,602 vs $5,600. Still to do: a unit test that builds a book with our own resting ask and
   checks no take is sent at or through it.
6. **The `external()` fallback is untested by every backtest.** In backtest mode it returns the raw
   book, so the CV never runs the live branch.
   - **The problem:** we were alone at the touch in 57% of the at-touch batches. In that case
     `external()` invents a level (the last one seen, or one tick behind with 100 shares) that feeds
     the microprice and every return feature.
   - **Fixes:** inject our own quotes into the recorded book in a backtest variant so this path
     runs, and compare the fallbacks (last real level, mid of the known levels, freeze features).
7. **EXPOSURE_LIMIT on replaces** (7 in 10 min). Find out why our 580 margin is not enough. The
   likely cause is how the exchange counts a replace (old and new both in the worst case). It's
   harmless for now (a skipped requote), but each one costs a message.
8. **Self-impact.** At about 9k shares/min we may move a thin book, and the return features then
   read our own impact as momentum. This can't be measured in the backtest (the book never reacts)
   or on the sandbox (placeholder prices).
   - **Mitigations to test:** smaller clip sizes; feature returns that skip batches right after our
     own takes.
   - **First real check:** R1's report, comparing markout by our trade size.

## P1: edge (after P0)

9. **News trading** (PROMPTS.md prompt 5). The largest untapped credited markout. Hooks: `on_news`,
   `rebase()`, `FAIR_CAP` bypass. GUI kill switch for it. Test with `evalx` synthetic news plus
   unit tests on made-up headlines. Sandbox news is placeholder: "news here moves nothing".
10. **Adverse-selection defence** (prompt 3). Quote only the side the forecast favours when
    \|fair − micro\| is large; widen the other side. Accept on maker ¢/share under stress.
11. **Join, don't improve.** We were alone at the best price 57% of the time, so we are
    setting new prices, and those get picked off first. Try joining the existing level unless the
    edge is large. With 60–70 teams in the market, queue position and fill share are much worse
    than in the backtest.
12. **ETF identity and arbitrage** (prompts 2 and 4). HPPO = BNKY + 2 BRRY exactly in true value.
    Tie the three forecasts together, and test an arbitrage leg (with the $0.02 conversion fee).
13. **Horizon vs true value.** All research scores against the mid 60 s later; the 240-batch
    markout is higher, so the true value may lead the mid. Re-check once R1 publishes true-value
    paths.

## P2: tooling and operations

14. **Live dashboard in the trader GUI** with `self.record()`: fair − micro, rolling own markout,
    skill state, position vs cap, messages used.
15. **Unit tests** (`tests/`). *Started: `tests/test_alpha_live.py` (self-trade guard, `external()`,
    arbitrage guards); `research/hardening.py` covers halts, dropout, late start and news.* Still to add: take guard, `external()` branches, `room()` and exposure bounds, news
    pause and rebase, halt cooldown, kill switch set in `on_start`.
16. **Sandbox mechanics retest** after every change, with `research/sandbox_probe.py` (it runs
    `alpha.py` unchanged and exits cleanly after `PROBE_BATCHES`). Judge only mechanics there: rejects,
    messages, timing, halts. Never judge PnL or markout there.
17. **Runbook** (prompt 7). **The parameter schema is locked per session.** The sandbox rejected
    v5's schema with `param_schema: the parameters cannot change during a session` (BAD_REQUEST),
    because a teammate's program had already declared a different one in sandbox-0197.
    - **Consequence:** the GUI panel cannot reach the parameters a later version adds, and the
      program runs on its defaults.
    - **Rule:** freeze `MODEL_PARAMS` before R1; a mid-session hotfix may change code but never
      parameters.
    - **Coordination:** decide which teammate's program declares first.
    - **Before:** `pqt check` before each session; one algo key per program (a second copy
      supersedes the first); `--cancel-on-disconnect`.
    - **Kill switch:** the floor is set in `on_start`. Reconsider the floor: $20,000 never fires in
      practice.
    - **During:** which GUI knob to turn when, decided in advance (`take_on` off, `make_edge` up,
      `size_scale` down, force the defensive state).
18. **After R1** (score = R1 + 2 × R2, so R1 is our calibration run):
    - **Inputs:** the official report's markout curves (1/10/30/60 s by symbol) and our fill log.
    - **Decide for R2:** maker ¢/share (below about 1¢ means widen `make_edge`), taker markout
      (keep or drop taking), fill share vs backtest, news repricing time (sets `news_pause`).
    - **Prepare:** the analysis script, before R1.

## What the sandbox can never tell us

Real markout and adverse selection, live fill share against 60–70 teams, news repricing speed, the
real market's volatility, and whether we move the market. All five wait for R1's report.
