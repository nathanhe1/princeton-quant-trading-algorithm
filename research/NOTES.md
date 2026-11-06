# Research notes (alpha v1)

## Setup
- Python 3.12 venv in `.venv` (made with `uv`; macOS system Python is 3.9, too old). Activate: `. .venv/bin/activate`
- SDK 0.2.3, data in `data/` (24 sessions).

## What the data says
| finding | evidence |
|---|---|
| Prices follow the hidden true value with a lag → **short-term momentum** | own 15 s return vs next 60 s: corr ≈ 0.3 (HPPO, BNKY, CRRT) |
| Signed aggressor flow predicts the next minute | 60-batch flow vs next 60 s: corr ≈ 0.2 |
| **BRRY lags** the ETF identity; HPPO and BNKY lead | BRRY mid closes ~58% of the gap to (HPPO−BNKY)/2 within 60 s; HPPO/BNKY ≈ 0% |
| CRRT moves against HPPO and lags it | HPPO past 15 s return vs CRRT next 60 s: corr −0.26 |
| Microprice beats mid as a fair value everywhere | `research/fair.py` |
| The predictable part is large compared with costs | predicted 60 s move std: HPPO 4.7¢, BRRY 4.8¢, CRRT 2.1¢, BNKY 1.6¢; out-of-sample slope 0.91–0.97 |

The 240-batch markout is consistently above the 120-batch one, so the mid is still catching up after
a minute: the score against the true value V is probably higher than our 120-batch proxy.

## Model
`fair[s] = micro[s] + Σ coef·features`. Features: microprice returns over 10/30/60/120/240 batches for all
4 symbols, signed flow over 30/120 batches, micro−mid, and the BRRY ETF gap. Ridge regression on
mid(t+120) − micro(t), pooled over sessions (`research/fit.py`), baked into `strategies/alpha.py` by
`research/bake.py`.

## Strategy (`strategies/alpha.py`)
Quote both sides of all 4 symbols around fair ± `make_edge`·mult, lean against inventory; take only
when fair is beyond the touch by fee + `take_edge`·mult. Pre-checks for position and exposure limits;
own quotes removed from the book live; quotes off after news and halts; kill switch.

## Results: 3-fold cross-validation (coefficients never saw the test sessions)
`python research/final_cv.py`. Proxy score = Σ q·(mid 120 batches later − price) − fees − penalties.

| | per session |
|---|---|
| proxy markout (120) | **$5,291** mean, sd $949, min $3,215, 24/24 positive |
| maker | 265k shares at 1.96¢/share |
| taker | 1.5k shares at 5.9¢/share |
| by symbol | CRRT $2,375 · BNKY $1,466 · HPPO $764 · BRRY $685 |

Sweeps: `sweep_take.py`, `sweep2.py`, `sweep3.py`. Bug found and fixed on the way: replacing a bid up
through our own ask self-traded ($0.10 penalty); now the side that is getting out of the way moves first.

## What will be worse live (be sceptical of the dollar figure)
1. **Volume share.** The backtest fills our quote whenever a recorded trade prints through it; 60–70
   teams quoting means we get a fraction of that. Per-share edge (≈2¢) is the more meaningful number.
2. **Adverse selection and competition.** Others will find the same momentum; the lag may be shorter live.
3. **Own-order feedback.** Handled in code (`external()`), but only exercised by a smoke test so far:
   verify on the sandbox.
4. **News** is absent from the data. We just step aside for `news_pause` batches; reading it is open alpha.

## Next
- Sandbox run (`pqt check`, then `pqt run ... --mode test`): verify the live paths, the panel and the logs.
- Stress the fill model: haircut maker fills, require trades ≥ 2 ticks through, and re-tune `make_edge`.
- News parser once the sandbox's news format is visible.
- Per-symbol edge and size tuning; per-session regime (volatility) scaling.

## 2026-11 research update
- `research/model.py` now compares the 120-, 180- and 240-batch targets with fold-specific
  coefficients and strategy markouts. The 120-batch model remains best: mean proxy markout
  $5,291/session versus $4,474 (180) and $4,081 (240); its held-out R² is 0.118 HPPO,
  0.078 BNKY, 0.158 CRRT and 0.286 BRRY. No fair-value coefficients were changed.
- `research/stress.py` implements seeded trade-size haircuts, stricter through-price fills and a
  deliberately adverse at-price fill. Its base scenario reproduces `final_cv.py`.
- Skew was tuned only on s01-s08 and checked separately on s17-s24 and s09-s16. Increasing all
  four per-symbol skew values by 50% improved the realistic stress mean on both check sets without
  reducing base markout. `strategies/alpha.py` now uses that setting; the prior strategy is saved
  as `strategies/archive/alpha_v1.py`.
- After the change, fold-based 120-batch markout is $5,600/session, worst $3,382, with all 24
  sessions positive. The baseline was $5,291/session, worst $3,215. Maker markout changed from
  1.96¢ to 1.93¢ per share, so the dollar gain comes from higher filled volume, not higher unit
  edge. Recheck under the completed stress run before relying on the stressed result.
