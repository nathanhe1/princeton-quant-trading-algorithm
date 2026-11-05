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
