# Runbook: R1 and R2

For whoever runs the strategy on competition day. Each session is 45 minutes. Final score is
**R1 + 2 × R2**, so R1 is also our calibration run for R2. The score is the 60-second markout of our
fills. Estimated PnL in the GUI is **not** the score: don't react to it alone.

## Before the day

- [ ] **Freeze the strategy and its parameter list.** The exchange fixes the model-parameter panel
  for a session the first time a program declares it. A program with a different list is refused
  (`BAD_REQUEST`, "the parameters cannot change during a session") and runs on its defaults, with
  no GUI control. **Defaults count:** changing only one default (e.g. `arb_on` 1 → 0) is refused
  too (seen on the sandbox). A mid-session hotfix may change code, **never `MODEL_PARAMS`, not even
  a default**: change the value in the GUI instead.
- [ ] **Agree who runs what.** Exactly one program trades: `strategies/alpha.py`, on algo key 1.
  All keys and logins share one account, one set of limits and one 20-message budget per batch. If
  a teammate tests something else at the same time, both programs trade worse, and the first one
  to connect fixes the parameter panel.
- [ ] Run the checks:
  `.venv/bin/python tests/test_alpha_live.py`, then from `research/`: `../.venv/bin/python
  hardening.py` (24 replays with halts, one-sided books, dropout, late start, news) and
  `../.venv/bin/python final_cv.py`.
- [ ] **Decide the kill switch.** `kill_floor` is $20,000 of realized loss, which in practice never
  fires. Realized PnL = score + luck, so too tight a floor can stop a good session. Pick a floor as
  a team, in the GUI or as the default, before R1.
- [ ] Laptop: wired network, plugged in, and `caffeinate -i` in a spare terminal (macOS).

## Start of each session (R1, then R2)

```bash
cd pqt-student-kit && . .venv/bin/activate
set -a && . ./.env && set +a              # PQT_API_KEY = algo key 1; never paste keys in chat
pqt check --server <COMPETITION_URL> --key "$PQT_API_KEY"     # every line OK
pqt run strategies/alpha.py --mode live --cancel-on-disconnect \
    --server <COMPETITION_URL> --key "$PQT_API_KEY" 2>&1 | tee -a r1.log    # r2.log for R2
```

- You can start early: it waits for the session.
- `--cancel-on-disconnect` means our quotes are pulled if the laptop drops.
- One member opens the trader GUI (`<COMPETITION_URL>/trade/`) and takes the **control token** for
  the model panel. Check the panel lists every parameter (`make_edge` … `pause_BRRY`, `calib_on`).
  A short list means another program declared first: see "Something breaks".

## What to watch

The log prints one line a minute:
`minute N: positions | fair-micro | model weight | est PnL`.

- **Model weight** (0–1 per symbol) is the online calibration. Near 1 means the forecast is
  working; near 0 means it has stopped predicting this session and we are quoting around the
  microprice. Weights falling across all symbols in the first 10 minutes is the main early warning.
- **Positions pinned at the cap** (HPPO ±400, BNKY/CRRT ±1,500, BRRY ±400) for minutes on end
  mean one-sided quoting.
- **Rejects:** a few `UNKNOWN_ORDER` are normal. Repeated `EXPOSURE_LIMIT`, any `RATE_LIMIT` or
  `BLOCKED` is not.

## GUI levers (decided in advance; change one at a time and note the minute)

| situation | lever |
|---|---|
| model weights near 0 on most symbols, or our fills plainly losing | `take_on` 0, then `make_edge` 2.5 → 4 |
| one symbol misbehaving (halts, wild prints) | `pause_SYM` 1 |
| strong human view after news, e.g. obvious good news on BNKY | `bias_BNKY` (cents) for a minute, **then back to 0** |
| arbitrage legs filling one-sided (log lines `arb …` without all three fills) | `arb_on` 0 |
| everything is wrong | `enabled` 0 (cancels everything, stops); or the GUI kill switch |
| calibration clearly shrinking a model that is working | `calib_on` 0 (it keeps learning, so turning it back on is immediate) |

## Something breaks

- **Program crashed or the laptop died.** Restart with the same command. Resting orders persist
  unless the connection was lost with `--cancel-on-disconnect`. The strategy waits for 30 batches
  of fresh history before trading (late-start safe, tested in `hardening.py`).
- **"SUPERSEDED" (code 4001).** Another program used the same key. Find it and stop it.
- **Panel shows the wrong parameters (`BAD_REQUEST` param_schema).** Another program declared
  first. Stop it if it is not ours. Our program still trades on its defaults; GUI control of our
  parameters comes back only next session.
- **Exchange restart** (broadcast; all symbols halted, all orders cancelled with reason `SYSTEM`).
  Nothing to do: the strategy stops quoting halted symbols and requotes 20 batches after each
  resumes.
- **News.** Quotes are pulled for `news_pause` (10) batches on every item. The strategy does not
  trade news yet. A human view goes in through `bias_SYM`.

## After R1, before R2

- The official report gives score by symbol, markout curves at 1, 10, 30 and 60 s, and realized PnL
  vs score.
- **Compare** with the backtest:
  - maker ¢/share (backtest ~1.9¢; below ~1¢ means widen `make_edge`);
  - taker and arbitrage markout (negative means `take_on` 0 or `arb_on` 0 for R2);
  - shares filled (backtest ~340k a session: how much of that did we really get?);
  - how fast prices repriced after each news item (sets `news_pause`).
- **Change defaults** in `alpha.py` only between sessions, with an archive copy first
  (`strategies/archive/`), the checks above re-run, and the parameter list unchanged unless the
  whole team switches together.
