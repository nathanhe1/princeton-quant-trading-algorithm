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
- [ ] **Kill switch: default floor −$2,000 of realized PnL** (`kill_floor` 2000, set in `on_start`).
  In `research/killswitch_study.py` the deepest intra-session realized PnL across all 24 historical
  sessions, in every scenario including a wrong-signed model under stress, is −$411. The v8 disaster
  on the sandbox reached −$1,954 by minute 12. So −$2,000 never trips a healthy session and does
  stop a runaway. When it fires, the exchange cancels the team's orders for the rest of the session.
  The floor is team-wide (manual GUI fills count too) and watches **realized** PnL only: a position
  going bad without being closed does not trip it. Watch the markout line and estimated PnL as
  well. The strategy arms the floor in its first batch, and skips it (with a warning) if realized
  PnL is already below it after a restart. Changing `kill_floor` in the panel applies at once.
  Only `strategies/alpha.py` runs in R1/R2: archive copies declare a different parameter list.
- [ ] **No manual resting orders while the strategy runs.** It sees the whole team's open orders
  and will move or cancel a resting GUI limit order as if it were its own (also with `enabled` 0,
  `pause_SYM` and after news). Manual trades must be IOC or market orders.
- [ ] Laptop: wired network, plugged in, and `caffeinate -i` in a spare terminal (macOS).

## Start of each session (R1, then R2)

```bash
cd pqt-student-kit && . .venv/bin/activate
set -a && . ./.env && set +a              # PQT_API_KEY = algo key 1; never paste keys in chat
pqt check --server <COMPETITION_URL> --key "$PQT_API_KEY"     # every line OK
PROBE_BATCHES=0 pqt run research/sandbox_probe.py:Probe --mode live --cancel-on-disconnect \
    --server <COMPETITION_URL> --key "$PQT_API_KEY" 2>&1 | tee -a r1.log    # r2.log for R2
```

- This runs `alpha.py` unchanged through the logger. `PROBE_BATCHES=0` means it never stops
  itself. Every fill, the book every batch, every news item, halt and reject go to
  `research/sandbox_runs/<time>.jsonl`: R1's file is what we tune R2 on.
- You can start early: it waits for the session.
- **One machine per algo key.** If anyone else connects with the same key (a teammate's `pqt run`,
  a test script), the exchange drops our connection ("SUPERSEDED: a newer connection with this key
  replaced this one") and the logger exits without reconnecting. Our quotes are pulled but the
  positions stay, unmanaged. Seen on the sandbox, 2026-11-07. During R1 nobody else runs anything
  with key 1; if the log shows SUPERSEDED, find the other program first, then restart ours.
- **Start a fresh process for each session.** Once a process sees a session end, the SDK refuses
  every later order client-side ("the session has ended"), so a logger left running from R1 sits
  idle through R2 while looking connected (positions stay 0, est PnL $0). Worse, when R2 opens, that
  idle process declares R2's parameter list with **its** defaults, and the list is then locked for
  R2 (seen on the sandbox: a leftover logger locked in make_edge 2.5 and a $20,000 kill switch). Stop
  it with Ctrl-C right after R1 ends, before R2 opens, and run the command again for R2.
- `--cancel-on-disconnect` means our quotes are pulled if the laptop drops.
- One member opens the trader GUI (`<COMPETITION_URL>/trade/`) and takes the **control token** for
  the model panel. Check the panel lists every parameter (`make_edge` … `pause_BRRY`, `calib_on`).
  A short list means another program declared first: see "Something breaks".

## What to watch

The log prints one line a minute:
`minute N: positions | fair-micro | model weight | est PnL`.

- **Markout line** (`markout, last 10 min of fills …`): our fills vs the mid 60 s later, per
  symbol, M(aker)/T(aker), ¢/share and shares. It is the closest live view of the score. Backtest
  norms: maker about +1 to +3¢ per symbol; taker small volume.
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
| from minute 12, the markout line is negative overall two minutes running | `size_scale` 1.5 → 0.75 (sandbox: full size + takes lost, small size didn't) |
| taker (T) markout negative on the markout line | `take_on` 0 |
| still negative 5 minutes after cutting size | `make_edge` 1.5 → 2.5, then → 4 |
| model weights near 0 on most symbols | nothing by itself: calibration already shrank the model; watch the markout line |
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
- **Our own data first:** `.venv/bin/python research/sandbox_report.py research/sandbox_runs/<R1 file>.jsonl`
  gives markout by symbol and fill type, fill rate, rejects, each news item with the mid path
  1–60 s after it (repricing speed), and halts. Paste R1's news headlines into
  `tests/test_alpha_live.py::test_read_news` and check `read_news` gets them right before turning
  `news_trade` on for R2.
- **Compare** with the backtest:
  - maker ¢/share (backtest ~1.9¢; below ~1¢ means widen `make_edge`);
  - taker and arbitrage markout (negative means `take_on` 0 or `arb_on` 0 for R2);
  - shares filled (backtest ~340k a session: how much of that did we really get?);
  - how fast prices repriced after each news item (sets `news_pause`).
- **Change defaults** in `alpha.py` only between sessions, with an archive copy first
  (`strategies/archive/`), the checks above re-run, and the parameter list unchanged unless the
  whole team switches together.
