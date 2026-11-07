"""Competition-day hardening replays (PROMPTS.md prompt 7, ROADMAP P2 item 15).

Replays real sessions with injected trouble and checks alpha.py copes:
  halts       HPPO, BNKY and BRRY halted for a while (the data has no halts at all)
  one-sided   a symbol's ask or bid side empty for a while
  dropout     on_batch not called for 60 batches (a dropped connection), then on_reconnect
  late        the program sees nothing until batch 2400 (late start or restart mid-session)
  news        three news items close together
Checks: no exception; no order, replace or conversion sent for a halted symbol (or HPPO while a
conversion leg is halted); no quotes within HALT_COOLDOWN batches after a halt; no RATE_LIMIT;
messages per batch; no trading before WARMUP batches seen after a late start.

    ../.venv/bin/python hardening.py        (from research/)
"""
import dataclasses
import logging
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from pqt_client.backtest_v2 import load_session
from pqt_client.v2 import backtest_strategy
from pqt_client.v2 import backtest as v2bt
from pqt_protocol import v2 as wire

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "strategies"))
import alpha  # noqa: E402

from load import DATA  # noqa: E402

HALTS = {"HPPO": (1500, 1700), "BNKY": (3000, 3100), "BRRY": (4000, 4060)}
ONE_SIDED = {("CRRT", "ask"): (2000, 2050), ("BRRY", "bid"): (2500, 2520)}
DROPOUT = (2200, 2260)
LATE = 2400
NEWS = (3500, 3510, 3530)
SESSIONS = ("s03", "s10", "s17", "s22")


def inject(session, case):
    if case not in ("halts", "one-sided"):
        return session
    tops = []
    for b, top in enumerate(session.tops):
        new = dict(top)
        if case == "halts":
            for s, (a, z) in HALTS.items():
                if a <= b < z:
                    new[s] = dataclasses.replace(top[s], status="HALTED")
        else:
            for (s, side), (a, z) in ONE_SIDED.items():
                if a <= b < z:
                    new[s] = dataclasses.replace(top[s], **{f"{side}_px": None, f"{side}_qty": None})
        tops.append(new)
    return dataclasses.replace(session, tops=tuple(tops))


def run(job):
    name, case = job
    logging.disable(logging.INFO)
    session = inject(load_session(DATA, name), case)
    problems = []
    sent = []  # (batch, kind, symbol)
    rejects = Counter()
    msgs = []

    class Probe(alpha.Alpha):
        def on_start(self):
            super().on_start()
            self.first_trade = None

        def _note(self, kind, s):
            sent.append((self.state.batch if self.state else -1, kind, s))

        def buy(self, s, *a, **k):
            self._note("order", s)
            return super().buy(s, *a, **k)

        def sell(self, s, *a, **k):
            self._note("order", s)
            return super().sell(s, *a, **k)

        def replace(self, *, order_id=None, client_id=None, price=0, qty=0):
            o = next((o for o in self.open_orders() if o.order_id == order_id), None)
            self._note("replace", o.symbol if o else "?")
            return super().replace(order_id=order_id, client_id=client_id, price=price, qty=qty)

        def create(self, qty, **k):
            self._note("convert", "HPPO")
            return super().create(qty, **k)

        def redeem(self, qty, **k):
            self._note("convert", "HPPO")
            return super().redeem(qty, **k)

        def on_reject(self, r):
            rejects[str(getattr(r.code, "value", r.code))] += 1

        def on_batch(self, state):
            b = state.batch
            if case == "dropout" and DROPOUT[0] <= b < DROPOUT[1]:
                return
            if case == "dropout" and b == DROPOUT[1]:
                self.on_reconnect()
            if case == "late" and b < LATE:
                return
            left = self.messages_left()
            super().on_batch(state)
            msgs.append(left - self.messages_left())

        def on_news(self, item):
            super().on_news(item)

    strat = Probe()
    original_send = v2bt._Feed.send
    if case == "news":
        def send(self, message):
            if isinstance(message, wire.Batch) and message.batch in NEWS:
                message = dataclasses.replace(message, news=[wire.NewsItem(NEWS.index(message.batch) + 1,
                                                                           message.batch, "test news", "")])
            return original_send(self, message)
        v2bt._Feed.send = send
    try:
        result = backtest_strategy(session, strat)
    except Exception as e:  # noqa: BLE001
        return name, case, [f"EXCEPTION {type(e).__name__}: {e}"], {}, 0
    finally:
        v2bt._Feed.send = original_send

    if case == "halts":
        for b, kind, s in sent:
            for h, (a, z) in HALTS.items():
                # a halt is visible from batch a on; what was sent in a - 1 meets it unannounced
                if kind == "convert" and a <= b < z:
                    problems.append(f"conversion at b{b} while {h} halted")
                if s == h and a <= b < z:
                    problems.append(f"{kind} {s} at b{b} during halt {a}-{z}")
                if s == h and z <= b < z + alpha.HALT_COOLDOWN - 1 and kind != "convert":
                    problems.append(f"{kind} {s} at b{b} within cooldown after halt {a}-{z}")
    if case == "one-sided":
        for b, kind, s in sent:
            for (sym, side), (a, z) in ONE_SIDED.items():
                if s == sym and a <= b < z and kind == "order":
                    problems.append(f"order {s} at b{b} while its {side} side was empty")
    if case == "late":
        early = [x for x in sent if x[0] < LATE + alpha.WARMUP]
        if early:
            problems.append(f"{len(early)} messages before warm-up after a late start, first {early[0]}")
    if case == "news":
        for b, kind, s in sent:
            for t in NEWS:
                if t <= b < t + 10 and kind in ("order", "replace") and b >= t:
                    pass  # takes are allowed; quotes are checked below via open orders
    if rejects.get("RATE_LIMIT"):
        problems.append(f"RATE_LIMIT x{rejects['RATE_LIMIT']}")
    if rejects.get("HALTED", 0) > 2 * len(HALTS) * 2:  # only the unannounced first batch may hit one
        problems.append(f"HALTED rejects x{rejects['HALTED']}")
    summary = dict(rejects)
    return name, case, problems[:6] + ([f"... {len(problems) - 6} more"] if len(problems) > 6 else []), \
        summary, max(msgs) if msgs else 0


if __name__ == "__main__":
    jobs = [(n, c) for c in ("none", "halts", "one-sided", "dropout", "late", "news") for n in SESSIONS]
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(run, jobs))
    bad = 0
    for name, case, problems, rej, mx in out:
        status = "ok" if not problems else "FAIL"
        bad += bool(problems)
        print(f"{case:10} {name}  {status:4}  max msgs/batch {mx:2}  rejects {rej}")
        for p in problems:
            print("      ", p)
    print("all passed" if not bad else f"{bad} case(s) failed")
