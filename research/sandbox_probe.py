"""Sandbox probe: runs strategies/alpha.py unchanged and logs what prompt 6 needs to check.

    pqt run research/sandbox_probe.py:Probe --mode test --cancel-on-disconnect \
        --server https://sandbox.princeton-quant.com --key $PQT_API_KEY

Writes research/sandbox_runs/<utc time>.jsonl, one JSON object per line ("ev" says what it is):
batch (timing, messages, raw and external touch, positions), reject, fill, news, halt, params,
broadcast, round. After PROBE_BATCHES batches (default 1200 = 10 min) it stops quoting, cancels
everything and exits. Analyse with research/sandbox_report.py.
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "strategies"))
from alpha import SYMS, Alpha  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "sandbox_runs"
STOP_AFTER = int(os.environ.get("PROBE_BATCHES", "1200"))


class Probe(Alpha):
    def on_start(self) -> None:
        super().on_start()
        OUT_DIR.mkdir(exist_ok=True)
        self.path = OUT_DIR / time.strftime("%Y%m%dT%H%M%SZ.jsonl", time.gmtime())
        self.fh = open(self.path, "w", buffering=1)
        self.first = None
        self.stop_at = None
        self.rejects = Counter()
        self.halted = {s: False for s in SYMS}
        self.last_wall = None
        self.emit("start", session=getattr(self.session, "session_id", str(self.session)),
                  params=dict(self.model_params), limits=str(self.limits),
                  symbol_info={s: str(i) for s, i in self.symbol_info.items()})
        self.log.info("probe logging to %s", self.path)

    def emit(self, ev, **kw):
        kw["ev"] = ev
        kw["t"] = round(time.time(), 3)
        self.fh.write(json.dumps(kw, default=str) + "\n")

    # ---------------------------------------------------------------- events
    def on_reject(self, r) -> None:
        self.rejects[str(r.code)] += 1
        self.emit("reject", batch=r.batch, code=str(r.code), cid=r.client_id, oid=r.order_id,
                  detail=r.detail)

    def on_fill(self, f) -> None:
        super().on_fill(f)
        s = f.symbol
        self.emit("fill", batch=f.batch, sym=s, side=str(f.side), px=f.price, qty=f.qty,
                  liq=str(f.liquidity), fee=f.fee, fair=self.fair.get(s),
                  micro=self.ext[s].micro if s in self.ext else None)

    def on_news(self, item) -> None:
        tops = {s: [t.bid_px, t.ask_px] for s, t in self.state.top.items()} if self.state else {}
        self.emit("news", batch=item.batch, id=item.news_id, headline=item.headline,
                  body=item.body, tops=tops)
        super().on_news(item)
        self.emit("news_pause", pause_until=self.pause_until, rebase_at=self.rebase_at)

    def on_model_params(self, changed) -> None:
        self.emit("params", changed=dict(changed), all=dict(self.model_params))
        self.log.info("model params changed: %s", changed)

    def on_broadcast(self, b) -> None:
        self.emit("broadcast", text=b.text, level=b.level)

    def on_round_state(self, r) -> None:
        self.emit("round", phase=str(r.phase), text=getattr(r, "text", ""))

    def on_reconnect(self) -> None:
        super().on_reconnect()
        self.emit("reconnect")

    # ---------------------------------------------------------------- batch
    def on_batch(self, state) -> None:
        wall = time.time()
        if self.first is None:
            self.first = state.batch
        n = state.batch - self.first
        for s in SYMS:
            h = state.top[s].halted
            if h != self.halted[s]:
                self.emit("halt", batch=state.batch, sym=s, halted=h)
                self.halted[s] = h

        if self.stop_at is not None:  # winding down: our cancels went out last batch
            if state.batch >= self.stop_at + 4 or not self.open_orders():
                self.emit("stop", batch=state.batch, open=len(self.open_orders()),
                          rejects=dict(self.rejects))
                self.fh.close()
                self.log.info("probe done after %d batches; log %s", n, self.path)
                os._exit(0)
            self.cancel_all()
            return
        if n >= STOP_AFTER:
            self.params_override = {"make_on": False, "take_on": False}
            self.cancel_all()
            self.stop_at = state.batch
            self.emit("stopping", batch=state.batch)
            return

        used_before = self.limits.max_msgs_per_batch - self.messages_left()
        t0 = time.perf_counter()
        super().on_batch(state)
        dt = (time.perf_counter() - t0) * 1000
        used = self.limits.max_msgs_per_batch - self.messages_left()

        raw, ext, ours = {}, {}, {}
        for s in SYMS:
            t = state.top[s]
            raw[s] = [t.bid_px, t.bid_qty, t.ask_px, t.ask_qty]
            e = self.ext.get(s)
            ext[s] = [e.bid, e.bq, e.ask, e.aq] if e else None
            oo = self.open_orders(s)
            ours[s] = [[str(o.side), o.price, o.remaining] for o in oo]
        self.emit("batch", batch=state.batch, n=n, dt_ms=round(dt, 2), msgs=used,
                  msgs_events=used_before, gap_ms=round((wall - self.last_wall) * 1000) if self.last_wall else None,
                  raw=raw, ext=ext, ours=ours, pos={s: self.position(s) for s in SYMS},
                  fair={s: round(v, 2) for s, v in self.fair.items()},
                  paused=state.batch < self.pause_until, blocked=state.blocked,
                  pnl=state.pnl_est_mills, kill=str(state.kill_switch) if hasattr(state, "kill_switch") else None)
        self.last_wall = wall
