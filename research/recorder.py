"""Record a live session as a backtest data set: sends no orders, writes the top of book every
batch and every trade, in the historical format (DATA.md), to research/sandbox_data/.

    pqt run research/recorder.py --mode test --server https://sandbox.princeton-quant.com --key $PQT_API_KEY_ALGO2

Run it alone in a session (no strategy of ours trading) so the book holds no orders of ours. Start it
before the session: it writes <session>.top.csv / .trades.csv as batches arrive; at the end it writes
manifest.json listing every complete recording. Then: load_session("research/sandbox_data", name).
"""

import json
import os
from pathlib import Path

from pqt_client.v2 import BatchState, Strategy

OUT = Path(__file__).resolve().parent / "sandbox_data"


class Recorder(Strategy):
    def on_start(self) -> None:
        OUT.mkdir(exist_ok=True)
        name = getattr(self.session, "session_id", None) or str(self.session)
        self.name = str(name).replace("/", "_")
        self.top = open(OUT / f"{self.name}.top.csv", "w")
        self.trades = open(OUT / f"{self.name}.trades.csv", "w")
        self.top.write("batch,symbol,bid_px,bid_qty,ask_px,ask_qty,status\n")
        self.trades.write("batch,symbol,price,qty,aggressor,self_trade,aux\n")
        self.first = None
        self.last = None
        self.log.info("recording %s to %s", self.name, OUT)

    def on_batch(self, state: BatchState) -> None:
        b = state.batch
        if self.first is None:
            self.first = b
        self.last = b
        for s in ("HPPO", "BNKY", "CRRT", "BRRY"):
            t = state.top[s]
            cell = lambda v: "" if v is None else str(v)  # noqa: E731
            status = getattr(t, "status", None)
            status = getattr(status, "value", status) or ("HALTED" if t.halted else "TRADING")
            self.top.write(f"{b},{s},{cell(t.bid_px)},{cell(t.bid_qty)},{cell(t.ask_px)},{cell(t.ask_qty)},{status}\n")
        for tr in state.trades:
            agg = getattr(tr.aggressor, "value", tr.aggressor)
            aux = json.dumps(dict(tr.aux or {})).replace('"', '""')
            self.trades.write(f'{tr.batch},{tr.symbol},{tr.price},{tr.qty},{agg},'
                              f'{"true" if tr.self_trade else "false"},"{aux}"\n')
        if b % 120 == 0:
            self.top.flush()
            self.trades.flush()

    def on_end(self, state: BatchState) -> None:
        self.top.close()
        self.trades.close()
        complete = self.first == 0 and self.last == state.n_batches - 1
        self.log.info("recorded batches %s..%s (%s)", self.first, self.last, "complete" if complete else "PARTIAL")
        man = json.load(open(Path(__file__).resolve().parent.parent / "data" / "manifest.json"))
        done = OUT / "complete.json"
        names = json.load(open(done)) if done.exists() else []
        if complete and self.name not in names:
            names.append(self.name)
        json.dump(names, open(done, "w"))
        man["sessions"] = names
        json.dump(man, open(OUT / "manifest.json", "w"), indent=2)
