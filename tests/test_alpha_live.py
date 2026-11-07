"""Unit tests for alpha.py's live-only paths (the backtest never runs them).
    .venv/bin/python -m pytest -q tests   (or: .venv/bin/python tests/test_alpha_live.py)"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "strategies"))
from alpha import SYMS, Alpha, Ext  # noqa: E402


def order(side, price, qty=50, oid=1):
    return NS(side=side, price=price, remaining=qty, order_id=oid)


class Fake(Alpha):
    """Alpha with the SDK replaced: open orders and sent messages are plain lists."""

    def __init__(self, orders=(), msgs_left=20):
        self._orders = list(orders)
        self.sent = []
        self._left = msgs_left
        self.mode = "test"
        self.ext = {}
        self.symbol_info = {s: NS(max_order_qty=q, position_limit=1000) for s, q in
                            {"HPPO": 200, "BNKY": 500, "CRRT": 500, "BRRY": 120}.items()}
        self.params_override = {p.name: p.default for p in self.MODEL_PARAMS}

    def open_orders(self, symbol=None):
        return tuple(o for o in self._orders if symbol is None or getattr(o, "symbol", symbol) == symbol)

    def messages_left(self):
        return self._left - len(self.sent)

    def position(self, s):
        return 0

    def buy(self, s, q, px, ioc=False):
        self.sent.append(("BUY", s, q, px))

    def sell(self, s, q, px, ioc=False):
        self.sent.append(("SELL", s, q, px))

    def cancel(self, order_id=None, client_id=None):
        self.sent.append(("CANCEL", order_id))


def top(bid, bq, ask, aq):
    return NS(two_sided=True, bid_px=bid, bid_qty=bq, ask_px=ask, ask_qty=aq, halted=False)


def test_self_trade_guard():
    a = Fake([order("SELL", 100)])
    assert a.own_at_or_through("X", "SELL", 100)
    assert a.own_at_or_through("X", "SELL", 101)
    assert not a.own_at_or_through("X", "SELL", 99)
    assert not a.own_at_or_through("X", "BUY", 100)


def test_external_joined_and_alone():
    a = Fake([order("BUY", 99, 30), order("SELL", 101, 50)])
    e = a.external("X", top(99, 80, 101, 50))  # joined on the bid, alone on the ask
    assert (e.bid, e.bq, e.guessed_bid) == (99, 50, False)
    assert (e.ask, e.aq, e.guessed_ask) == (102, 100, True)


def test_arb_skips_guessed_leg():
    a = Fake()
    a.ext = {"HPPO": Ext(20010, 50, 20011, 50), "BNKY": Ext(9990, 50, 9991, 50),
             "BRRY": Ext(4994, 100, 4995, 100, guessed_ask=True), "CRRT": Ext(3000, 50, 3001, 50)}
    # sell ETF at 20010 vs basket 9991 + 2*4995 = 19981: gap 29c, far beyond cost
    state = NS(batch=100, top={s: NS(halted=False, two_sided=True) for s in SYMS})
    a.cool_until = {s: 0 for s in SYMS}
    a.CAP = Alpha.CAP
    lw = {s: 0 for s in SYMS}
    sw = dict(lw)
    a.room = lambda *args: 1000
    assert a.trade_arb(state, lw, sw) == set()
    a.ext["BRRY"].guessed_ask = False
    assert a.trade_arb(state, lw, sw) == {"HPPO", "BNKY", "BRRY"}
    assert sum(1 for m in a.sent if m[0] in ("BUY", "SELL")) == 3


def test_arb_counts_split_brry_orders():
    a = Fake(msgs_left=4)  # 50 units need HPPO 1 + BNKY 1 + BRRY 100 shares (1 order) = 3: fits
    a.ext = {"HPPO": Ext(20010, 50, 20011, 50), "BNKY": Ext(9990, 50, 9991, 50),
             "BRRY": Ext(4994, 300, 4995, 300), "CRRT": Ext(3000, 50, 3001, 50)}
    state = NS(batch=100, top={s: NS(halted=False, two_sided=True) for s in SYMS})
    a.cool_until = {s: 0 for s in SYMS}
    a.room = lambda *args: 1000
    lw = {s: 0 for s in SYMS}
    assert a.trade_arb(state, lw, dict(lw)) != set()
    b = Fake(msgs_left=4)  # 150 units: BRRY 300 shares = 3 orders, 5 in all: must not send
    b.ext = {**a.ext, "HPPO": Ext(20010, 150, 20011, 150), "BNKY": Ext(9990, 150, 9991, 150)}
    b.cool_until, b.room = a.cool_until, a.room
    assert b.trade_arb(state, lw, dict(lw)) == set()
    assert b.sent == []





def test_read_news():
    from alpha import read_news
    assert read_news("BNKY beats estimates, record quarter") == {"BNKY": (1, 1.0), "HPPO": (1, 1.0)}
    assert read_news("BRRY recalls product") == {"BRRY": (-1, 0.75), "HPPO": (-1, 0.75)}
    assert read_news("CRRT wins contract") == {"CRRT": (1, 0.75)}  # CRRT is not in the ETF
    assert read_news("CRRT denies fraud allegations") == {}  # negation: unclear
    assert read_news("CRRT shares jump but outlook weak") == {}  # both directions
    assert read_news("BNKY and BRRY to merge") == {}  # two stocks
    assert read_news("HPPO rebalances") == {}  # the ETF alone
    # real items seen on the sandbox (2026-11-07)
    assert read_news("Sandbox: routine notice", "This is the sandbox. Its other participants and its "
                     "prices are placeholders; news here moves nothing.") == {}
    assert read_news("Sandbox: another routine notice",
                     "Nothing learned on the sandbox transfers to the competition's market.") == {}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
