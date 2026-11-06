"""Evaluate alpha.py with deliberately harsher resting-quote fills.

Run from ``research/`` with ``../.venv/bin/python stress.py``.
The base scenario preserves the SDK's strict-through fill rule. Other scenarios
require deeper prints, haircut trade size, and/or add adverse at-price fills.
"""
import json
import random
import sys
from dataclasses import dataclass

import pandas as pd
from pqt_client.backtest_v2 import _Replay

from bt import load_cls, run
from cv import FOLDS, pick


@dataclass(frozen=True)
class Scenario:
    name: str
    through: int = 1
    share: float = 1.0
    adverse: bool = False


SCENARIOS = (
    Scenario("base"),
    Scenario("through=2", through=2),
    Scenario("share=0.3", share=0.3),
    Scenario("share=0.1", share=0.1),
    Scenario("share=0.3+through=2", through=2, share=0.3),
    Scenario("adverse", adverse=True),
    Scenario(
        "share=0.3+through=2+adverse",
        through=2,
        share=0.3,
        adverse=True,
    ),
    Scenario(
        "all combined (share=0.1)",
        through=2,
        share=0.1,
        adverse=True,
    ),
)
REALISTIC = SCENARIOS[6]
TUNING_CONFIGS = (
    ("current", {}, {}),
    ("make_edge=2.0", {"make_edge": 2.0}, {}),
    ("make_edge=3.0", {"make_edge": 3.0}, {}),
    ("size_scale=1.0", {"size_scale": 1.0}, {}),
    ("size_scale=2.0", {"size_scale": 2.0}, {}),
    (
        "narrower BNKY/CRRT edges",
        {},
        {"EDGE_MULT": {"HPPO": 1.0, "BNKY": 0.4, "CRRT": 0.5, "BRRY": 1.0}},
    ),
    (
        "wider BNKY/CRRT edges",
        {},
        {"EDGE_MULT": {"HPPO": 1.0, "BNKY": 0.7, "CRRT": 0.8, "BRRY": 1.0}},
    ),
    (
        "stronger inventory skew",
        {},
        {"SKEW": {"HPPO": 4.5, "BNKY": 1.5, "CRRT": 2.25, "BRRY": 4.5}},
    ),
    (
        "stronger HPPO/BRRY skew",
        {},
        {"SKEW": {"HPPO": 4.5, "BNKY": 1.0, "CRRT": 1.5, "BRRY": 4.5}},
    ),
    (
        "stronger BNKY/CRRT skew",
        {},
        {"SKEW": {"HPPO": 3.0, "BNKY": 1.5, "CRRT": 2.25, "BRRY": 3.0}},
    ),
    (
        "stronger skew with narrower BNKY/CRRT",
        {},
        {
            "SKEW": {"HPPO": 4.5, "BNKY": 1.5, "CRRT": 2.25, "BRRY": 4.5},
            "EDGE_MULT": {"HPPO": 1.0, "BNKY": 0.4, "CRRT": 0.5, "BRRY": 1.0},
        },
    ),
)


def stressed_rest_fills(self, trades, *, scenario, seed):
    """Replacement for _Replay.rest_fills with deterministic stress knobs."""
    if self.batch == 0:
        session_number = int(self.session.name[1:]) if self.session.name[1:].isdigit() else 0
        self._stress_rng = random.Random(seed + session_number)

    for trade in trades:
        exact_available = trade.qty * scenario.share
        available = int(exact_available)
        if exact_available > available and self._stress_rng.random() < exact_available - available:
            available += 1
        left = available
        for order in sorted(
            (
                order
                for order in self.resting.values()
                if order.symbol == trade.symbol and order.placed < self.batch
            ),
            key=lambda order: (
                -order.price if order.side == "BUY" else order.price,
                order.seq,
            ),
        ):
            if left <= 0 or order.order_id not in self.resting:
                continue
            if order.side == "BUY":
                through = trade.price <= order.price - scenario.through
                adverse_at_price = False
                if scenario.adverse and trade.price == order.price:
                    mids = _stress_mids(self, trade.batch, trade.symbol)
                    adverse_at_price = mids is not None and mids[1] < mids[0]
            else:
                through = trade.price >= order.price + scenario.through
                adverse_at_price = False
                if scenario.adverse and trade.price == order.price:
                    mids = _stress_mids(self, trade.batch, trade.symbol)
                    adverse_at_price = mids is not None and mids[1] > mids[0]
            if not (through or adverse_at_price):
                continue
            quantity = min(left, order.remaining)
            left -= quantity
            order.remaining -= quantity
            fee = -self.rules.maker_rebate_mills * quantity
            self._fill(order, order.price, quantity, "MAKER", fee)
            if order.remaining == 0:
                del self.resting[order.order_id]
        self._check_kill_switch()


def _stress_mids(self, batch, symbol):
    """Return current and 120-batch mids when both observed books are two-sided."""
    current_top = self.session.tops[batch].get(symbol)
    future = min(batch + 120, self.session.n_batches - 1)
    future_top = self.session.tops[future].get(symbol)
    if (
        current_top is None
        or current_top.bid_px is None
        or current_top.ask_px is None
        or future_top is None
        or future_top.bid_px is None
        or future_top.ask_px is None
    ):
        return None
    return (
        (current_top.bid_px + current_top.ask_px) / 2,
        (future_top.bid_px + future_top.ask_px) / 2,
    )


def _install(scenario, seed):
    original = _Replay.rest_fills
    _Replay.rest_fills = lambda replay, trades: stressed_rest_fills(
        replay, trades, scenario=scenario, seed=seed
    )
    return original


def evaluate_scenario(cls, scenario, seed=2026):
    summaries, fills = [], []
    for fold_number, (coef_file, session_range) in enumerate(FOLDS, start=1):
        sessions = pick(session_range)
        with open(coef_file) as file:
            model = json.load(file)
        original = _install(scenario, seed)
        try:
            for session in sessions:
                summary, trades = run(
                    cls, [session], model
                )
                summaries.append(summary)
                if not trades.empty:
                    fills.append(trades)
        finally:
            _Replay.rest_fills = original
    return pd.concat(summaries), pd.concat(fills) if fills else pd.DataFrame()


def evaluate_sessions(cls, scenario, names, model, over=None, attrs=None, seed=2026):
    summaries, fills = [], []
    original = _install(scenario, seed)
    try:
        for name in names:
            summary, trades = run(
                cls, [name], model, over=over, attrs=attrs
            )
            summaries.append(summary)
            if not trades.empty:
                fills.append(trades)
    finally:
        _Replay.rest_fills = original
    return pd.concat(summaries), pd.concat(fills) if fills else pd.DataFrame()


def report(scenario, summaries, fills):
    session_count = len(summaries)
    print(
        f"{scenario.name}: ${summaries.mk120.mean():,.0f}/session "
        f"(worst ${summaries.mk120.min():,.0f}; "
        f"{(summaries.mk120 > 0).sum()}/{session_count} positive), "
        f"{summaries.shares.mean():,.0f} shares/session"
    )
    if fills.empty:
        return
    grouped = fills.groupby(["sym", "liq"]).agg(
        shares=("q", lambda quantity: quantity.abs().sum()),
        markout_cents=("mk120", "sum"),
    )
    grouped["cents_per_share"] = grouped.markout_cents / grouped.shares
    grouped["dollars_per_session"] = grouped.markout_cents / 100 / session_count
    print(
        grouped[["shares", "cents_per_share", "dollars_per_session"]]
        .round(2)
        .to_string()
    )


def tune_and_holdout(cls):
    with open("fold1.json") as file:
        tuning_model = json.load(file)
    with open("fold3.json") as file:
        holdout_model = json.load(file)
    tuning_sessions = pick("s01-s08")
    holdout_sessions = pick("s17-s24")
    tuning_results = []

    print("Tuning on s01-s08; final holdout is s17-s24.")
    for name, over, attrs in TUNING_CONFIGS:
        summaries, _ = evaluate_sessions(
            cls, REALISTIC, tuning_sessions, tuning_model, over, attrs
        )
        result = {
            "name": name,
            "over": over,
            "attrs": attrs,
            "mean": summaries.mk120.mean(),
            "worst": summaries.mk120.min(),
            "shares": summaries.shares.mean(),
        }
        tuning_results.append(result)
        print(
            f"{name}: ${result['mean']:,.0f}/session "
            f"(worst ${result['worst']:,.0f}, "
            f"{result['shares']:,.0f} shares/session)",
            flush=True,
        )

    selected = max(tuning_results, key=lambda item: (item["worst"], item["mean"]))
    print(f"\nSelected on tuning data: {selected['name']}")
    for name, over, attrs in (
        ("before", {}, {}),
        ("after", selected["over"], selected["attrs"]),
    ):
        for scenario in (SCENARIOS[0], REALISTIC):
            summaries, fills = evaluate_sessions(
                cls, scenario, holdout_sessions, holdout_model, over, attrs
            )
            print(f"\n{name}, {scenario.name}, s17-s24:")
            report(scenario, summaries, fills)


def validate_selected(cls):
    configs = (
        ("before", {}, {}),
        (
            "after",
            {},
            {"SKEW": {"HPPO": 4.5, "BNKY": 1.5, "CRRT": 2.25, "BRRY": 4.5}},
        ),
    )
    with open("fold2.json") as file:
        model = json.load(file)
    test_sessions = pick("s09-s16")
    for label, over, attrs in configs:
        for scenario in (SCENARIOS[0], REALISTIC):
            summaries, fills = evaluate_sessions(
                cls, scenario, test_sessions, model, over, attrs
            )
            print(f"\n{label}, {scenario.name}, independent s09-s16 check:")
            report(scenario, summaries, fills)


def main():
    cls = load_cls("../strategies/alpha.py")
    if sys.argv[1:] == ["tune"]:
        tune_and_holdout(cls)
        return
    if sys.argv[1:] == ["validate"]:
        validate_selected(cls)
        return
    selected = set(sys.argv[1:])
    scenarios = [item for item in SCENARIOS if not selected or item.name in selected]
    unknown = selected - {item.name for item in SCENARIOS}
    if unknown:
        raise SystemExit(f"unknown scenarios: {', '.join(sorted(unknown))}")
    results = []
    for scenario in scenarios:
        summaries, fills = evaluate_scenario(cls, scenario)
        report(scenario, summaries, fills)
        results.append(
            {
                "scenario": scenario.name,
                "mean_$120": summaries.mk120.mean(),
                "worst_$120": summaries.mk120.min(),
                "positive": int((summaries.mk120 > 0).sum()),
            }
        )
    print("\nScenario comparison:")
    print(pd.DataFrame(results).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
