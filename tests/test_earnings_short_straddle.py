"""Tests for the earnings_short_straddle IV-crush strategy."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd

from lambdaclass.strategies.base import OpenOptionView, StrategyContext


def _load_strategy():
    spec = importlib.util.spec_from_file_location(
        "earnings_short_straddle",
        Path("strategies/2026-09/earnings_short_straddle.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StrategyImpl()


def _dummy_chain(spot: float = 100.0) -> pd.DataFrame:
    """Minimal ATM options chain with call + put at the spot strike."""
    rows = []
    for side in ("call", "put"):
        for strike in (95, 100, 105):
            rows.append(
                {
                    "contract_symbol": f"TEST_{strike}{side[0].upper()}",
                    "side": side,
                    "strike": float(strike),
                    "expiry": "2026-10-17",
                    "implied_volatility": 0.35,
                    "bid": 3.0,
                    "ask": 3.2,
                    "underlying_last": spot,
                }
            )
    return pd.DataFrame(rows)


def test_strategy_name_and_defaults():
    strat = _load_strategy()
    assert strat.name == "earnings_short_straddle"
    assert strat.params["enter_days_before"] == 1
    assert strat.params["exit_days_after"] == 1
    assert strat.params["min_dte"] == 7


def test_short_straddle_enters_before_earnings():
    """When days_to_next_earnings == 1, strategy sells an ATM straddle."""
    strat = _load_strategy()
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=1,
        next_earnings_date="2026-09-21",
        earnings_timing="AMC",
        options_chain=_dummy_chain(),
    )
    decision = strat.on_bar(ctx)
    assert len(decision.option_legs) == 2
    sides = {leg.side for leg in decision.option_legs}
    assert sides == {"call", "put"}
    # Short legs — quantity must be negative
    assert all(leg.quantity < 0 for leg in decision.option_legs)
    # Both legs at the ATM strike (100)
    assert all(leg.strike == 100.0 for leg in decision.option_legs)
    assert decision.metadata.get("reason") == "earnings_entry"
    assert decision.metadata.get("days_to") == 1


def test_short_straddle_does_not_enter_when_days_to_mismatch():
    """enter_days_before == 1, so days_to_next_earnings == 3 should NOT enter."""
    strat = _load_strategy()
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=3,
        options_chain=_dummy_chain(),
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []


def test_short_straddle_exits_after_earnings():
    """When open position exists and days_since_last_earnings >= 1, close legs."""
    strat = _load_strategy()
    open_options = {
        "TEST_100C": OpenOptionView(
            side="call",
            strike=100.0,
            expiry=date(2026, 10, 17),
            quantity=-1,
            avg_entry_mid=3.1,
        ),
        "TEST_100P": OpenOptionView(
            side="put",
            strike=100.0,
            expiry=date(2026, 10, 17),
            quantity=-1,
            avg_entry_mid=3.1,
        ),
    }
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-22", "close": 98.0}),
        cash=10000,
        position=0,
        days_since_last_earnings=1,
        open_options=open_options,
    )
    decision = strat.on_bar(ctx)
    assert len(decision.option_legs) == 2
    # Closing a short requires buying back → positive quantity
    assert all(leg.quantity > 0 for leg in decision.option_legs)
    assert all(leg.reduce_only for leg in decision.option_legs)
    assert decision.metadata.get("reason") == "earnings_exit"


def test_short_straddle_holds_before_exit_window():
    """Open position but days_since_last_earnings == 0 → hold, no close legs."""
    strat = _load_strategy()
    open_options = {
        "TEST_100C": OpenOptionView(
            side="call",
            strike=100.0,
            expiry=date(2026, 10, 17),
            quantity=-1,
            avg_entry_mid=3.1,
        ),
    }
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-21", "close": 100.0}),
        cash=10000,
        position=0,
        days_since_last_earnings=0,
        open_options=open_options,
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []