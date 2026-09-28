"""Tests for the earnings_long_strangle strategy."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd

from lambdaclass.strategies.base import OpenOptionView, StrategyContext


def _load_strategy():
    spec = importlib.util.spec_from_file_location(
        "earnings_long_strangle",
        Path("strategies/2026-09/earnings_long_strangle.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StrategyImpl()


def _dummy_chain(spot: float = 100.0) -> pd.DataFrame:
    """Minimal options chain with calls and puts at 95/100/105 strikes."""
    rows = []
    for side in ("call", "put"):
        for strike in (90, 95, 100, 105, 110):
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
    assert strat.name == "earnings_long_strangle"
    assert strat.params["enter_days_before"] == 5
    assert strat.params["exit_days_after"] == 1
    assert strat.params["min_dte"] == 7
    assert strat.params["offset"] == 5.0


def test_long_strangle_enters_before_earnings():
    """When days_to_next_earnings == 5, strategy buys an OTM strangle."""
    strat = _load_strategy()
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=5,
        next_earnings_date="2026-09-25",
        earnings_timing="AMC",
        options_chain=_dummy_chain(),
    )
    decision = strat.on_bar(ctx)
    assert len(decision.option_legs) == 2
    sides = {leg.side for leg in decision.option_legs}
    assert sides == {"call", "put"}
    # Long legs — quantity must be positive
    assert all(leg.quantity > 0 for leg in decision.option_legs)
    # OTM strangle: put strike below spot, call strike above spot
    put_leg = next(leg for leg in decision.option_legs if leg.side == "put")
    call_leg = next(leg for leg in decision.option_legs if leg.side == "call")
    assert put_leg.strike == 95.0  # spot - offset = 100 - 5
    assert call_leg.strike == 105.0  # spot + offset = 100 + 5
    assert decision.metadata.get("reason") == "earnings_entry"
    assert decision.metadata.get("days_to") == 5


def test_long_strangle_does_not_enter_when_days_to_mismatch():
    """enter_days_before == 5, so days_to_next_earnings == 3 should NOT enter."""
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


def test_long_strangle_exits_after_earnings():
    """When open position exists and days_since_last_earnings >= 1, close legs."""
    strat = _load_strategy()
    open_options = {
        "TEST_105C": OpenOptionView(
            side="call",
            strike=105.0,
            expiry=date(2026, 10, 17),
            quantity=1,
            avg_entry_mid=3.1,
        ),
        "TEST_95P": OpenOptionView(
            side="put",
            strike=95.0,
            expiry=date(2026, 10, 17),
            quantity=1,
            avg_entry_mid=3.1,
        ),
    }
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-26", "close": 102.0}),
        cash=10000,
        position=0,
        days_since_last_earnings=1,
        open_options=open_options,
    )
    decision = strat.on_bar(ctx)
    assert len(decision.option_legs) == 2
    # Closing a long requires selling → negative quantity
    assert all(leg.quantity < 0 for leg in decision.option_legs)
    assert all(leg.reduce_only for leg in decision.option_legs)
    assert decision.metadata.get("reason") == "earnings_exit"


def test_long_strangle_holds_before_exit_window():
    """Open position but days_since_last_earnings == 0 → hold, no close legs."""
    strat = _load_strategy()
    open_options = {
        "TEST_105C": OpenOptionView(
            side="call",
            strike=105.0,
            expiry=date(2026, 10, 17),
            quantity=1,
            avg_entry_mid=3.1,
        ),
    }
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-25", "close": 100.0}),
        cash=10000,
        position=0,
        days_since_last_earnings=0,
        open_options=open_options,
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []


def test_long_strangle_holds_without_chain():
    """No options chain → hold."""
    strat = _load_strategy()
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=5,
        options_chain=None,
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []