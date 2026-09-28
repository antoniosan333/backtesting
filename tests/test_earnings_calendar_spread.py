"""Tests for the earnings_calendar_spread IV-premium isolation strategy."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd

from lambdaclass.strategies.base import OpenOptionView, StrategyContext


def _load_strategy():
    spec = importlib.util.spec_from_file_location(
        "earnings_calendar_spread",
        Path("strategies/2026-09/earnings_calendar_spread.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StrategyImpl()


def _dummy_chain(spot: float = 100.0) -> pd.DataFrame:
    """Chain with two expiries: front (2026-09-25, DTE 5) and back (2026-10-02, DTE 12).

    Bar date is 2026-09-20, so front DTE=5 >= min_front_dte(3) and back DTE=12 >= min_back_dte(10).
    """
    rows = []
    for expiry in ("2026-09-25", "2026-10-02"):
        for side in ("call", "put"):
            for strike in (95, 100, 105):
                rows.append(
                    {
                        "contract_symbol": f"TEST_{strike}{side[0].upper()}_{expiry}",
                        "side": side,
                        "strike": float(strike),
                        "expiry": expiry,
                        "implied_volatility": 0.35,
                        "bid": 3.0,
                        "ask": 3.2,
                        "underlying_last": spot,
                    }
                )
    return pd.DataFrame(rows)


def test_strategy_name_and_defaults():
    strat = _load_strategy()
    assert strat.name == "earnings_calendar_spread"
    assert strat.params["enter_days_before"] == 5
    assert strat.params["exit_days_after"] == 1
    assert strat.params["min_front_dte"] == 3
    assert strat.params["min_back_dte"] == 10


def test_calendar_spread_enters_before_earnings():
    """When days_to_next_earnings == 5, strategy sells front straddle + buys back straddle."""
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
    assert len(decision.option_legs) == 4
    sides = {leg.side for leg in decision.option_legs}
    assert sides == {"call", "put"}
    # Two short legs (front) and two long legs (back)
    short_legs = [leg for leg in decision.option_legs if leg.quantity < 0]
    long_legs = [leg for leg in decision.option_legs if leg.quantity > 0]
    assert len(short_legs) == 2
    assert len(long_legs) == 2
    # All legs at the ATM strike (100)
    assert all(leg.strike == 100.0 for leg in decision.option_legs)
    # Front expiry on short legs, back expiry on long legs
    assert all(leg.expiry == "2026-09-25" for leg in short_legs)
    assert all(leg.expiry == "2026-10-02" for leg in long_legs)
    assert decision.metadata.get("reason") == "earnings_entry"
    assert decision.metadata.get("days_to") == 5
    assert decision.metadata.get("front_expiry") == "2026-09-25"
    assert decision.metadata.get("back_expiry") == "2026-10-02"


def test_calendar_spread_does_not_enter_when_days_to_mismatch():
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


def test_calendar_spread_exits_after_earnings():
    """When open position exists and days_since_last_earnings >= 1, close legs."""
    strat = _load_strategy()
    open_options = {
        "TEST_100C_2026-09-25": OpenOptionView(
            side="call",
            strike=100.0,
            expiry=date(2026, 9, 25),
            quantity=-1,
            avg_entry_mid=3.1,
        ),
        "TEST_100P_2026-09-25": OpenOptionView(
            side="put",
            strike=100.0,
            expiry=date(2026, 9, 25),
            quantity=-1,
            avg_entry_mid=3.1,
        ),
        "TEST_100C_2026-10-02": OpenOptionView(
            side="call",
            strike=100.0,
            expiry=date(2026, 10, 2),
            quantity=1,
            avg_entry_mid=3.1,
        ),
        "TEST_100P_2026-10-02": OpenOptionView(
            side="put",
            strike=100.0,
            expiry=date(2026, 10, 2),
            quantity=1,
            avg_entry_mid=3.1,
        ),
    }
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-26", "close": 98.0}),
        cash=10000,
        position=0,
        days_since_last_earnings=1,
        open_options=open_options,
    )
    decision = strat.on_bar(ctx)
    assert len(decision.option_legs) == 4
    assert all(leg.reduce_only for leg in decision.option_legs)
    assert decision.metadata.get("reason") == "earnings_exit"


def test_calendar_spread_holds_before_exit_window():
    """Open position but days_since_last_earnings == 0 → hold, no close legs."""
    strat = _load_strategy()
    open_options = {
        "TEST_100C_2026-09-25": OpenOptionView(
            side="call",
            strike=100.0,
            expiry=date(2026, 9, 25),
            quantity=-1,
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


def test_calendar_spread_no_entry_without_two_expiries():
    """If the chain only has one expiry, front == back and strategy should not enter."""
    strat = _load_strategy()
    # Chain with a single expiry that satisfies min_back_dte but front==back
    rows = []
    for side in ("call", "put"):
        for strike in (95, 100, 105):
            rows.append(
                {
                    "contract_symbol": f"TEST_{strike}{side[0].upper()}",
                    "side": side,
                    "strike": float(strike),
                    "expiry": "2026-10-02",
                    "implied_volatility": 0.35,
                    "bid": 3.0,
                    "ask": 3.2,
                    "underlying_last": 100.0,
                }
            )
    single_chain = pd.DataFrame(rows)
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=5,
        options_chain=single_chain,
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []