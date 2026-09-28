"""Tests for the earnings_crush_short_straddle post-event IV-crush strategy."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd

from lambdaclass.strategies.base import OpenOptionView, StrategyContext


def _load_strategy():
    spec = importlib.util.spec_from_file_location(
        "earnings_crush_short_straddle",
        Path("strategies/2026-09/earnings_crush_short_straddle.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StrategyImpl()


def _dummy_chain(spot: float = 100.0) -> pd.DataFrame:
    """Minimal ATM options chain with call + put at several strikes."""
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
    assert strat.name == "earnings_crush_short_straddle"
    assert strat.params["enter_days_after"] == 1
    assert strat.params["hold_days"] == 5
    assert strat.params["min_dte"] == 7
    assert strat.params["min_iv_rank"] is None


def test_enters_after_earnings():
    """When days_since_last_earnings == enter_days_after (1), sell ATM straddle."""
    strat = _load_strategy()
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-22", "close": 100.0}),
        cash=10000,
        position=0,
        days_since_last_earnings=1,
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
    assert decision.metadata.get("reason") == "crush_entry"
    assert decision.metadata.get("days_since") == 1


def test_does_not_enter_when_days_since_mismatch():
    """enter_days_after == 1, so days_since_last_earnings == 3 should NOT enter."""
    strat = _load_strategy()
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-24", "close": 100.0}),
        cash=10000,
        position=0,
        days_since_last_earnings=3,
        options_chain=_dummy_chain(),
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []


def test_does_not_enter_when_days_since_is_none():
    """No earnings info → no entry."""
    strat = _load_strategy()
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-22", "close": 100.0}),
        cash=10000,
        position=0,
        days_since_last_earnings=None,
        options_chain=_dummy_chain(),
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []


def test_exits_after_hold_period():
    """Open position + days_since >= enter_days_after + hold_days → close legs."""
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
    # enter_days_after=1, hold_days=5 → exit at days_since >= 6
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-30", "close": 98.0}),
        cash=10000,
        position=0,
        days_since_last_earnings=6,
        open_options=open_options,
    )
    decision = strat.on_bar(ctx)
    assert len(decision.option_legs) == 2
    # Closing a short requires buying back → positive quantity
    assert all(leg.quantity > 0 for leg in decision.option_legs)
    assert all(leg.reduce_only for leg in decision.option_legs)
    assert decision.metadata.get("reason") == "crush_exit"
    assert decision.metadata.get("days_since") == 6


def test_holds_before_exit_window():
    """Open position but days_since < enter_days_after + hold_days → hold, no close."""
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
    # days_since=3 < 6 (enter_days_after + hold_days) → hold
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-26", "close": 100.0}),
        cash=10000,
        position=0,
        days_since_last_earnings=3,
        open_options=open_options,
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []


def test_iv_rank_filter_blocks_entry():
    """min_iv_rank set and iv_rank_252 below it → no entry, iv_rank_too_low."""
    strat = _load_strategy()
    strat.params = {**strat.params, "min_iv_rank": 0.5}
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-22", "close": 100.0}),
        cash=10000,
        position=0,
        days_since_last_earnings=1,
        iv_rank_252=0.3,
        options_chain=_dummy_chain(),
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []
    assert decision.metadata.get("reason") == "iv_rank_too_low"


def test_iv_rank_filter_allows_entry_when_above():
    """min_iv_rank set and iv_rank_252 above it → entry proceeds."""
    strat = _load_strategy()
    strat.params = {**strat.params, "min_iv_rank": 0.5}
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-22", "close": 100.0}),
        cash=10000,
        position=0,
        days_since_last_earnings=1,
        iv_rank_252=0.7,
        options_chain=_dummy_chain(),
    )
    decision = strat.on_bar(ctx)
    assert len(decision.option_legs) == 2
    assert decision.metadata.get("reason") == "crush_entry"


def test_no_chain_no_entry():
    """No options chain → no entry even if days_since matches."""
    strat = _load_strategy()
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-22", "close": 100.0}),
        cash=10000,
        position=0,
        days_since_last_earnings=1,
        options_chain=None,
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []