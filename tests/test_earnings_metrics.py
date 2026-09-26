"""Per-event earnings metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from lambdaclass.reporting.earnings_metrics import (
    average_iv_from_option_trades,
    compute_earnings_events,
    summarize_earnings_events,
)


def test_compute_earnings_events_straddle_roundtrip() -> None:
    bars = pd.DataFrame(
        {
            "date": ["2026-01-20", "2026-01-25", "2026-01-26"],
            "close": [100.0, 100.0, 110.0],
        }
    )
    earnings = pd.DataFrame(
        {
            "symbol": ["ZZZ"],
            "earnings_date": ["2026-01-25"],
            "timing": ["AMC"],
            "source": ["test"],
            "fetched_at": ["x"],
        }
    )
    # Long straddle: pay 800 on open, receive 1200 on close → pnl +400
    option_trades = pd.DataFrame(
        [
            {
                "date": "2026-01-20",
                "action": "open",
                "contract_symbol": "C",
                "side": "call",
                "strike": 100.0,
                "expiry": "2026-02-20",
                "quantity": 1,
                "mid_price": 4.0,
                "iv": 0.40,
                "premium": 400.0,
                "commission": 0.0,
                "cash_after": 99600.0,
            },
            {
                "date": "2026-01-20",
                "action": "open",
                "contract_symbol": "P",
                "side": "put",
                "strike": 100.0,
                "expiry": "2026-02-20",
                "quantity": 1,
                "mid_price": 4.0,
                "iv": 0.40,
                "premium": 400.0,
                "commission": 0.0,
                "cash_after": 99200.0,
            },
            {
                "date": "2026-01-26",
                "action": "close",
                "contract_symbol": "C",
                "side": "call",
                "strike": 100.0,
                "expiry": "2026-02-20",
                "quantity": -1,
                "mid_price": 8.0,
                "iv": 0.25,
                "premium": -800.0,
                "commission": 0.0,
                "cash_after": 100000.0,
            },
            {
                "date": "2026-01-26",
                "action": "close",
                "contract_symbol": "P",
                "side": "put",
                "strike": 100.0,
                "expiry": "2026-02-20",
                "quantity": -1,
                "mid_price": 1.0,
                "iv": 0.25,
                "premium": -100.0,
                "commission": 0.0,
                "cash_after": 100100.0,
            },
        ]
    )
    iv_map = average_iv_from_option_trades(option_trades)
    expected_moves = pd.DataFrame(
        [
            {
                "asof": "2026-01-20",
                "expiry": "2026-02-20",
                "straddle": 7.0,
                "iv_1sd": 9.0,
            }
        ]
    )
    events = compute_earnings_events(
        option_trades=option_trades,
        earnings=earnings,
        bars=bars,
        iv_by_date=iv_map,
        expected_moves=expected_moves,
    )
    assert len(events) == 1
    row = events.iloc[0]
    assert row["earnings_date"] == "2026-01-25"
    assert row["realized_move_pct"] == pytest.approx(0.10, abs=1e-6)
    assert row["implied_move_pct"] == pytest.approx(0.07, abs=1e-6)
    assert row["implied_move_1sd_pct"] == pytest.approx(0.09, abs=1e-6)
    assert row["iv_crush"] == pytest.approx(0.15, abs=1e-6)
    assert row["event_pnl"] == pytest.approx(100.0, abs=1.0)
    assert bool(row["beat_implied"]) is True
    summary = summarize_earnings_events(events)
    assert summary["earnings_event_count"] == 1.0
    assert summary["earnings_total_pnl"] == pytest.approx(100.0, abs=1.0)


def test_engine_passes_earnings_context() -> None:
    from lambdaclass.backtest.engine import run_backtest
    from lambdaclass.config import DEFAULT_PREFERENCES
    from lambdaclass.strategies.base import Strategy, StrategyContext, StrategyDecision

    class Probe(Strategy):
        name = "probe"
        params: dict = {}

        def __init__(self) -> None:
            self.seen: list[int | None] = []

        def on_bar(self, context: StrategyContext) -> StrategyDecision:
            self.seen.append(context.days_to_next_earnings)
            return StrategyDecision(action="hold")

    bars = pd.DataFrame(
        {
            "date": ["2026-01-20", "2026-01-21"],
            "open": [100, 100],
            "high": [100, 100],
            "low": [100, 100],
            "close": [100.0, 100.0],
            "volume": [1, 1],
        }
    )
    earnings = pd.DataFrame(
        {
            "symbol": ["X"],
            "earnings_date": ["2026-01-25"],
            "timing": ["AMC"],
            "source": ["t"],
            "fetched_at": ["x"],
        }
    )
    probe = Probe()
    run_backtest(probe, bars, pd.DataFrame(), DEFAULT_PREFERENCES, earnings=earnings)
    assert probe.seen[0] == 5
    assert probe.seen[1] == 4
