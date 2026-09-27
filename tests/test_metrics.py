from __future__ import annotations

import math

import pandas as pd
import pytest

from lambdaclass.reporting.metrics import (
    avg_holding_days,
    compute_metrics,
    num_trades,
    win_rate_per_trade,
)


def test_compute_metrics_known_answers() -> None:
    curve = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "equity": [100.0, 110.0, 99.0],
        }
    )

    metrics = compute_metrics(curve, risk_free_rate=0.0, periods_per_year=2)

    assert metrics["total_return"] == pytest.approx(-0.01)
    assert metrics["cagr"] == pytest.approx(-0.01)
    assert metrics["max_drawdown"] == pytest.approx(-0.10)
    assert metrics["annualized_volatility"] == pytest.approx(math.sqrt(2) * 0.10)
    assert metrics["sharpe"] == pytest.approx(0.0)
    assert metrics["sortino"] == pytest.approx(0.0)
    assert metrics["calmar"] == pytest.approx(-0.10)
    assert metrics["up_bar_ratio"] == pytest.approx(0.5)
    assert metrics["hit_rate"] == metrics["up_bar_ratio"]


def test_compute_metrics_subtracts_compounded_periodic_risk_free_rate() -> None:
    curve = pd.DataFrame({"equity": [100.0, 102.0, 102.0]})
    annual_rate = 1.01**12 - 1.0

    metrics = compute_metrics(curve, risk_free_rate=annual_rate, periods_per_year=12)

    assert metrics["sharpe"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["sortino"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("dates", "expected_periods"),
    [
        (["2026-01-05", "2026-01-06", "2026-01-07"], 252),
        (["2026-01-05", "2026-01-12", "2026-01-19"], 52),
        (["2026-01-31", "2026-02-28", "2026-03-31"], 12),
    ],
)
def test_compute_metrics_infers_periods_from_median_date_spacing(
    dates: list[str], expected_periods: int
) -> None:
    curve = pd.DataFrame({"date": dates, "equity": [100.0, 110.0, 99.0]})

    inferred = compute_metrics(curve, risk_free_rate=0.0)
    explicit = compute_metrics(curve, risk_free_rate=0.0, periods_per_year=expected_periods)

    assert inferred == pytest.approx(explicit)


@pytest.mark.parametrize(
    "curve",
    [
        pd.DataFrame(),
        pd.DataFrame({"other": [1.0]}),
        pd.DataFrame({"equity": []}),
        pd.DataFrame({"equity": [0.0, 0.0]}),
        pd.DataFrame({"equity": [100.0, 0.0, -10.0]}),
        pd.DataFrame({"equity": [-100.0, -90.0]}),
    ],
)
def test_compute_metrics_is_finite_for_empty_zero_and_negative_equity(
    curve: pd.DataFrame,
) -> None:
    metrics = compute_metrics(curve, risk_free_rate=0.04)

    assert set(metrics) == {
        "cagr",
        "max_drawdown",
        "sharpe",
        "hit_rate",
        "up_bar_ratio",
        "total_return",
        "annualized_volatility",
        "sortino",
        "calmar",
    }
    assert all(math.isfinite(value) for value in metrics.values())


def test_stock_trade_statistics_are_shared_metrics_helpers() -> None:
    trades = pd.DataFrame(
        [
            {"date": "2026-01-02", "action": "buy", "price": 100.0},
            {"date": "2026-01-05", "action": "sell", "price": 110.0},
            {"date": "2026-01-10", "action": "buy", "price": 120.0},
            {"date": "2026-01-12", "action": "sell", "price": 110.0},
        ]
    )

    assert num_trades(trades) == 4
    assert avg_holding_days(trades) == 2.5
    assert win_rate_per_trade(trades) == 0.5
