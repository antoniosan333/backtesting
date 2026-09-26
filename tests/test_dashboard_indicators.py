from __future__ import annotations

import pandas as pd
import pytest

from lambdaclass.reporting.dashboard import indicators


def test_sma_basic() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = indicators.sma(s, window=3)
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(3.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_recursive_matches_known_seed() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = indicators.ema(s, window=2)
    assert pd.isna(out.iloc[0])
    alpha = 2.0 / (2 + 1)
    expected_seed = alpha * 2.0 + (1 - alpha) * 1.0
    assert out.iloc[1] == pytest.approx(expected_seed)


def test_rsi_all_gains_is_100() -> None:
    s = pd.Series([float(i) for i in range(1, 30)])
    out = indicators.rsi(s, window=14)
    assert out.iloc[-1] == pytest.approx(100.0)


def test_bollinger_widths_symmetric() -> None:
    s = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
    bands = indicators.bollinger(s, window=5, num_std=2.0)
    last_mid = bands.middle.iloc[-1]
    assert bands.upper.iloc[-1] - last_mid == pytest.approx(last_mid - bands.lower.iloc[-1])


def test_equity_drawdown_zero_then_negative() -> None:
    eq = pd.Series([100.0, 110.0, 120.0, 90.0, 100.0])
    dd = indicators.equity_drawdown(eq)
    assert dd.iloc[0] == pytest.approx(0.0)
    assert dd.iloc[2] == pytest.approx(0.0)
    assert dd.iloc[3] == pytest.approx(-0.25)
    assert dd.iloc[4] == pytest.approx((100.0 - 120.0) / 120.0)


def test_rolling_sharpe_zero_volatility_is_nan() -> None:
    eq = pd.Series([100.0] * 30)
    rs = indicators.rolling_sharpe(eq, window=10)
    assert pd.isna(rs.iloc[-1])


def test_monthly_returns_table_handles_two_months() -> None:
    dates = pd.date_range("2024-01-01", "2024-02-29", freq="B")
    eq_values = [100.0 + i for i in range(len(dates))]
    df = pd.DataFrame({"date": dates.astype(str), "equity": eq_values})
    table = indicators.monthly_returns_table(df)
    assert not table.empty
    assert 1 in table.columns and 2 in table.columns


def test_monthly_returns_table_empty_input() -> None:
    table = indicators.monthly_returns_table(pd.DataFrame(columns=["date", "equity"]))
    assert table.empty


def test_derived_run_stats_merges_metrics() -> None:
    metrics = {"total_return": 0.1, "sharpe": 1.2}
    out = indicators.derived_run_stats(metrics, pd.DataFrame(), 3, 2.5, 0.5)
    assert out["total_return"] == pytest.approx(0.1)
    assert out["sharpe"] == pytest.approx(1.2)
    assert out["num_trades"] == 3.0
    assert out["avg_holding_days"] == 2.5
    assert out["win_rate_per_trade"] == 0.5


def test_invalid_windows_raise() -> None:
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        indicators.sma(s, window=0)
    with pytest.raises(ValueError):
        indicators.ema(s, window=0)
    with pytest.raises(ValueError):
        indicators.rsi(s, window=0)
    with pytest.raises(ValueError):
        indicators.rolling_sharpe(s, window=1)
