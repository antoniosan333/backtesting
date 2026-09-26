from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lambdaclass.reporting.dashboard import charts


def _bars(n: int = 5) -> pd.DataFrame:
    dates = [f"2024-01-{i:02d}" for i in range(2, 2 + n)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1000.0] * n,
        }
    )


def _equity(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [f"2024-01-{i:02d}" for i in range(2, 2 + n)],
            "equity": [100000.0 + 100.0 * i for i in range(n)],
        }
    )


def test_equity_with_drawdown_returns_two_traces() -> None:
    fig = charts.equity_with_drawdown(_equity(), include_drawdown=True)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_equity_with_drawdown_single_panel() -> None:
    fig = charts.equity_with_drawdown(_equity(), include_drawdown=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_equity_with_drawdown_empty_state() -> None:
    fig = charts.equity_with_drawdown(pd.DataFrame())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_price_with_signals_includes_overlays_and_markers() -> None:
    bars = _bars(5)
    trades = pd.DataFrame(
        [
            {"date": "2024-01-03", "action": "buy", "quantity": 1, "price": 102.0, "cash_after": 0.0},
            {"date": "2024-01-05", "action": "sell", "quantity": 1, "price": 105.0, "cash_after": 200.0},
        ]
    )
    overlays = {"SMA(2)": pd.Series([100.0, 100.5, 101.5, 102.5, 103.5])}
    fig = charts.price_with_signals(bars, trades, overlays)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4


def test_price_with_signals_empty_bars() -> None:
    fig = charts.price_with_signals(pd.DataFrame())
    assert len(fig.data) == 0


def test_price_with_signals_includes_expected_move_band() -> None:
    moves = pd.DataFrame(
        [
            {"asof": "2024-01-02", "spot": 100.0, "tos": 3.0, "dte": 7.0},
            {"asof": "2024-01-02", "spot": 100.0, "tos": 5.0, "dte": 30.0},
        ]
    )

    fig = charts.price_with_signals(_bars(2), expected_moves=moves)

    assert [trace.name for trace in fig.data[-2:]] == [
        "Expected move upper",
        "Expected move lower",
    ]
    assert list(fig.data[-2].y) == [103.0]
    assert list(fig.data[-1].y) == [97.0]


def test_equity_overlay_skips_empty_runs() -> None:
    runs = {
        "alpha": _equity(),
        "beta": pd.DataFrame(),
        "gamma": _equity(3),
    }
    fig = charts.equity_overlay(runs)
    assert len(fig.data) == 2


def test_equity_overlay_empty_when_no_runs() -> None:
    fig = charts.equity_overlay({})
    assert len(fig.data) == 0


def test_monthly_heatmap_returns_one_trace() -> None:
    table = pd.DataFrame(
        [[0.01, 0.02, 0.03, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        index=[2024],
        columns=list(range(1, 13)),
    )
    fig = charts.monthly_heatmap(table)
    assert len(fig.data) == 1


def test_monthly_heatmap_empty_state() -> None:
    fig = charts.monthly_heatmap(pd.DataFrame())
    assert len(fig.data) == 0


def test_rolling_sharpe_chart_basic() -> None:
    s = pd.Series([0.1, 0.2, 0.3], index=["2024-01-02", "2024-01-03", "2024-01-04"])
    fig = charts.rolling_sharpe_chart(s)
    assert len(fig.data) == 1


def test_chain_iv_scatter_handles_missing_columns() -> None:
    fig = charts.chain_iv_scatter(pd.DataFrame({"strike": [100], "side": ["call"]}))
    assert len(fig.data) == 0


def test_strategy_pnl_chart_smoke() -> None:
    S = np.linspace(80.0, 120.0, 50)
    pe = S - 100.0
    pn = pe * 0.95
    fig = charts.strategy_pnl_chart(S, pe, pn, spot=100.0, breakevens=[90.0, 110.0])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    assert any(getattr(t, "name", None) == "At expiration" for t in fig.data)
    assert any(getattr(t, "name", None) == "At evaluation date" for t in fig.data)


def test_price_with_earnings_smoke() -> None:
    bars = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "close": [100.0, 101.0, 102.0],
        }
    )
    earn = pd.DataFrame({"earnings_date": ["2024-01-03"], "timing": ["AMC"]})
    fig = charts.price_with_earnings(bars, earn)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
