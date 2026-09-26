from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lambdaclass.backtest.engine import run_backtest
from lambdaclass.config import Preferences
from lambdaclass.storage.duckdb_store import DuckDBStore
from lambdaclass.strategies.base import Strategy, StrategyContext, StrategyDecision


class _BuyTenOnce(Strategy):
    name = "buy_ten_once"

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        if context.position == 0:
            return StrategyDecision(action="buy", quantity=10)
        return StrategyDecision(action="hold")


def _bars(dividends: list[float] | None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1_000, 1_000, 1_000],
        }
    )
    if dividends is not None:
        frame["dividends"] = dividends
    return frame


def _prefs() -> Preferences:
    prefs = Preferences()
    prefs.defaults.slippage_bps = 0.0
    prefs.risk.max_position_pct = 1.0
    return prefs


def test_dividend_is_credited_to_held_shares() -> None:
    result = run_backtest(_BuyTenOnce(), _bars([0.0, 1.0, 0.0]), pd.DataFrame(), _prefs())

    equity = result.equity_curve["equity"].tolist()
    assert equity[0] == pytest.approx(100_000.0)
    assert equity[1] == pytest.approx(100_010.0)
    assert equity[2] == pytest.approx(100_010.0)
    dividend_rows = result.trades[result.trades["action"] == "dividend"]
    assert len(dividend_rows) == 1
    assert dividend_rows.iloc[0]["quantity"] == 10
    assert dividend_rows.iloc[0]["price"] == pytest.approx(1.0)


def test_dividend_before_position_opened_is_not_credited() -> None:
    result = run_backtest(_BuyTenOnce(), _bars([1.0, 0.0, 0.0]), pd.DataFrame(), _prefs())

    assert result.equity_curve["equity"].iloc[-1] == pytest.approx(100_000.0)
    assert (result.trades["action"] == "dividend").sum() == 0


def test_bars_without_dividends_column_run_unchanged() -> None:
    with_column = run_backtest(_BuyTenOnce(), _bars([0.0, 0.0, 0.0]), pd.DataFrame(), _prefs())
    without_column = run_backtest(_BuyTenOnce(), _bars(None), pd.DataFrame(), _prefs())

    assert with_column.equity_curve.equals(without_column.equity_curve)


def test_store_backfills_missing_dividends_column(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path)
    legacy = _bars(None)
    legacy["symbol"] = "SPY"
    legacy.to_parquet(store.stocks_dir / "SPY.parquet", index=False)

    bars = store.read_bars("SPY")

    assert "dividends" in bars.columns
    assert bars["dividends"].tolist() == [0.0, 0.0, 0.0]
    assert bars.attrs.get("dividends_backfilled") is True


def test_store_round_trips_dividends(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path)
    store.write_bars("SPY", _bars([0.0, 0.5, 0.0]))

    bars = store.read_bars("SPY")

    assert bars["dividends"].tolist() == [0.0, 0.5, 0.0]
    assert not bars.attrs.get("dividends_backfilled")
