from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from lambdaclass.data_adapters import yfinance_adapter
from lambdaclass.data_adapters.yfinance_adapter import YFinanceAdapter


class _FakeTicker:
    calls: list[dict[str, Any]] = []

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def history(self, **kwargs: Any) -> pd.DataFrame:
        _FakeTicker.calls.append(kwargs)
        index = pd.DatetimeIndex(["2024-03-14", "2024-03-15"], name="Date")
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Adj Close": [100.5, 101.5],
                "Volume": [1_000, 1_100],
                "Dividends": [0.0, 1.25],
                "Stock Splits": [0.0, 0.0],
            },
            index=index,
        )


@pytest.fixture(autouse=True)
def _fake_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeTicker.calls = []
    monkeypatch.setattr(yfinance_adapter.yf, "Ticker", _FakeTicker)


def test_get_stock_bars_includes_requested_end_date() -> None:
    YFinanceAdapter().get_stock_bars("SPY", date(2024, 3, 1), date(2024, 3, 15))

    assert _FakeTicker.calls[0]["start"] == "2024-03-01"
    assert _FakeTicker.calls[0]["end"] == "2024-03-16"


def test_get_stock_bars_keeps_dividends() -> None:
    bars = YFinanceAdapter().get_stock_bars("SPY", date(2024, 3, 1), date(2024, 3, 15))

    assert list(bars.columns) == ["date", "open", "high", "low", "close", "volume", "dividends"]
    assert bars["dividends"].tolist() == [0.0, 1.25]
    assert bars["date"].tolist() == ["2024-03-14", "2024-03-15"]
