from datetime import date

import pandas as pd

from lambdaclass.data_adapters.base import MarketDataAdapter


class DummyAdapter:
    def get_stock_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame(
            [{"date": "2026-01-01", "open": 1.0, "high": 2.0, "low": 1.0, "close": 2.0, "volume": 100}]
        )

    def get_option_chain(self, symbol: str, asof: date, expiry: date | None = None) -> pd.DataFrame:
        return pd.DataFrame([{"contract_symbol": "TEST", "asof": asof.isoformat()}])


def test_dummy_adapter_matches_protocol() -> None:
    adapter = DummyAdapter()
    assert isinstance(adapter, MarketDataAdapter)
