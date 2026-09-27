from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataAdapter(Protocol):
    def get_stock_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...

    def get_option_chain(self, symbol: str, asof: date, expiry: date | None = None) -> pd.DataFrame: ...
