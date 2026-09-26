"""Cboe "Available Weeklys" list: which underlyings list weekly option expirations."""

from __future__ import annotations

import csv
import io
import urllib.request

import pandas as pd

from lambdaclass.symbols import SYMBOL_PATTERN

CBOE_WEEKLYS_URL = "https://www.cboe.com/available_weeklys/get_csv_download/"
UNIVERSE_COLUMNS = ["symbol", "name", "category"]
_SECTION_CATEGORIES = {
    "exchange traded products": "etp",
    "equity": "equity",
}
_USER_AGENT = "Mozilla/5.0 (compatible; lambdaclass-backtester)"


def to_vendor_neutral_symbol(symbol: str) -> str:
    """Upper-case and use ``-`` for share classes (``BRK.B`` / ``BRK/B`` → ``BRK-B``), as yfinance does."""
    return symbol.strip().upper().replace(".", "-").replace("/", "-")


def parse_cboe_weeklys(text: str) -> pd.DataFrame:
    """Parse the Cboe CSV into ``symbol, name, category`` rows.

    The file starts with index expiration schedules (``SPXW (MON)``, ...) that are
    not underlyings; only rows under the ``Available Weeklys - Exchange Traded
    Products`` and ``Available Weeklys - Equity`` headers are kept.
    """
    rows: list[dict[str, str]] = []
    category: str | None = None
    for record in csv.reader(io.StringIO(text)):
        if not record or not any(cell.strip() for cell in record):
            continue
        first = record[0].strip()
        if first.lower().startswith("available weeklys"):
            section = first.split("-", 1)[-1].strip().lower()
            category = next(
                (value for key, value in _SECTION_CATEGORIES.items() if section.startswith(key)),
                None,
            )
            continue
        if category is None or len(record) < 2:
            continue
        symbol = to_vendor_neutral_symbol(first)
        if not SYMBOL_PATTERN.fullmatch(symbol) or symbol.startswith("^"):
            # Symbols become Parquet file names downstream; drop anything unsafe.
            continue
        rows.append(
            {
                "symbol": symbol,
                "name": record[1].strip(),
                "category": category,
            }
        )
    frame = pd.DataFrame(rows, columns=UNIVERSE_COLUMNS)
    return frame.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)


def download_cboe_weeklys(timeout: float = 30.0) -> str:
    request = urllib.request.Request(CBOE_WEEKLYS_URL, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload: bytes = response.read()
    return payload.decode("utf-8-sig", errors="replace")
