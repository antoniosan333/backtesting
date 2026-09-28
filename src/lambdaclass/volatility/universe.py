"""Earnings-research symbol lists."""

from __future__ import annotations

import csv
from pathlib import Path

PILOT_SYMBOLS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMD",
    "META",
    "AMZN",
    "GOOGL",
    "JPM",
    "XOM",
    "AVGO",
    "CRM",
    "COST",
    "BA",
    "DIS",
    "INTC",
    "QCOM",
    "PYPL",
    "COIN",
    "UBER",
    "SNOW",
    "PANW",
    "LLY",
    "UNH",
    "BAC",
)


def load_stock_universe(path: Path) -> list[str]:
    """Read ticker symbols from ``weekly_options_stocks.csv``."""
    if not path.is_file():
        raise FileNotFoundError(f"Earnings universe not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["symbol"].strip().upper() for row in csv.DictReader(handle) if row.get("symbol")]
