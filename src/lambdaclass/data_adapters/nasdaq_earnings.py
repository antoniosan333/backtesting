"""Nasdaq earnings calendar: every company reporting on a given day, with EPS results.

Historical days carry actual and consensus EPS but almost never the time of day
(``time-not-supplied``); upcoming days usually do. See ``lambdaclass.earnings.events``
for inferring historical timing from prices.
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from lambdaclass.data_adapters.cboe_weeklys import to_vendor_neutral_symbol

NASDAQ_EARNINGS_URL = "https://api.nasdaq.com/api/calendar/earnings?date={day}"
EARNINGS_DAY_COLUMNS = [
    "earnings_date",
    "symbol",
    "company_name",
    "timing",
    "eps_actual",
    "eps_estimate",
    "surprise_pct",
    "num_estimates",
    "fiscal_quarter_ending",
    "market_cap",
    "source",
    "fetched_at",
]
_TIMING = {"time-pre-market": "BMO", "time-after-hours": "AMC"}
_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) lambdaclass-backtester"
_NUMBER = re.compile(r"[^0-9.\-]")


def parse_money(raw: object) -> float:
    """``$1.23`` → 1.23, ``($0.09)`` → -0.09, ``N/A`` / blank → NaN."""
    text = str(raw or "").strip()
    if not text or text.upper() in {"N/A", "NA", "--", "-"}:
        return float("nan")
    negative = text.startswith("(") and text.endswith(")")
    cleaned = _NUMBER.sub("", text)
    if cleaned in {"", "-", "."}:
        return float("nan")
    try:
        value = float(cleaned)
    except ValueError:
        return float("nan")
    return -abs(value) if negative else value


def empty_earnings_day() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=_dtype(column)) for column in EARNINGS_DAY_COLUMNS})


def _dtype(column: str) -> str:
    numeric = {"eps_actual", "eps_estimate", "surprise_pct", "num_estimates", "market_cap"}
    return "float64" if column in numeric else "object"


def parse_nasdaq_earnings_rows(rows: list[dict[str, Any]], day: date) -> pd.DataFrame:
    """Normalize the ``data.rows`` list of one calendar day."""
    if not rows:
        return empty_earnings_day()
    fetched_at = datetime.now(tz=UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = [
        {
            "earnings_date": day.isoformat(),
            "symbol": to_vendor_neutral_symbol(str(row.get("symbol", ""))),
            "company_name": str(row.get("name") or "").strip(),
            "timing": _TIMING.get(str(row.get("time") or "").strip().lower(), "unknown"),
            "eps_actual": parse_money(row.get("eps")),
            "eps_estimate": parse_money(row.get("epsForecast")),
            "surprise_pct": parse_money(row.get("surprise")),
            "num_estimates": parse_money(row.get("noOfEsts")),
            "fiscal_quarter_ending": str(row.get("fiscalQuarterEnding") or "").strip(),
            "market_cap": parse_money(row.get("marketCap")),
            "source": "nasdaq",
            "fetched_at": fetched_at,
        }
        for row in rows
        if str(row.get("symbol") or "").strip()
    ]
    frame = pd.DataFrame(records, columns=EARNINGS_DAY_COLUMNS)
    return frame.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)


class NasdaqEarningsAdapter:
    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout

    def get_earnings_day(self, day: date) -> pd.DataFrame:
        request = urllib.request.Request(
            NASDAQ_EARNINGS_URL.format(day=day.isoformat()),
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data") or {}
        return parse_nasdaq_earnings_rows(data.get("rows") or [], day)
