"""Historical option chains and volatility from the public DoltHub options database.

API: https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master?q={SQL}

Two tables:
    option_chain      — EOD chain rows (bid/ask/IV/Greeks per strike/expiry)
    volatility_history — daily IV/HV summary with year-high/low (for ramp/crush)
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_dolthub_chain(
    symbol: str,
    start: str,
    end: str,
    *,
    http_get: Callable[[str], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Page ``option_chain`` by month and return the backtest chain schema."""
    ticker = symbol.upper()
    if not _SYMBOL.match(ticker):
        raise ValueError(f"Unsupported symbol {symbol!r}")
    getter = http_get or _default_get
    frames: list[pd.DataFrame] = []
    for month_start, month_end in _months(start, end):
        sql = (
            "SELECT date, expiration, strike, call_put, bid, ask, vol "
            "FROM option_chain "
            f"WHERE act_symbol = '{ticker}' "
            f"AND date >= '{month_start}' AND date < '{month_end}'"
        )
        payload = getter(f"{API}?q={quote(sql)}")
        status = str(payload.get("query_execution_status", ""))
        if status != "Success":
            message = str(payload.get("query_execution_message", status))
            raise RuntimeError(f"DoltHub query failed for {ticker} {month_start}: {message}")
        frame = _rows_to_chain(ticker, payload.get("rows") or [])
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_dolthub_vol_history(
    symbol: str,
    start: str,
    end: str,
    *,
    http_get: Callable[[str], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Fetch ``volatility_history`` rows for *symbol* in *[start, end]*.

    Returns one row per trading day with columns:
    date, symbol, iv_current, hv_current, iv_week_ago, iv_month_ago,
    iv_year_high, iv_year_high_date, iv_year_low, iv_year_low_date,
    hv_week_ago, hv_month_ago, hv_year_high, hv_year_high_date,
    hv_year_low, hv_year_low_date.
    """
    ticker = symbol.upper()
    if not _SYMBOL.match(ticker):
        raise ValueError(f"Unsupported symbol {symbol!r}")
    getter = http_get or _default_get
    sql = (
        "SELECT * FROM volatility_history "
        f"WHERE act_symbol = '{ticker}' "
        f"AND date >= '{start[:10]}' AND date <= '{end[:10]}' "
        "ORDER BY date"
    )
    payload = getter(f"{API}?q={quote(sql)}")
    status = str(payload.get("query_execution_status", ""))
    if status != "Success":
        message = str(payload.get("query_execution_message", status))
        raise RuntimeError(f"DoltHub vol_history query failed for {ticker}: {message}")
    return _rows_to_vol_history(ticker, payload.get("rows") or [])


# ---------------------------------------------------------------------------
# Row mappers
# ---------------------------------------------------------------------------

def _rows_to_chain(symbol: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in rows:
        side = str(row.get("call_put", "")).strip().lower()
        if side not in ("call", "put"):
            continue
        expiry = str(row.get("expiration", ""))[:10]
        asof = str(row.get("date", ""))[:10]
        strike = pd.to_numeric(row.get("strike"), errors="coerce")
        strike_val = float(strike) if pd.notna(strike) else 0.0
        # Build synthetic OCC-style contract symbol
        exp_compact = expiry.replace("-", "")
        c_p = "C" if side == "call" else "P"
        strike_int = int(round(strike_val * 1000))
        contract_symbol = f"{symbol}_{exp_compact}{c_p}_{strike_int:08d}"
        record = {
            "symbol": symbol,
            "side": side,
            "strike": strike_val,
            "expiry": expiry,
            "asof": asof,
            "bid": _f(row.get("bid")),
            "ask": _f(row.get("ask")),
            "last_price": 0.0,
            "implied_volatility": _f(row.get("vol")),
            "open_interest": 0.0,
            "volume": 0.0,
            "contract_symbol": contract_symbol,
        }
        records.append(record)
    return pd.DataFrame(records)


def _rows_to_vol_history(symbol: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "date", "symbol", "iv_current", "hv_current",
        "iv_week_ago", "iv_month_ago", "iv_year_high", "iv_year_high_date",
        "iv_year_low", "iv_year_low_date",
        "hv_week_ago", "hv_month_ago", "hv_year_high", "hv_year_high_date",
        "hv_year_low", "hv_year_low_date",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, object]] = []
    for row in rows:
        records.append({
            "date": str(row.get("date", ""))[:10],
            "symbol": symbol,
            "iv_current": _f(row.get("iv_current")),
            "hv_current": _f(row.get("hv_current")),
            "iv_week_ago": _f(row.get("iv_week_ago")),
            "iv_month_ago": _f(row.get("iv_month_ago")),
            "iv_year_high": _f(row.get("iv_year_high")),
            "iv_year_high_date": str(row.get("iv_year_high_date", ""))[:10],
            "iv_year_low": _f(row.get("iv_year_low")),
            "iv_year_low_date": str(row.get("iv_year_low_date", ""))[:10],
            "hv_week_ago": _f(row.get("hv_week_ago")),
            "hv_month_ago": _f(row.get("hv_month_ago")),
            "hv_year_high": _f(row.get("hv_year_high")),
            "hv_year_high_date": str(row.get("hv_year_high_date", ""))[:10],
            "hv_year_low": _f(row.get("hv_year_low")),
            "hv_year_low_date": str(row.get("hv_year_low_date", ""))[:10],
        })
    return pd.DataFrame(records, columns=columns)


def _f(value: Any) -> float:
    v = pd.to_numeric(value, errors="coerce")
    return float(v) if pd.notna(v) else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _months(start: str, end: str) -> list[tuple[str, str]]:
    first = date.fromisoformat(start[:10]).replace(day=1)
    last = date.fromisoformat(end[:10])
    months: list[tuple[str, str]] = []
    cursor = first
    while cursor <= last:
        if cursor.month == 12:
            nxt = date(cursor.year + 1, 1, 1)
        else:
            nxt = date(cursor.year, cursor.month + 1, 1)
        months.append((cursor.isoformat(), nxt.isoformat()))
        cursor = nxt
    return months


def _default_get(url: str) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))