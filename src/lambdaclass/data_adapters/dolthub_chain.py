"""Historical option chains from the public DoltHub options database."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from lambdaclass.data_adapters.optionsdx_chain_loader import _synthetic_contract_symbol

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")


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


def _rows_to_chain(symbol: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in rows:
        side = str(row.get("call_put", "")).strip().lower()
        if side not in ("call", "put"):
            continue
        expiry = str(row.get("expiration", ""))[:10]
        asof = str(row.get("date", ""))[:10]
        strike = pd.to_numeric(row.get("strike"), errors="coerce")
        record = {
            "symbol": symbol,
            "side": side,
            "strike": float(strike) if pd.notna(strike) else 0.0,
            "expiry": expiry,
            "expire_date": expiry,
            "asof": asof,
            "bid": float(pd.to_numeric(row.get("bid"), errors="coerce") or 0.0),
            "ask": float(pd.to_numeric(row.get("ask"), errors="coerce") or 0.0),
            "last_price": 0.0,
            "implied_volatility": float(pd.to_numeric(row.get("vol"), errors="coerce") or 0.0),
            "open_interest": 0.0,
            "volume": 0.0,
        }
        record["contract_symbol"] = _synthetic_contract_symbol(pd.Series(record))
        records.append(record)
    return pd.DataFrame(records)


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
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))
