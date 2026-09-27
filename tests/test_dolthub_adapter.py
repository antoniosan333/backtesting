"""Tests for DoltHub chain and vol history adapters."""

from __future__ import annotations

import pandas as pd
import pytest

from lambdaclass.data_adapters.dolthub_chain import (
    _rows_to_chain,
    _rows_to_vol_history,
    fetch_dolthub_chain,
    fetch_dolthub_vol_history,
)


# ---------------------------------------------------------------------------
# Chain row mapping
# ---------------------------------------------------------------------------

def test_rows_to_chain_basic() -> None:
    rows = [
        {"date": "2024-01-02", "expiration": "2024-01-19", "strike": "190.00",
         "call_put": "Call", "bid": "2.50", "ask": "2.60", "vol": "0.35"},
        {"date": "2024-01-02", "expiration": "2024-01-19", "strike": "190.00",
         "call_put": "Put", "bid": "1.80", "ask": "1.85", "vol": "0.32"},
    ]
    df = _rows_to_chain("AAPL", rows)
    assert len(df) == 2
    assert list(df["symbol"]) == ["AAPL", "AAPL"]
    assert df.iloc[0]["side"] == "call"
    assert df.iloc[1]["side"] == "put"
    assert df.iloc[0]["strike"] == 190.0
    assert df.iloc[0]["bid"] == 2.50
    assert df.iloc[0]["ask"] == 2.60
    assert df.iloc[0]["implied_volatility"] == pytest.approx(0.35)
    assert df.iloc[0]["expiry"] == "2024-01-19"
    assert df.iloc[0]["asof"] == "2024-01-02"
    assert "contract_symbol" in df.columns


def test_rows_to_chain_empty() -> None:
    df = _rows_to_chain("AAPL", [])
    assert df.empty


def test_rows_to_chain_skips_unknown_side() -> None:
    rows = [
        {"date": "2024-01-02", "expiration": "2024-01-19", "strike": "190.00",
         "call_put": "Invalid", "bid": "0", "ask": "0", "vol": "0"},
    ]
    df = _rows_to_chain("AAPL", rows)
    assert df.empty


def test_rows_to_chain_handles_missing_fields() -> None:
    rows = [{"date": "2024-01-02", "expiration": "2024-01-19", "strike": "190.00",
             "call_put": "Call"}]
    df = _rows_to_chain("AAPL", rows)
    assert len(df) == 1
    assert df.iloc[0]["bid"] == 0.0
    assert df.iloc[0]["ask"] == 0.0
    assert df.iloc[0]["implied_volatility"] == 0.0


# ---------------------------------------------------------------------------
# Vol history row mapping
# ---------------------------------------------------------------------------

def test_rows_to_vol_history_basic() -> None:
    rows = [{
        "date": "2024-01-02", "act_symbol": "AAPL",
        "hv_current": "0.25", "hv_week_ago": "0.30", "hv_month_ago": "0.34",
        "hv_year_high": "0.40", "hv_year_high_date": "2024-01-03",
        "hv_year_low": "0.12", "hv_year_low_date": "2023-10-03",
        "iv_current": "0.28", "iv_week_ago": "0.25", "iv_month_ago": "0.29",
        "iv_year_high": "0.36", "iv_year_high_date": "2023-12-24",
        "iv_year_low": "0.16", "iv_year_low_date": "2023-08-31",
    }]
    df = _rows_to_vol_history("AAPL", rows)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["date"] == "2024-01-02"
    assert row["symbol"] == "AAPL"
    assert row["iv_current"] == pytest.approx(0.28)
    assert row["hv_current"] == pytest.approx(0.25)
    assert row["iv_year_high"] == pytest.approx(0.36)
    assert row["iv_year_low"] == pytest.approx(0.16)


def test_rows_to_vol_history_empty() -> None:
    df = _rows_to_vol_history("AAPL", [])
    assert df.empty
    assert "iv_current" in df.columns
    assert "hv_current" in df.columns


# ---------------------------------------------------------------------------
# Mocked API calls
# ---------------------------------------------------------------------------

def _mock_api(rows: list[dict]) -> object:
    """Return a callable that mimics _default_get with canned rows."""
    def _get(url: str) -> dict:
        return {"query_execution_status": "Success", "rows": rows}
    return _get


def test_fetch_dolthub_chain_mocked() -> None:
    rows = [
        {"date": "2024-01-02", "expiration": "2024-01-19", "strike": "190.00",
         "call_put": "Call", "bid": "2.50", "ask": "2.60", "vol": "0.35"},
    ]
    df = fetch_dolthub_chain("AAPL", "2024-01-01", "2024-01-31", http_get=_mock_api(rows))
    assert len(df) == 1
    assert df.iloc[0]["side"] == "call"
    assert df.iloc[0]["implied_volatility"] == pytest.approx(0.35)


def test_fetch_dolthub_chain_empty_mocked() -> None:
    df = fetch_dolthub_chain("ZZZZ", "2024-01-01", "2024-01-31", http_get=_mock_api([]))
    assert df.empty


def test_fetch_dolthub_vol_history_mocked() -> None:
    rows = [{
        "date": "2024-01-02", "act_symbol": "AAPL",
        "hv_current": "0.25", "hv_week_ago": "0.30", "hv_month_ago": "0.34",
        "hv_year_high": "0.40", "hv_year_high_date": "2024-01-03",
        "hv_year_low": "0.12", "hv_year_low_date": "2023-10-03",
        "iv_current": "0.28", "iv_week_ago": "0.25", "iv_month_ago": "0.29",
        "iv_year_high": "0.36", "iv_year_high_date": "2023-12-24",
        "iv_year_low": "0.16", "iv_year_low_date": "2023-08-31",
    }]
    df = fetch_dolthub_vol_history("AAPL", "2024-01-01", "2024-01-31", http_get=_mock_api(rows))
    assert len(df) == 1
    assert df.iloc[0]["iv_current"] == pytest.approx(0.28)


def test_fetch_dolthub_chain_invalid_symbol() -> None:
    with pytest.raises(ValueError):
        fetch_dolthub_chain("bad-symbol!", "2024-01-01", "2024-01-31", http_get=_mock_api([]))


def test_fetch_dolthub_chain_api_error() -> None:
    def error_get(url: str) -> dict:
        return {"query_execution_status": "Error", "query_execution_message": "table not found"}
    with pytest.raises(RuntimeError, match="DoltHub query failed"):
        fetch_dolthub_chain("AAPL", "2024-01-01", "2024-01-31", http_get=error_get)