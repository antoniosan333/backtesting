"""Narrow reads of normalized OptionsDX Parquet."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from lambdaclass.data_adapters.optionsdx_chain_loader import CHAIN_COLUMNS


def read_atm_slice(
    normalized_root: Path,
    symbol: str,
    start: str,
    end: str,
    *,
    max_strike_distance_pct: float = 0.15,
) -> pd.DataFrame:
    """Read near-the-money quotes for one symbol between two quote dates."""
    root = Path(normalized_root) / symbol.upper()
    paths = sorted(root.rglob("*.parquet")) if root.is_dir() else []
    if not paths:
        return pd.DataFrame(columns=CHAIN_COLUMNS)
    placeholders = ", ".join("?" for _ in paths)
    query = f"""
        SELECT
            quote_date AS quote_asof,
            expire_date AS expiry,
            side,
            strike,
            bid,
            ask,
            iv AS implied_volatility,
            underlying_last,
            dte,
            symbol
        FROM read_parquet([{placeholders}])
        WHERE substr(CAST(quote_date AS VARCHAR), 1, 10) >= ?
          AND substr(CAST(quote_date AS VARCHAR), 1, 10) <= ?
          AND (
                strike_distance_pct IS NULL
                OR abs(strike_distance_pct) <= ?
          )
        ORDER BY quote_asof, expiry, strike
    """
    with duckdb.connect() as con:
        frame = con.execute(
            query,
            [str(path) for path in paths] + [start[:10], end[:10], max_strike_distance_pct],
        ).df()
    if frame.empty:
        return pd.DataFrame(columns=CHAIN_COLUMNS)
    frame["asof"] = frame["quote_asof"].astype(str).str[:10]
    frame["expiry"] = frame["expiry"].astype(str).str[:10]
    frame["side"] = frame["side"].astype(str).str.lower()
    frame["contract_symbol"] = [
        f"{symbol.upper()}_{expiry.replace('-', '')}_{'C' if side.startswith('c') else 'P'}_{int(round(float(strike) * 1000))}"
        for expiry, side, strike in zip(frame["expiry"], frame["side"], frame["strike"], strict=False)
    ]
    frame["last_price"] = 0.0
    frame["open_interest"] = 0.0
    frame["volume"] = 0.0
    for column in CHAIN_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[CHAIN_COLUMNS]
