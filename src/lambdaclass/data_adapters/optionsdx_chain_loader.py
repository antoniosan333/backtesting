from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

CHAIN_COLUMNS = [
    "contract_symbol",
    "side",
    "strike",
    "last_price",
    "bid",
    "ask",
    "implied_volatility",
    "open_interest",
    "volume",
    "expiry",
    "asof",
    "symbol",
    "underlying_last",
    "dte",
]


def _quote_asof(quote_date: object) -> str:
    s = str(quote_date).strip()
    return s[:10] if len(s) >= 10 else s


def _synthetic_contract_symbol(row: pd.Series) -> str:
    sym = str(row.get("symbol", "")).upper()
    exp = str(row.get("expire_date", "") or "").replace("-", "")[:8]
    side = str(row.get("side", "") or "").lower()
    cp = "C" if side.startswith("c") else "P" if side.startswith("p") else "X"
    strike = row.get("strike")
    if strike is None or (isinstance(strike, float) and pd.isna(strike)):
        sk = 0
    else:
        sk = int(round(float(strike) * 1000))
    return f"{sym}_{exp}_{cp}_{sk}"


def load_normalized_optionsdx_chain(
    normalized_root: Path,
    symbol: str,
    bar_dates: Sequence[str] | pd.Series,
) -> pd.DataFrame:
    """
    Load OptionsDX-normalized Parquet under ``normalized_root/<SYMBOL>/**/`` and
    return a frame aligned with ``YFinanceAdapter.get_option_chain`` plus ``symbol``,
    filtered to ``quote_date`` dates present in ``bar_dates`` (YYYY-MM-DD strings).
    """
    sym = symbol.upper()
    root = normalized_root.resolve()
    sym_dir = root / sym
    if not sym_dir.is_dir():
        return pd.DataFrame(columns=CHAIN_COLUMNS)

    paths = sorted(sym_dir.rglob("*.parquet"))
    if not paths:
        return pd.DataFrame(columns=CHAIN_COLUMNS)

    frames = [pd.read_parquet(p) for p in paths]
    raw = pd.concat(frames, ignore_index=True)
    if raw.empty:
        return pd.DataFrame(columns=CHAIN_COLUMNS)

    raw = raw[raw["symbol"].astype(str).str.upper() == sym].copy()
    if raw.empty:
        return pd.DataFrame(columns=CHAIN_COLUMNS)

    date_set = {str(d)[:10] for d in bar_dates}
    raw["_asof"] = raw["quote_date"].map(_quote_asof)
    raw = raw[raw["_asof"].isin(date_set)].copy()
    if raw.empty:
        return pd.DataFrame(columns=CHAIN_COLUMNS)

    def _contract_cell(row: pd.Series) -> str:
        c = row.get("contract_symbol")
        if c is not None and pd.notna(c):
            t = str(c).strip()
            if t and t.lower() not in ("nan", "none"):
                return t
        return _synthetic_contract_symbol(row)

    raw["_contract"] = raw.apply(_contract_cell, axis=1)

    oi = raw["open_interest"] if "open_interest" in raw.columns else pd.Series(0.0, index=raw.index, dtype=float)
    out = pd.DataFrame(
        {
            "contract_symbol": raw["_contract"],
            "side": raw["side"].astype(str).str.lower(),
            "strike": pd.to_numeric(raw["strike"], errors="coerce").fillna(0.0),
            "last_price": pd.to_numeric(raw["last"], errors="coerce").fillna(0.0),
            "bid": pd.to_numeric(raw["bid"], errors="coerce").fillna(0.0),
            "ask": pd.to_numeric(raw["ask"], errors="coerce").fillna(0.0),
            "implied_volatility": pd.to_numeric(raw["iv"], errors="coerce").fillna(0.0),
            "open_interest": pd.to_numeric(oi, errors="coerce").fillna(0.0),
            "volume": pd.to_numeric(raw["volume"], errors="coerce").fillna(0.0),
            "expiry": raw["expire_date"].astype(str),
            "asof": raw["_asof"],
            "symbol": sym,
            "underlying_last": pd.to_numeric(raw["underlying_last"], errors="coerce"),
            "dte": pd.to_numeric(raw["dte"], errors="coerce"),
        }
    )
    return out.sort_values(["asof", "contract_symbol"]).reset_index(drop=True)
