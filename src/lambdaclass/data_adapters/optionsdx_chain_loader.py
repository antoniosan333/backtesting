from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import duckdb
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
]


def _requested_months(date_strings: set[str]) -> set[tuple[str, str]]:
    months: set[tuple[str, str]] = set()
    for value in date_strings:
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            continue
        months.add((f"{parsed.year:04d}", f"{parsed.month:02d}"))
    return months


def _partition_paths(sym_dir: Path, requested_months: set[tuple[str, str]]) -> list[Path]:
    """Return matching canonical partitions and all noncanonical legacy files."""
    selected: list[Path] = []
    for path in sorted(sym_dir.rglob("*.parquet")):
        relative_parts = path.relative_to(sym_dir).parts
        partition: tuple[str, str] | None = None
        if len(relative_parts) >= 3:
            year, month = relative_parts[:2]
            if (
                len(year) == 4
                and year.isdigit()
                and len(month) == 2
                and month.isdigit()
                and 1 <= int(month) <= 12
            ):
                partition = (year, month)
        if partition is None or partition in requested_months:
            selected.append(path)
    return selected


def _read_filtered(paths: list[Path], symbol: str, date_set: set[str]) -> pd.DataFrame:
    parquet_paths = [str(path) for path in paths]
    requested = pd.DataFrame({"requested_date": sorted(date_set)})
    with duckdb.connect() as connection:
        connection.register("requested_dates", requested)
        source = "read_parquet(?, union_by_name = true)"
        columns = {
            description[0]
            for description in connection.execute(
                f"SELECT * FROM {source} LIMIT 0", [parquet_paths]
            ).description
        }
        required = {
            "symbol",
            "quote_date",
            "expire_date",
            "side",
            "strike",
            "last",
            "bid",
            "ask",
            "iv",
            "volume",
        }
        missing = required - columns
        if missing:
            raise KeyError(next(iter(sorted(missing))))
        projected = sorted(required)
        projected.extend(column for column in ("contract_symbol", "open_interest") if column in columns)
        projection = ", ".join(f'"{column}"' for column in projected)
        query = f"""
            SELECT {projection},
                   substr(trim(CAST(quote_date AS VARCHAR)), 1, 10) AS _asof
            FROM {source}
            WHERE upper(CAST(symbol AS VARCHAR)) = ?
              AND substr(trim(CAST(quote_date AS VARCHAR)), 1, 10)
                  IN (SELECT requested_date FROM requested_dates)
        """
        return connection.execute(query, [parquet_paths, symbol]).fetchdf()


def _contract_symbols(raw: pd.DataFrame) -> pd.Series:
    if "contract_symbol" in raw:
        contract = raw["contract_symbol"].astype("string").str.strip()
        valid_contract = contract.notna() & ~contract.str.lower().isin(("nan", "none", ""))
    else:
        contract = pd.Series(pd.NA, index=raw.index, dtype="string")
        valid_contract = pd.Series(False, index=raw.index)

    expiry = raw["expire_date"].fillna("").astype(str).str.replace("-", "", regex=False).str[:8]
    side = raw["side"].fillna("").astype(str).str.lower()
    call_put = pd.Series("X", index=raw.index)
    call_put.loc[side.str.startswith("c")] = "C"
    call_put.loc[side.str.startswith("p")] = "P"
    strike = (
        pd.to_numeric(raw["strike"], errors="coerce").mul(1000).round().fillna(0).astype("int64").astype(str)
    )
    synthetic = raw["symbol"].astype(str).str.upper() + "_" + expiry + "_" + call_put + "_" + strike
    return contract.where(valid_contract, synthetic)


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

    date_set = {str(d)[:10] for d in bar_dates}
    if not date_set:
        return pd.DataFrame(columns=CHAIN_COLUMNS)

    paths = _partition_paths(sym_dir, _requested_months(date_set))
    if not paths:
        return pd.DataFrame(columns=CHAIN_COLUMNS)

    raw = _read_filtered(paths, sym, date_set)
    if raw.empty:
        return pd.DataFrame(columns=CHAIN_COLUMNS)

    raw["_contract"] = _contract_symbols(raw)

    oi = (
        raw["open_interest"]
        if "open_interest" in raw.columns
        else pd.Series(0.0, index=raw.index, dtype=float)
    )
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
        }
    )
    return out.sort_values(["asof", "contract_symbol"]).reset_index(drop=True)
