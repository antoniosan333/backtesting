from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _checked_name(name: str) -> str:
    """Dataset names become file names; keep them to a safe lowercase slug."""
    if not _NAME_PATTERN.match(name):
        raise ValueError(f"Invalid dataset name {name!r}; use lowercase letters, digits, and underscores")
    return name


class DuckDBStore:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.stocks_dir = data_root / "stocks"
        self.options_dir = data_root / "options"
        self.earnings_dir = data_root / "earnings"
        self.stocks_dir.mkdir(parents=True, exist_ok=True)
        self.options_dir.mkdir(parents=True, exist_ok=True)
        self.earnings_dir.mkdir(parents=True, exist_ok=True)

    def _stock_path(self, symbol: str) -> Path:
        return self.stocks_dir / f"{symbol.upper()}.parquet"

    def _options_path(self, symbol: str) -> Path:
        return self.options_dir / f"{symbol.upper()}.parquet"

    def _earnings_path(self, symbol: str) -> Path:
        return self.earnings_dir / f"{symbol.upper()}.parquet"

    def write_bars(self, symbol: str, bars: pd.DataFrame) -> Path:
        if bars.empty:
            return self._stock_path(symbol)
        frame = bars.copy()
        frame["symbol"] = symbol.upper()
        frame["year"] = pd.to_datetime(frame["date"], errors="coerce").dt.year.fillna(0).astype(int)
        path = self._stock_path(symbol)
        if path.exists():
            existing = pd.read_parquet(path)
            frame = pd.concat([existing, frame], ignore_index=True)
            frame = frame.drop_duplicates(subset=["symbol", "date"], keep="last")
        frame = frame.sort_values("date").reset_index(drop=True)
        frame.to_parquet(path, index=False)
        return path

    def read_bars(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        path = self._stock_path(symbol)
        if not path.exists():
            return pd.DataFrame()
        query = "SELECT * FROM read_parquet(?)"
        clauses: list[str] = []
        params: list[str] = [str(path)]
        if start:
            clauses.append("date >= ?")
            params.append(start)
        if end:
            clauses.append("date <= ?")
            params.append(end)
        if clauses:
            query = f"{query} WHERE {' AND '.join(clauses)}"
        with duckdb.connect() as con:
            bars = con.execute(query, params).df()
        if "dividends" not in bars.columns:
            bars["dividends"] = 0.0
            bars.attrs["dividends_backfilled"] = True
        else:
            bars["dividends"] = pd.to_numeric(bars["dividends"], errors="coerce").fillna(0.0)
        return bars

    def read_bars_many(self, symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
        """Bars for many symbols in one query; symbols without stored bars are omitted."""
        frame = self._read_many([self._stock_path(symbol) for symbol in symbols], order_by="symbol, date")
        if frame.empty:
            return {}
        if "dividends" not in frame.columns:
            frame["dividends"] = 0.0
        frame["dividends"] = pd.to_numeric(frame["dividends"], errors="coerce").fillna(0.0)
        return {
            str(symbol): group.reset_index(drop=True) for symbol, group in frame.groupby("symbol", sort=False)
        }

    def _read_many(self, paths: Sequence[Path], *, order_by: str) -> pd.DataFrame:
        existing = [str(path) for path in paths if path.exists()]
        if not existing:
            return pd.DataFrame()
        with duckdb.connect() as con:
            return con.execute(
                f"SELECT * FROM read_parquet(?, union_by_name = true) ORDER BY {order_by}", [existing]
            ).df()

    def bar_date_range(self, symbol: str) -> tuple[str, str] | None:
        """First and last stored bar date for ``symbol``, or ``None`` when nothing is stored."""
        path = self._stock_path(symbol)
        if not path.exists():
            return None
        with duckdb.connect() as con:
            row = con.execute(
                "SELECT min(date)::VARCHAR, max(date)::VARCHAR FROM read_parquet(?)", [str(path)]
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return str(row[0])[:10], str(row[1])[:10]

    def write_chain(self, symbol: str, chain: pd.DataFrame) -> Path:
        if chain.empty:
            return self._options_path(symbol)
        frame = chain.copy()
        frame["symbol"] = symbol.upper()
        frame["expiry_year"] = pd.to_datetime(frame["expiry"], errors="coerce").dt.year.fillna(0).astype(int)
        path = self._options_path(symbol)
        if path.exists():
            existing = pd.read_parquet(path)
            frame = pd.concat([existing, frame], ignore_index=True)
            frame = frame.drop_duplicates(subset=["symbol", "contract_symbol", "asof"], keep="last")
        frame = frame.sort_values(["asof", "contract_symbol"]).reset_index(drop=True)
        frame.to_parquet(path, index=False)
        return path

    def read_chain(self, symbol: str, asof: str | None = None) -> pd.DataFrame:
        path = self._options_path(symbol)
        if not path.exists():
            return pd.DataFrame()
        query = "SELECT * FROM read_parquet(?)"
        params: list[str] = [str(path)]
        if asof:
            query += " WHERE asof = ?"
            params.append(asof)
        with duckdb.connect() as con:
            return con.execute(query, params).df()

    def write_earnings(self, symbol: str, earnings: pd.DataFrame) -> Path:
        """Append/dedupe earnings calendar rows for ``symbol``."""
        path = self._earnings_path(symbol)
        if earnings is None or earnings.empty:
            return path
        frame = earnings.copy()
        frame["symbol"] = symbol.upper()
        if path.exists():
            existing = pd.read_parquet(path)
            frame = pd.concat([existing, frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["symbol", "earnings_date"], keep="last")
        frame = frame.sort_values("earnings_date").reset_index(drop=True)
        frame.to_parquet(path, index=False)
        return path

    def _universe_path(self, name: str) -> Path:
        return self.data_root / "universe" / f"{_checked_name(name)}.parquet"

    def write_universe(self, name: str, universe: pd.DataFrame, *, as_of: date) -> Path:
        """Replace the current ``name`` universe and keep a dated snapshot of it.

        Snapshots preserve membership as of each fetch, since vendor lists only
        describe the present.
        """
        frame = universe.copy()
        frame["list_date"] = as_of.isoformat()
        path = self._universe_path(name)
        snapshot = (
            self.data_root / "universe" / "snapshots" / _checked_name(name) / f"{as_of.isoformat()}.parquet"
        )
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(snapshot, index=False)
        frame.to_parquet(path, index=False)
        return path

    def read_universe(self, name: str, category: str | None = None) -> pd.DataFrame:
        path = self._universe_path(name)
        if not path.exists():
            return pd.DataFrame(columns=["symbol", "name", "category", "list_date"])
        frame = pd.read_parquet(path)
        if category is not None:
            frame = frame[frame["category"] == category]
        return frame.reset_index(drop=True)

    def earnings_day_path(self, day: date) -> Path:
        return self.earnings_dir / "calendar" / f"{day.year:04d}" / f"{day.isoformat()}.parquet"

    def write_earnings_day(self, day: date, rows: pd.DataFrame) -> Path:
        """Cache one calendar day; empty days are written too so they are not refetched."""
        path = self.earnings_day_path(day)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(path, index=False)
        return path

    def read_earnings_day(self, day: date) -> pd.DataFrame | None:
        path = self.earnings_day_path(day)
        return pd.read_parquet(path) if path.exists() else None

    def cached_earnings_days(self) -> set[date]:
        root = self.earnings_dir / "calendar"
        if not root.is_dir():
            return set()
        return {date.fromisoformat(path.stem) for path in root.glob("*/*.parquet")}

    def read_earnings_calendar(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        symbols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """All cached calendar rows, optionally filtered by date range and symbols."""
        root = self.earnings_dir / "calendar"
        if not root.is_dir() or not any(root.glob("*/*.parquet")):
            return pd.DataFrame()
        query = "SELECT * FROM read_parquet(?, union_by_name = true)"
        clauses: list[str] = []
        params: list[Any] = [str(root / "*" / "*.parquet")]
        if start:
            clauses.append("earnings_date >= ?")
            params.append(start)
        if end:
            clauses.append("earnings_date <= ?")
            params.append(end)
        if symbols is not None:
            if not symbols:
                return pd.DataFrame()
            clauses.append("list_contains(?, symbol)")
            params.append([symbol.upper() for symbol in symbols])
        if clauses:
            query = f"{query} WHERE {' AND '.join(clauses)}"
        query += " ORDER BY earnings_date, symbol"
        with duckdb.connect() as con:
            return con.execute(query, params).df()

    def _events_path(self, name: str) -> Path:
        return self.earnings_dir / "events" / f"{_checked_name(name)}.parquet"

    def write_earnings_events(self, name: str, events: pd.DataFrame) -> Path:
        path = self._events_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(path, index=False)
        return path

    def read_earnings_events(self, name: str) -> pd.DataFrame:
        path = self._events_path(name)
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def read_earnings_many(self, symbols: Sequence[str]) -> pd.DataFrame:
        """Stored per-symbol earnings rows for many symbols in one query."""
        return self._read_many(
            [self._earnings_path(symbol) for symbol in symbols], order_by="symbol, earnings_date"
        )

    def read_earnings(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        path = self._earnings_path(symbol)
        if not path.exists():
            return pd.DataFrame(columns=["symbol", "earnings_date", "timing", "source", "fetched_at"])
        query = "SELECT * FROM read_parquet(?)"
        clauses: list[str] = []
        params: list[str] = [str(path)]
        if start:
            clauses.append("earnings_date >= ?")
            params.append(start)
        if end:
            clauses.append("earnings_date <= ?")
            params.append(end)
        if clauses:
            query = f"{query} WHERE {' AND '.join(clauses)}"
        query += " ORDER BY earnings_date"
        with duckdb.connect() as con:
            return con.execute(query, params).df()
