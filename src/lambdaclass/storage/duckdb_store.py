from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


class DuckDBStore:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.stocks_dir = data_root / "stocks"
        self.options_dir = data_root / "options"
        self.earnings_dir = data_root / "earnings"
        self.cache_dir = data_root / "cache"
        self.stocks_dir.mkdir(parents=True, exist_ok=True)
        self.options_dir.mkdir(parents=True, exist_ok=True)
        self.earnings_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
            return con.execute(query, params).df()

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

    def read_earnings(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        path = self._earnings_path(symbol)
        if not path.exists():
            return pd.DataFrame(
                columns=["symbol", "earnings_date", "timing", "source", "fetched_at"]
            )
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
