from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

BAR_COLUMNS = ["date", "open", "high", "low", "close", "volume", "dividends"]


class YFinanceAdapter:
    def get_stock_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Daily bars for ``start``..``end`` inclusive.

        ``close`` is split-adjusted but not dividend-adjusted, so it stays
        comparable with option strikes; cash dividends arrive in ``dividends``
        on the ex-date instead.
        """
        ticker = yf.Ticker(symbol)
        # yfinance treats ``end`` as exclusive.
        exclusive_end = end + timedelta(days=1)
        bars = ticker.history(start=start.isoformat(), end=exclusive_end.isoformat(), auto_adjust=False)
        if bars.empty:
            return pd.DataFrame(columns=BAR_COLUMNS)
        bars = bars.reset_index()
        bars.columns = [col.lower().replace(" ", "_") for col in bars.columns]
        bars = bars.rename(columns={"datetime": "date"})
        for field in BAR_COLUMNS:
            if field not in bars:
                bars[field] = 0.0
        bars["dividends"] = pd.to_numeric(bars["dividends"], errors="coerce").fillna(0.0).astype(float)
        bars["date"] = pd.to_datetime(bars["date"]).dt.date.astype(str)
        return bars[BAR_COLUMNS]

    def get_option_chain(self, symbol: str, asof: date, expiry: date | None = None) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        current_asof = date.today()
        expirations = ticker.options or []
        if not expirations:
            return pd.DataFrame()
        selected_expiries = [expiry.isoformat()] if expiry else expirations[:3]
        frames: list[pd.DataFrame] = []
        for exp in selected_expiries:
            if exp not in expirations:
                continue
            chain = ticker.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["side"] = "call"
            puts["side"] = "put"
            merged = pd.concat([calls, puts], ignore_index=True)
            merged["expiry"] = exp
            merged["asof"] = current_asof.isoformat()
            frames.append(merged)
        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, ignore_index=True)
        needed = [
            "contractSymbol",
            "side",
            "strike",
            "lastPrice",
            "bid",
            "ask",
            "impliedVolatility",
            "openInterest",
            "volume",
            "expiry",
            "asof",
        ]
        for col in needed:
            if col not in frame:
                frame[col] = 0.0
        frame = frame.rename(
            columns={
                "contractSymbol": "contract_symbol",
                "lastPrice": "last_price",
                "impliedVolatility": "implied_volatility",
                "openInterest": "open_interest",
            }
        )
        return frame[
            [
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
            ]
        ]

    def get_earnings_dates(self, symbol: str, limit: int = 40) -> pd.DataFrame:
        """Return raw earnings dates; empty frame when unavailable.

        Columns: ``earnings_date``, ``timing`` (vendor string; normalize via
        ``lambdaclass.earnings.calendar.normalize_earnings_frame``).
        """
        ticker = yf.Ticker(symbol)
        raw: pd.DataFrame | None = None
        try:
            if hasattr(ticker, "get_earnings_dates"):
                raw = ticker.get_earnings_dates(limit=limit)
            elif hasattr(ticker, "earnings_dates") and ticker.earnings_dates is not None:
                raw = ticker.earnings_dates
        except Exception:
            return pd.DataFrame(columns=["earnings_date", "timing"])
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["earnings_date", "timing"])
        frame = raw.reset_index()
        # Index is often the earnings datetime
        date_col = None
        for candidate in ("Earnings Date", "earnings_date", "Date", "index", frame.columns[0]):
            if candidate in frame.columns:
                date_col = candidate
                break
        if date_col is None:
            return pd.DataFrame(columns=["earnings_date", "timing"])
        timing_col = None
        for candidate in ("Event Type", "Earnings Timing", "timing", "Time"):
            if candidate in frame.columns:
                timing_col = candidate
                break
        out = pd.DataFrame(
            {
                "earnings_date": pd.to_datetime(frame[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
                "timing": frame[timing_col] if timing_col else "unknown",
            }
        )
        out = out.dropna(subset=["earnings_date"])
        return out.reset_index(drop=True)
