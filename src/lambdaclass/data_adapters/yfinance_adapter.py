from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

BAR_COLUMNS = ["date", "open", "high", "low", "close", "volume", "dividends"]
YAHOO_EARNINGS_COLUMNS = [
    "earnings_date",
    "timing",
    "announce_time",
    "eps_estimate",
    "eps_actual",
    "surprise_pct",
]
YAHOO_EARNINGS_LIMIT = 100
MARKET_OPEN_HOUR = 9.5
MARKET_CLOSE_HOUR = 16.0
DUPLICATE_WINDOW_DAYS = 3
_NEW_YORK = "America/New_York"


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

    def get_earnings_dates(self, symbol: str, limit: int = YAHOO_EARNINGS_LIMIT) -> pd.DataFrame:
        """Up to ``limit`` (max 100) past and upcoming reports; see ``parse_yahoo_earnings_dates``.

        Network and parsing errors propagate so callers can retry; symbols without
        earnings (ETFs, unknown tickers) return an empty frame.
        """
        raw = yf.Ticker(symbol).get_earnings_dates(limit=min(limit, YAHOO_EARNINGS_LIMIT))
        return parse_yahoo_earnings_dates(raw)


def parse_yahoo_earnings_dates(raw: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize ``Ticker.get_earnings_dates`` output to ``YAHOO_EARNINGS_COLUMNS``.

    The index holds the announcement time. Before 09:30 New York time is ``BMO``,
    16:00 or later is ``AMC``, and in-session releases are ``unknown``. Yahoo marks
    a report with no known time as midnight UTC (19:00/20:00 New York the day
    before), so those keep their UTC date and get ``unknown`` timing.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(
            {column: pd.Series(dtype=_yahoo_dtype(column)) for column in YAHOO_EARNINGS_COLUMNS}
        )
    stamps = pd.DatetimeIndex(pd.to_datetime(raw.index))
    if stamps.tz is None:
        stamps = stamps.tz_localize(_NEW_YORK)
    utc = stamps.tz_convert("UTC")
    local = stamps.tz_convert(_NEW_YORK)
    date_only = (utc.hour == 0) & (utc.minute == 0) & (utc.second == 0)
    hours = local.hour + local.minute / 60.0
    timing = np.where(hours < MARKET_OPEN_HOUR, "BMO", np.where(hours >= MARKET_CLOSE_HOUR, "AMC", "unknown"))
    frame = pd.DataFrame(
        {
            "earnings_date": np.where(date_only, utc.strftime("%Y-%m-%d"), local.strftime("%Y-%m-%d")),
            "timing": np.where(date_only, "unknown", timing),
            "announce_time": pd.Series(
                [
                    None if unknown else text
                    for text, unknown in zip(local.strftime("%H:%M"), date_only, strict=True)
                ],
                dtype=object,
            ),
            "eps_estimate": _numeric_column(raw, "EPS Estimate"),
            "eps_actual": _numeric_column(raw, "Reported EPS"),
            "surprise_pct": _numeric_column(raw, "Surprise(%)"),
        }
    )
    return _collapse_near_duplicates(frame)[YAHOO_EARNINGS_COLUMNS]


def _collapse_near_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Yahoo sometimes lists one report twice, days apart; keep one row per cluster.

    Rows within ``DUPLICATE_WINDOW_DAYS`` of the previous row form a cluster; the
    row with a known announcement time wins, else the earliest.
    """
    ordered = frame.assign(_known=frame["timing"].ne("unknown")).sort_values("earnings_date")
    days = pd.to_datetime(ordered["earnings_date"])
    cluster = (days.diff().dt.days.fillna(DUPLICATE_WINDOW_DAYS + 1) > DUPLICATE_WINDOW_DAYS).cumsum()
    best = ordered.assign(_cluster=cluster.to_numpy()).sort_values(
        ["_cluster", "_known", "earnings_date"], ascending=[True, False, True], kind="stable"
    )
    return (
        best.drop_duplicates(subset=["_cluster"], keep="first")
        .sort_values("earnings_date")
        .reset_index(drop=True)
    )


def _numeric_column(raw: pd.DataFrame, column: str) -> np.ndarray:
    if column not in raw.columns:
        return np.full(len(raw), np.nan)
    return pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float)


def _yahoo_dtype(column: str) -> str:
    return "float64" if column in {"eps_estimate", "eps_actual", "surprise_pct"} else "object"
