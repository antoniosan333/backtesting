"""Plan and merge day-by-day earnings calendar fetches (pure; no network)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

import numpy as np
import pandas as pd

from lambdaclass.earnings.calendar import EARNINGS_COLUMNS

KNOWN_TIMINGS = frozenset({"BMO", "AMC"})
MATCH_TOLERANCE_DAYS = 3
EPS_COLUMNS = ("eps_actual", "eps_estimate", "surprise_pct")
COMBINED_COLUMNS = [
    "symbol",
    "earnings_date",
    "timing",
    "timing_vendor",
    "announce_time",
    "source",
    *EPS_COLUMNS,
]
SYMBOL_EARNINGS_EXTRA_COLUMNS = (*EPS_COLUMNS, "announce_time", "timing_vendor", "timing_source")


def plan_calendar_days(
    start: date,
    end: date,
    *,
    cached: Iterable[date],
    today: date,
    refresh_days: int = 7,
    include_weekends: bool = False,
) -> list[date]:
    """Days in ``[start, end]`` that need a (re)fetch.

    Cached days are skipped unless they fall within ``refresh_days`` before
    ``today`` or later: recent and upcoming days change as EPS is reported,
    timing is announced, or companies reschedule.
    """
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    if refresh_days < 0:
        raise ValueError("refresh_days must be >= 0")
    cached_days = set(cached)
    refresh_from = today - timedelta(days=refresh_days)
    days: list[date] = []
    current = start
    while current <= end:
        if (include_weekends or current.weekday() < 5) and (
            current not in cached_days or current >= refresh_from
        ):
            days.append(current)
        current += timedelta(days=1)
    return days


def merge_calendar_day(previous: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    """Combine a refetched day with its cached copy.

    Nasdaq drops the BMO/AMC time once a day is in the past, so a known timing
    from an earlier fetch is kept when the refetch says ``unknown``. An empty
    refetch over a non-empty cache is treated as a vendor glitch and ignored.
    """
    if previous is None or previous.empty:
        return fresh
    if fresh.empty:
        return previous
    known = previous[previous["timing"].isin(KNOWN_TIMINGS)].set_index("symbol")["timing"]
    if known.empty:
        return fresh
    merged = fresh.copy()
    carry = merged["timing"].eq("unknown") & merged["symbol"].isin(known.index)
    merged.loc[carry, "timing"] = merged.loc[carry, "symbol"].map(known)
    return merged


def combine_earnings_sources(
    nasdaq: pd.DataFrame,
    yahoo: pd.DataFrame,
    *,
    tolerance_days: int = MATCH_TOLERANCE_DAYS,
) -> pd.DataFrame:
    """One row per report from Nasdaq day files and Yahoo per-symbol histories.

    A Yahoo report matches the nearest unmatched Nasdaq report of the same symbol
    within ``tolerance_days`` (quarterly reports are ~90 days apart). For a pair:

    * timing: Yahoo's timestamp-derived BMO/AMC, else Nasdaq's, else ``unknown``
      (``timing_vendor`` records which one);
    * date: Yahoo's when it knows the time of day, else Nasdaq's;
    * EPS fields: Nasdaq's, filled from Yahoo where missing.
    """
    nasdaq_rows = _source_frame(nasdaq, "nasdaq")
    yahoo_rows = _source_frame(yahoo, "yahoo")
    if nasdaq_rows.empty or yahoo_rows.empty:
        return _finish_combined(pd.concat([nasdaq_rows, yahoo_rows], ignore_index=True))
    right = nasdaq_rows.add_prefix("n_").assign(
        _day=pd.to_datetime(nasdaq_rows["earnings_date"]).to_numpy(),
        symbol=nasdaq_rows["symbol"].to_numpy(),
        n_row=np.arange(len(nasdaq_rows)),
    )
    left = yahoo_rows.assign(_day=pd.to_datetime(yahoo_rows["earnings_date"]).to_numpy())
    pairs = pd.merge_asof(
        left.sort_values("_day"),
        right.sort_values("_day"),
        on="_day",
        by="symbol",
        direction="nearest",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    pairs.loc[pairs["n_row"].duplicated() & pairs["n_row"].notna(), "n_row"] = np.nan
    paired = pairs["n_row"].notna()
    yahoo_known = pairs["timing"].isin(KNOWN_TIMINGS)
    nasdaq_timing = paired & ~yahoo_known & pairs["n_timing"].isin(KNOWN_TIMINGS)
    combined = pd.DataFrame(
        {
            "symbol": pairs["symbol"],
            "earnings_date": pairs["earnings_date"].where(~paired | yahoo_known, pairs["n_earnings_date"]),
            "timing": pairs["timing"].where(~nasdaq_timing, pairs["n_timing"]),
            "timing_vendor": np.where(yahoo_known, "yahoo", np.where(nasdaq_timing, "nasdaq", None)),
            "announce_time": pairs["announce_time"],
            "source": np.where(paired, "nasdaq+yahoo", "yahoo"),
        }
    )
    for column in EPS_COLUMNS:
        nasdaq_value = pairs[f"n_{column}"].where(paired)
        combined[column] = nasdaq_value.where(nasdaq_value.notna(), pairs[column])
    unmatched = ~np.isin(np.arange(len(nasdaq_rows)), pairs.loc[paired, "n_row"].to_numpy())
    return _finish_combined(pd.concat([combined, nasdaq_rows[unmatched]], ignore_index=True))


def _source_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=COMBINED_COLUMNS)
    rows = frame.reindex(columns=[column for column in COMBINED_COLUMNS if column != "source"]).copy()
    rows["earnings_date"] = rows["earnings_date"].astype(str).str[:10]
    rows["timing"] = rows["timing"].where(rows["timing"].isin(KNOWN_TIMINGS), "unknown")
    rows["timing_vendor"] = np.where(rows["timing"].isin(KNOWN_TIMINGS), source, None)
    rows["source"] = source
    for column in EPS_COLUMNS:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows.drop_duplicates(subset=["symbol", "earnings_date"], keep="first")


def _finish_combined(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.reindex(columns=COMBINED_COLUMNS)
    rows = rows.drop_duplicates(subset=["symbol", "earnings_date"], keep="first")
    return rows.sort_values(["symbol", "earnings_date"]).reset_index(drop=True)


def symbol_earnings_frames(calendar: pd.DataFrame, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Split calendar rows into per-symbol frames in the canonical earnings schema.

    Only ``symbols`` are kept; the result feeds ``DuckDBStore.write_earnings`` so
    ``lambdaclass run`` sees the same dates through ``StrategyContext``.
    """
    wanted = {symbol.upper() for symbol in symbols}
    if calendar.empty or not wanted:
        return {}
    rows = calendar[calendar["symbol"].isin(wanted)].copy()
    if rows.empty:
        return {}
    if "timing_source" not in rows.columns:
        rows["timing_source"] = rows["timing"].map(
            lambda value: "vendor" if value in KNOWN_TIMINGS else "none"
        )
    columns = [*EARNINGS_COLUMNS, *SYMBOL_EARNINGS_EXTRA_COLUMNS]
    rows = rows.reindex(columns=columns)
    rows["earnings_date"] = rows["earnings_date"].astype(str).str[:10]
    return {
        str(symbol): frame.sort_values("earnings_date").reset_index(drop=True)
        for symbol, frame in rows.groupby("symbol", sort=True)
    }
