"""Plan and merge day-by-day earnings calendar fetches (pure; no network)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

import pandas as pd

from lambdaclass.earnings.calendar import EARNINGS_COLUMNS

KNOWN_TIMINGS = frozenset({"BMO", "AMC"})
SYMBOL_EARNINGS_EXTRA_COLUMNS = ("eps_actual", "eps_estimate", "surprise_pct", "timing_source")


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
