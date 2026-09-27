"""Earnings calendar helpers (pure; no Streamlit / network)."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

EARNINGS_COLUMNS = ("symbol", "earnings_date", "timing", "source", "fetched_at")
VALID_TIMINGS = frozenset({"BMO", "AMC", "unknown"})


def parse_date(d: str | date | datetime) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return date.fromisoformat(str(d).strip()[:10])


def normalize_timing(raw: object) -> str:
    """Map vendor strings to BMO | AMC | unknown."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "unknown"
    s = str(raw).strip().upper()
    if not s or s in ("NAN", "NONE", "NAT"):
        return "unknown"
    if "BMO" in s or s in ("BEFORE MARKET OPEN", "BEFORE OPEN", "PREMARKET"):
        return "BMO"
    if "AMC" in s or s in ("AFTER MARKET CLOSE", "AFTER CLOSE", "AFTER-MARKET", "TAS"):
        # TAS (time as scheduled) often after close for US names — treat as AMC
        return "AMC"
    if s in VALID_TIMINGS:
        return s
    return "unknown"


def empty_earnings_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(EARNINGS_COLUMNS))


def normalize_earnings_frame(df: pd.DataFrame, *, symbol: str, source: str) -> pd.DataFrame:
    """Normalize a raw earnings table to the canonical schema."""
    if df is None or df.empty:
        return empty_earnings_frame()
    frame = df.copy()
    # Accept common aliases
    rename = {}
    cols_lower = {c.lower().strip(): c for c in frame.columns}
    if "earnings_date" not in frame.columns:
        for alias in ("earnings date", "date", "reportdate", "report_date"):
            if alias in cols_lower:
                rename[cols_lower[alias]] = "earnings_date"
                break
    if "timing" not in frame.columns:
        for alias in ("earnings time", "time", "event"):
            if alias in cols_lower:
                rename[cols_lower[alias]] = "timing"
                break
    if rename:
        frame = frame.rename(columns=rename)
    if "earnings_date" not in frame.columns:
        raise ValueError("earnings frame requires earnings_date column")
    if "timing" not in frame.columns:
        frame["timing"] = "unknown"
    out = pd.DataFrame(
        {
            "symbol": symbol.upper(),
            "earnings_date": frame["earnings_date"].map(lambda x: parse_date(x).isoformat()),
            "timing": frame["timing"].map(normalize_timing),
            "source": source,
            "fetched_at": datetime.now(tz=UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    out = out.drop_duplicates(subset=["symbol", "earnings_date"], keep="last")
    return out.sort_values("earnings_date").reset_index(drop=True)


@dataclass(frozen=True)
class _EarningsEntry:
    earnings_date: date
    timing: str
    source_position: int

    def as_row(self) -> dict[str, Any]:
        return {"earnings_date": self.earnings_date.isoformat(), "timing": self.timing}


class EarningsCalendar:
    """Parsed, sorted earnings dates with logarithmic date lookups."""

    def __init__(self, earnings: pd.DataFrame | None) -> None:
        entries: list[_EarningsEntry] = []
        if earnings is not None and not earnings.empty:
            for position, (_, row) in enumerate(earnings.iterrows()):
                entries.append(
                    _EarningsEntry(
                        earnings_date=parse_date(row["earnings_date"]),
                        timing=normalize_timing(row.get("timing", "unknown")),
                        source_position=position,
                    )
                )
        self._entries = sorted(entries, key=lambda entry: entry.earnings_date)
        self._dates = [entry.earnings_date for entry in self._entries]
        self._first_by_date: dict[date, _EarningsEntry] = {}
        self._first_bmo_by_date: dict[date, _EarningsEntry] = {}
        for entry in entries:
            self._first_by_date.setdefault(entry.earnings_date, entry)
            if entry.timing == "BMO":
                self._first_bmo_by_date.setdefault(entry.earnings_date, entry)

    def _next(self, bar_date: date) -> _EarningsEntry | None:
        position = bisect_left(self._dates, bar_date)
        return self._entries[position] if position < len(self._entries) else None

    def _previous(self, bar_date: date) -> _EarningsEntry | None:
        same_day_bmo = self._first_bmo_by_date.get(bar_date)
        if same_day_bmo is not None:
            return same_day_bmo
        same_day_start = bisect_left(self._dates, bar_date)
        if same_day_start == 0:
            return None
        previous_date = self._entries[same_day_start - 1].earnings_date
        return self._first_by_date[previous_date]

    def next_earnings_row(self, bar_date: str | date) -> dict[str, Any] | None:
        entry = self._next(parse_date(bar_date))
        return None if entry is None else entry.as_row()

    def previous_earnings_row(self, bar_date: str | date) -> dict[str, Any] | None:
        entry = self._previous(parse_date(bar_date))
        return None if entry is None else entry.as_row()

    def days_to_next_earnings(self, bar_date: str | date) -> int | None:
        parsed = parse_date(bar_date)
        entry = self._next(parsed)
        return None if entry is None else (entry.earnings_date - parsed).days

    def days_since_last_earnings(self, bar_date: str | date) -> int | None:
        parsed = parse_date(bar_date)
        entry = self._previous(parsed)
        return None if entry is None else (parsed - entry.earnings_date).days

    def context_fields(self, bar_date: str | date) -> dict[str, Any]:
        parsed = parse_date(bar_date)
        next_entry = self._next(parsed)
        previous_entry = self._previous(parsed)
        return {
            "days_to_next_earnings": (
                None if next_entry is None else (next_entry.earnings_date - parsed).days
            ),
            "days_since_last_earnings": (
                None if previous_entry is None else (parsed - previous_entry.earnings_date).days
            ),
            "next_earnings_date": (None if next_entry is None else next_entry.earnings_date.isoformat()),
            "earnings_timing": None if next_entry is None else next_entry.timing,
        }

    def nearest_earnings_row(
        self, bar_date: str | date, *, max_days: int | None = None
    ) -> dict[str, Any] | None:
        parsed = parse_date(bar_date)
        position = bisect_left(self._dates, parsed)
        candidate_dates: set[date] = set()
        if position < len(self._dates):
            candidate_dates.add(self._dates[position])
        if position:
            candidate_dates.add(self._dates[position - 1])
        candidates = [self._first_by_date[candidate_date] for candidate_date in candidate_dates]
        if not candidates:
            return None
        best = min(
            candidates,
            key=lambda entry: (
                abs((entry.earnings_date - parsed).days),
                entry.source_position,
            ),
        )
        if max_days is not None and abs((best.earnings_date - parsed).days) > max_days:
            return None
        return best.as_row()


def next_earnings_row(bar_date: str | date, earnings: pd.DataFrame) -> dict[str, Any] | None:
    """Nearest earnings on or after ``bar_date`` (inclusive)."""
    return EarningsCalendar(earnings).next_earnings_row(bar_date)


def previous_earnings_row(bar_date: str | date, earnings: pd.DataFrame) -> dict[str, Any] | None:
    """Nearest earnings strictly before ``bar_date``, or same-day AMC after the release session.

    For countdown ``days_since``: last earnings date that has already occurred relative to the bar.
    Same-day BMO: event has occurred (days_since=0). Same-day AMC: event not yet (use previous).
    """
    return EarningsCalendar(earnings).previous_earnings_row(bar_date)


def days_to_next_earnings(bar_date: str | date, earnings: pd.DataFrame) -> int | None:
    return EarningsCalendar(earnings).days_to_next_earnings(bar_date)


def days_since_last_earnings(bar_date: str | date, earnings: pd.DataFrame) -> int | None:
    return EarningsCalendar(earnings).days_since_last_earnings(bar_date)


def context_fields(bar_date: str | date, earnings: pd.DataFrame) -> dict[str, Any]:
    """Fields to attach to StrategyContext for ``bar_date``."""
    return EarningsCalendar(earnings).context_fields(bar_date)
