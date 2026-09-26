"""Earnings calendar helpers (pure; no Streamlit / network)."""

from __future__ import annotations

from datetime import date, datetime, timezone
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
            "fetched_at": datetime.now(tz=timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    out = out.drop_duplicates(subset=["symbol", "earnings_date"], keep="last")
    return out.sort_values("earnings_date").reset_index(drop=True)


def next_earnings_row(bar_date: str | date, earnings: pd.DataFrame) -> dict[str, Any] | None:
    """Nearest earnings on or after ``bar_date`` (inclusive)."""
    if earnings is None or earnings.empty:
        return None
    bd = parse_date(bar_date)
    dates = pd.to_datetime(earnings["earnings_date"], errors="coerce")
    mask = dates.notna() & (dates.dt.date >= bd)
    sub = earnings.loc[mask].copy()
    if sub.empty:
        return None
    sub = sub.assign(_d=dates.loc[mask]).sort_values("_d")
    row = sub.iloc[0]
    return {
        "earnings_date": str(row["earnings_date"])[:10],
        "timing": normalize_timing(row.get("timing", "unknown")),
    }


def previous_earnings_row(bar_date: str | date, earnings: pd.DataFrame) -> dict[str, Any] | None:
    """Nearest earnings strictly before ``bar_date``, or same-day AMC after the release session.

    For countdown ``days_since``: last earnings date that has already occurred relative to the bar.
    Same-day BMO: event has occurred (days_since=0). Same-day AMC: event not yet (use previous).
    """
    if earnings is None or earnings.empty:
        return None
    bd = parse_date(bar_date)
    best: dict[str, Any] | None = None
    best_d: date | None = None
    for _, row in earnings.iterrows():
        ed = parse_date(row["earnings_date"])
        timing = normalize_timing(row.get("timing", "unknown"))
        if ed < bd:
            if best_d is None or ed > best_d:
                best_d = ed
                best = {"earnings_date": ed.isoformat(), "timing": timing}
        elif ed == bd and timing == "BMO":
            # Release already happened at open
            return {"earnings_date": ed.isoformat(), "timing": timing}
    return best


def days_to_next_earnings(bar_date: str | date, earnings: pd.DataFrame) -> int | None:
    nxt = next_earnings_row(bar_date, earnings)
    if nxt is None:
        return None
    return (parse_date(nxt["earnings_date"]) - parse_date(bar_date)).days


def days_since_last_earnings(bar_date: str | date, earnings: pd.DataFrame) -> int | None:
    prev = previous_earnings_row(bar_date, earnings)
    if prev is None:
        bd = parse_date(bar_date)
        # Same-day AMC: release not yet — no "since"
        for _, row in earnings.iterrows():
            ed = parse_date(row["earnings_date"])
            timing = normalize_timing(row.get("timing", "unknown"))
            if ed == bd and timing != "BMO":
                return None
        return None
    return (parse_date(bar_date) - parse_date(prev["earnings_date"])).days


def context_fields(bar_date: str | date, earnings: pd.DataFrame) -> dict[str, Any]:
    """Fields to attach to StrategyContext for ``bar_date``."""
    nxt = next_earnings_row(bar_date, earnings)
    return {
        "days_to_next_earnings": days_to_next_earnings(bar_date, earnings),
        "days_since_last_earnings": days_since_last_earnings(bar_date, earnings),
        "next_earnings_date": None if nxt is None else nxt["earnings_date"],
        "earnings_timing": None if nxt is None else nxt["timing"],
    }
