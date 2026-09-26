"""Earnings price-reaction events and per-symbol statistics (pure; no network).

Each calendar row is matched to daily bars. The *reaction bar* is the first
session that trades on the news: the report day for BMO (before market open),
the next session for AMC (after market close). Historical vendor timing is often
missing, so it is inferred from which overnight gap is larger:

* BMO when ``|open[D] / close[D-1] - 1| >= |open[D+1] / close[D] - 1|``
* AMC otherwise

``timing_confidence`` is the larger gap's share of both gaps (0.5 = coin flip,
1.0 = only one gap moved). Companies rarely change their reporting slot, so a
per-symbol vote (``symbol_timing``) overrides gap guesses below
``CONSENSUS_OVERRIDE_BELOW`` confidence (``timing_source = inferred_symbol``);
this matters for low-volatility names whose earnings gaps look like noise.
Inferred timing uses the reaction itself, so treat it as a labelling aid for
research, not something a live strategy knew in advance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lambdaclass.earnings.history import KNOWN_TIMINGS

DRIFT_HORIZONS = (1, 5, 20)
RUNUP_DAYS = 5
VOL_WINDOW = 20
CONSENSUS_MIN_SHARE = 0.2
CONSENSUS_OVERRIDE_BELOW = 0.9
VENDOR_VOTE_WEIGHT = 0.5
EVENT_COLUMNS = [
    "symbol",
    "earnings_date",
    "timing",
    "timing_source",
    "vendor_timing",
    "inferred_timing",
    "timing_confidence",
    "symbol_timing",
    "pre_date",
    "reaction_date",
    "pre_close",
    "reaction_open",
    "reaction_close",
    "gap_return",
    "reaction_return",
    "intraday_return",
    "abs_reaction",
    *(f"drift_{days}d" for days in DRIFT_HORIZONS),
    f"runup_{RUNUP_DAYS}d",
    f"vol_{VOL_WINDOW}d",
    "move_sigma",
    "volume_ratio",
    "eps_actual",
    "eps_estimate",
    "surprise_pct",
    "beat",
]


@dataclass(frozen=True)
class BarSeries:
    dates: np.ndarray
    open: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    dividends: np.ndarray
    daily_returns: np.ndarray

    @classmethod
    def from_frame(cls, bars: pd.DataFrame) -> BarSeries:
        frame = bars.sort_values("date")
        close = frame["close"].to_numpy(dtype=float)
        dividends = (
            frame["dividends"].fillna(0.0).to_numpy(dtype=float)
            if "dividends" in frame.columns
            else np.zeros(len(frame))
        )
        returns = np.full(len(close), np.nan)
        if len(close) > 1:
            returns[1:] = (close[1:] + dividends[1:]) / close[:-1] - 1.0
        return cls(
            dates=frame["date"].astype(str).str[:10].to_numpy(),
            open=frame["open"].to_numpy(dtype=float),
            close=close,
            volume=frame["volume"].to_numpy(dtype=float)
            if "volume" in frame.columns
            else np.full(len(frame), np.nan),
            dividends=dividends,
            daily_returns=returns,
        )

    def __len__(self) -> int:
        return len(self.dates)


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator - 1.0)


def infer_timing(bars: BarSeries, index: int) -> tuple[str | None, float]:
    """Infer BMO/AMC for a report dated ``bars.dates[index]`` from the two overnight gaps."""
    if index < 1 or index + 1 >= len(bars):
        return None, float("nan")
    report_gap = abs(_ratio(bars.open[index] + bars.dividends[index], bars.close[index - 1]))
    next_gap = abs(_ratio(bars.open[index + 1] + bars.dividends[index + 1], bars.close[index]))
    if not np.isfinite(report_gap) or not np.isfinite(next_gap):
        return None, float("nan")
    total = report_gap + next_gap
    confidence = 0.5 if total == 0 else max(report_gap, next_gap) / total
    return ("BMO" if report_gap >= next_gap else "AMC"), float(confidence)


@dataclass(frozen=True)
class _Placement:
    """Where one calendar row lands on the bars, with vendor and gap-inferred timing."""

    earnings_date: str
    index: int
    trading_day: bool
    vendor_timing: str | None
    inferred: str | None
    confidence: float


def _place(bars: BarSeries, calendar_row: Mapping[str, Any]) -> _Placement:
    earnings_date = str(calendar_row["earnings_date"])[:10]
    index = int(np.searchsorted(bars.dates, earnings_date, side="left"))
    trading_day = index < len(bars) and bars.dates[index] == earnings_date
    vendor = str(calendar_row.get("timing") or "unknown")
    inferred, confidence = infer_timing(bars, index) if trading_day else (None, float("nan"))
    return _Placement(
        earnings_date=earnings_date,
        index=index,
        trading_day=trading_day,
        vendor_timing=vendor if vendor in KNOWN_TIMINGS else None,
        inferred=inferred,
        confidence=confidence,
    )


def symbol_timing_vote(placements: list[_Placement]) -> str | None:
    """A symbol's habitual BMO/AMC slot from a confidence-weighted vote over its reports.

    Vendor timings (including upcoming reports) count as full-strength votes; each
    inferred timing counts ``confidence - 0.5``. Returns ``None`` when the net vote
    is under ``CONSENSUS_MIN_SHARE`` of the total weight.
    """
    net = total = 0.0
    for placement in placements:
        if placement.vendor_timing is not None:
            timing, weight = placement.vendor_timing, VENDOR_VOTE_WEIGHT
        elif placement.inferred is not None:
            timing, weight = placement.inferred, placement.confidence - 0.5
        else:
            continue
        net += weight if timing == "BMO" else -weight
        total += weight
    if total <= 0 or abs(net) / total < CONSENSUS_MIN_SHARE:
        return None
    return "BMO" if net > 0 else "AMC"


def _resolve_timing(placement: _Placement, symbol_timing: str | None) -> tuple[str, str, int] | None:
    """``(timing, timing_source, reaction_index)`` for one placement, or ``None`` if undecidable."""
    index = placement.index
    if not placement.trading_day:
        # Reported on a weekend/holiday: the next session is the first to react either way.
        return placement.vendor_timing or "BMO", "non_trading_day", index
    if placement.vendor_timing is not None:
        timing, source = placement.vendor_timing, "vendor"
    elif (
        symbol_timing is not None
        and symbol_timing != placement.inferred
        and (placement.inferred is None or placement.confidence < CONSENSUS_OVERRIDE_BELOW)
    ):
        timing, source = symbol_timing, "inferred_symbol"
    elif placement.inferred is not None:
        timing, source = placement.inferred, "inferred_gap"
    else:
        return None
    return timing, source, index if timing == "BMO" else index + 1


def _event_row(
    bars: BarSeries,
    calendar_row: Mapping[str, Any],
    placement: _Placement,
    symbol_timing: str | None,
) -> dict[str, Any] | None:
    if placement.index >= len(bars):
        return None
    resolved = _resolve_timing(placement, symbol_timing)
    if resolved is None:
        return None
    timing, timing_source, reaction = resolved
    pre = reaction - 1
    if pre < 0 or reaction >= len(bars):
        return None

    pre_close = bars.close[pre]
    reaction_open = bars.open[reaction]
    reaction_close = bars.close[reaction]
    dividend = bars.dividends[reaction]
    reaction_return = _ratio(reaction_close + dividend, pre_close)
    vol_start = pre - VOL_WINDOW + 1
    vol = float(np.std(bars.daily_returns[vol_start : pre + 1], ddof=1)) if vol_start >= 1 else float("nan")
    volume_window = bars.volume[max(pre - VOL_WINDOW + 1, 0) : pre + 1]
    avg_volume = float(np.nanmean(volume_window)) if np.isfinite(volume_window).any() else float("nan")
    eps_actual = float(calendar_row.get("eps_actual", np.nan))
    eps_estimate = float(calendar_row.get("eps_estimate", np.nan))
    row: dict[str, Any] = {
        "symbol": str(calendar_row["symbol"]),
        "earnings_date": placement.earnings_date,
        "timing": timing,
        "timing_source": timing_source,
        "vendor_timing": placement.vendor_timing,
        "inferred_timing": placement.inferred,
        "timing_confidence": placement.confidence,
        "symbol_timing": symbol_timing,
        "pre_date": bars.dates[pre],
        "reaction_date": bars.dates[reaction],
        "pre_close": pre_close,
        "reaction_open": reaction_open,
        "reaction_close": reaction_close,
        "gap_return": _ratio(reaction_open + dividend, pre_close),
        "reaction_return": reaction_return,
        "intraday_return": _ratio(reaction_close, reaction_open),
        "abs_reaction": abs(reaction_return),
        f"runup_{RUNUP_DAYS}d": _ratio(pre_close, bars.close[pre - RUNUP_DAYS])
        if pre >= RUNUP_DAYS
        else np.nan,
        f"vol_{VOL_WINDOW}d": vol,
        "move_sigma": reaction_return / vol if np.isfinite(vol) and vol > 0 else np.nan,
        "volume_ratio": bars.volume[reaction] / avg_volume if avg_volume > 0 else np.nan,
        "eps_actual": eps_actual,
        "eps_estimate": eps_estimate,
        "surprise_pct": float(calendar_row.get("surprise_pct", np.nan)),
        "beat": (eps_actual > eps_estimate)
        if np.isfinite(eps_actual) and np.isfinite(eps_estimate)
        else pd.NA,
    }
    for days in DRIFT_HORIZONS:
        end = reaction + days
        row[f"drift_{days}d"] = _ratio(bars.close[end], reaction_close) if end < len(bars) else np.nan
    return row


def build_earnings_events(calendar: pd.DataFrame, bars_by_symbol: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per reported earnings event with bars on both sides of the reaction.

    ``calendar`` holds ``symbol, earnings_date, timing`` plus optional EPS columns;
    events without enough bars (upcoming reports, missing history) are skipped.
    Drift columns measure close-to-close returns *after* the reaction close.
    """
    rows: list[dict[str, Any]] = []
    if not calendar.empty:
        for symbol, group in calendar.groupby("symbol", sort=True):
            frame = bars_by_symbol.get(str(symbol))
            if frame is None or frame.empty:
                continue
            bars = BarSeries.from_frame(frame)
            calendar_rows = group.sort_values("earnings_date").to_dict("records")
            placements = [_place(bars, calendar_row) for calendar_row in calendar_rows]
            symbol_timing = symbol_timing_vote(placements)
            for calendar_row, placement in zip(calendar_rows, placements, strict=True):
                event = _event_row(bars, calendar_row, placement, symbol_timing)
                if event is not None:
                    rows.append(event)
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    events["beat"] = events["beat"].astype("boolean")
    return events


def apply_event_timing(earnings: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Copy event timing into earnings rows whose timing did not come from a vendor.

    Rows without ``timing_source`` (older files) count as vendor-timed when their
    timing is already BMO/AMC. Rerunning replaces earlier inferences.
    """
    if earnings.empty or events.empty:
        return earnings
    resolved = events[events["timing_source"] != "vendor"].set_index(["symbol", "earnings_date"])
    if resolved.empty:
        return earnings
    frame = earnings.copy()
    vendor_default = pd.Series(
        np.where(frame["timing"].isin(KNOWN_TIMINGS), "vendor", "none"), index=frame.index
    )
    if "timing_source" in frame.columns:
        frame["timing_source"] = frame["timing_source"].fillna(vendor_default)
    else:
        frame["timing_source"] = vendor_default
    keys = pd.MultiIndex.from_arrays(
        [frame["symbol"].astype(str), frame["earnings_date"].astype(str).str[:10]]
    )
    timing = pd.Series(resolved["timing"].reindex(keys).to_numpy(), index=frame.index)
    source = pd.Series(resolved["timing_source"].reindex(keys).to_numpy(), index=frame.index)
    fill = frame["timing_source"].ne("vendor") & timing.notna()
    frame.loc[fill, "timing"] = timing[fill]
    frame.loc[fill, "timing_source"] = source[fill]
    return frame


def _beat_rate(beats: pd.Series) -> float:
    known = beats.dropna()
    return float(known.astype(float).mean()) if len(known) else float("nan")


def summarize_events_by_symbol(events: pd.DataFrame, *, min_events: int = 1) -> pd.DataFrame:
    """Per-symbol reaction statistics, sorted by median absolute move (largest first)."""
    columns = [
        "symbol",
        "events",
        "first_event",
        "last_event",
        "mean_abs_move",
        "median_abs_move",
        "max_abs_move",
        "mean_move",
        "up_rate",
        "mean_abs_gap",
        "mean_abs_sigma",
        "mean_drift_5d",
        "mean_drift_20d",
        "beat_rate",
        "bmo_share",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    grouped = events.groupby("symbol", sort=False)
    summary = pd.DataFrame(
        {
            "events": grouped.size(),
            "first_event": grouped["earnings_date"].min(),
            "last_event": grouped["earnings_date"].max(),
            "mean_abs_move": grouped["abs_reaction"].mean(),
            "median_abs_move": grouped["abs_reaction"].median(),
            "max_abs_move": grouped["abs_reaction"].max(),
            "mean_move": grouped["reaction_return"].mean(),
            "up_rate": grouped["reaction_return"].agg(lambda values: float((values > 0).mean())),
            "mean_abs_gap": grouped["gap_return"].agg(lambda values: float(values.abs().mean())),
            "mean_abs_sigma": grouped["move_sigma"].agg(lambda values: float(values.abs().mean())),
            "mean_drift_5d": grouped["drift_5d"].mean(),
            "mean_drift_20d": grouped["drift_20d"].mean(),
            "beat_rate": grouped["beat"].agg(_beat_rate),
            "bmo_share": grouped["timing"].agg(lambda values: float((values == "BMO").mean())),
        }
    ).reset_index()
    summary = summary[summary["events"] >= min_events]
    return summary.sort_values("median_abs_move", ascending=False).reset_index(drop=True)[columns]
