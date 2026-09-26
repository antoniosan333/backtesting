"""Earnings price-reaction events and per-symbol statistics (pure; no network).

Each calendar row is matched to daily bars. The *reaction bar* is the first
session that trades on the news: the report day for BMO (before market open),
the next session for AMC (after market close). Historical vendor timing is often
missing, so it is inferred from which overnight gap is larger:

* BMO when ``|open[D] / close[D-1] - 1| >= |open[D+1] / close[D] - 1|``
* AMC otherwise

``timing_confidence`` is the larger gap's share of both gaps (0.5 = coin flip,
1.0 = only one gap moved). Inferred timing uses the reaction itself, so treat it
as a labelling aid for research, not something a live strategy knew in advance.
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
EVENT_COLUMNS = [
    "symbol",
    "earnings_date",
    "timing",
    "timing_source",
    "vendor_timing",
    "inferred_timing",
    "timing_confidence",
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


def _event_row(bars: BarSeries, calendar_row: Mapping[str, Any]) -> dict[str, Any] | None:
    earnings_date = str(calendar_row["earnings_date"])[:10]
    index = int(np.searchsorted(bars.dates, earnings_date, side="left"))
    if index >= len(bars):
        return None
    vendor = str(calendar_row.get("timing") or "unknown")
    vendor_timing = vendor if vendor in KNOWN_TIMINGS else None
    trading_day = bars.dates[index] == earnings_date
    inferred, confidence = infer_timing(bars, index) if trading_day else (None, float("nan"))
    if not trading_day:
        # Reported on a weekend/holiday: the next session is the first to react either way.
        timing, timing_source, reaction = vendor_timing or "BMO", "non_trading_day", index
    elif vendor_timing is not None:
        timing, timing_source = vendor_timing, "vendor"
        reaction = index if vendor_timing == "BMO" else index + 1
    elif inferred is not None:
        timing, timing_source = inferred, "inferred_gap"
        reaction = index if inferred == "BMO" else index + 1
    else:
        return None
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
        "earnings_date": earnings_date,
        "timing": timing,
        "timing_source": timing_source,
        "vendor_timing": vendor_timing,
        "inferred_timing": inferred,
        "timing_confidence": confidence,
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
            for calendar_row in group.sort_values("earnings_date").to_dict("records"):
                event = _event_row(bars, calendar_row)
                if event is not None:
                    rows.append(event)
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    events["beat"] = events["beat"].astype("boolean")
    return events


def apply_event_timing(earnings: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Fill ``unknown`` timing in a per-symbol earnings frame from built events."""
    if earnings.empty or events.empty:
        return earnings
    resolved = events[events["timing_source"] != "vendor"].set_index(["symbol", "earnings_date"])
    if resolved.empty:
        return earnings
    frame = earnings.copy()
    if "timing_source" not in frame.columns:
        frame["timing_source"] = "none"
    keys = pd.MultiIndex.from_arrays(
        [frame["symbol"].astype(str), frame["earnings_date"].astype(str).str[:10]]
    )
    timing = pd.Series(resolved["timing"].reindex(keys).to_numpy(), index=frame.index)
    source = pd.Series(resolved["timing_source"].reindex(keys).to_numpy(), index=frame.index)
    fill = frame["timing"].eq("unknown") & timing.notna()
    frame.loc[fill, "timing"] = timing[fill]
    frame.loc[fill, "timing_source"] = source[fill]
    return frame


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
            "beat_rate": grouped["beat"].agg(lambda values: float(values.dropna().astype(float).mean())),
            "bmo_share": grouped["timing"].agg(lambda values: float((values == "BMO").mean())),
        }
    ).reset_index()
    summary = summary[summary["events"] >= min_events]
    return summary.sort_values("median_abs_move", ascending=False).reset_index(drop=True)[columns]
