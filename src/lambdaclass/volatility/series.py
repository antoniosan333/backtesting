"""Daily realized and implied volatility series."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from lambdaclass.earnings.calendar import context_fields, normalize_timing, parse_date
from lambdaclass.volatility.implied import (
    atm_iv_term_structure,
    constant_maturity_iv,
    event_implied_vol,
    front_iv,
)
from lambdaclass.volatility.realized import close_to_close, realized_forward, yang_zhang

DEFAULT_WINDOWS = (10, 20, 30, 60, 252)
DEFAULT_TARGETS = (7, 30, 60, 90)


def build_vol_series(
    bars: pd.DataFrame,
    chain: pd.DataFrame | None = None,
    earnings: pd.DataFrame | None = None,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    targets: Sequence[int] = DEFAULT_TARGETS,
    lookback_days: int = 252,
) -> pd.DataFrame:
    """One row per bar with realized vol, constant-maturity IV, and earnings context."""
    if bars is None or bars.empty:
        return pd.DataFrame()
    frame = bars.copy()
    frame["date"] = frame["date"].astype(str).str[:10]
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    windows = tuple(dict.fromkeys([*windows, 20]))
    close = pd.to_numeric(frame["close"], errors="coerce")
    for window in windows:
        frame[f"hv{window}"] = close_to_close(close, int(window)).to_numpy()
    if {"open", "high", "low", "close"}.issubset(frame.columns):
        frame["hv_yz20"] = yang_zhang(frame["open"], frame["high"], frame["low"], close, 20).to_numpy()
        frame["hv_yz60"] = yang_zhang(frame["open"], frame["high"], frame["low"], close, 60).to_numpy()
    else:
        frame["hv_yz20"] = np.nan
        frame["hv_yz60"] = np.nan
    earnings_df = earnings if earnings is not None else pd.DataFrame()
    excluded = _reaction_dates(frame["date"], earnings_df)
    frame["hv20_ex_earnings"] = close_to_close(
        close, 20, dates=frame["date"], exclude_dates=excluded
    ).to_numpy()
    frame["rv_fwd21"] = realized_forward(close, 21).to_numpy()
    implied = _implied_rows(frame, chain if chain is not None else pd.DataFrame(), earnings_df, targets)
    frame = frame.merge(implied, on="date", how="left")
    iv30 = pd.to_numeric(frame["iv30"], errors="coerce")
    frame["iv_rank_252"] = _iv_rank(iv30, lookback_days).to_numpy()
    frame["iv_pctile_252"] = _iv_percentile(iv30, lookback_days).to_numpy()
    hv20 = pd.to_numeric(frame["hv20"], errors="coerce")
    frame["iv_hv_spread"] = (iv30 - hv20).to_numpy()
    frame["iv_hv_ratio"] = (iv30 / hv20.replace(0.0, np.nan)).to_numpy()
    frame["iv30_minus_rv_fwd21"] = (iv30 - pd.to_numeric(frame["rv_fwd21"], errors="coerce")).to_numpy()
    context = [_context_columns(day, earnings_df) for day in frame["date"]]
    context_frame = pd.DataFrame(context)
    for column in context_frame.columns:
        frame[column] = context_frame[column]
    keep = [
        "date",
        "close",
        *[f"hv{window}" for window in windows],
        "hv_yz20",
        "hv_yz60",
        "hv20_ex_earnings",
        *[f"iv{target}" for target in targets],
        "iv_front",
        "front_expiry",
        "front_dte",
        "front_straddle",
        "iv_rank_252",
        "iv_pctile_252",
        "iv_hv_spread",
        "iv_hv_ratio",
        "rv_fwd21",
        "iv30_minus_rv_fwd21",
        "days_to_next_earnings",
        "days_since_last_earnings",
        "event_vol",
        "implied_event_move",
    ]
    return frame[keep]


def merge_vol_cache(
    previous: pd.DataFrame | None,
    updated: pd.DataFrame,
    *,
    forward_horizon: int = 21,
) -> pd.DataFrame:
    """Keep the stable history and refresh the forward-looking tail plus new dates."""
    if previous is None or previous.empty:
        return updated.reset_index(drop=True)
    if updated is None or updated.empty:
        return previous.reset_index(drop=True)
    old = previous.copy()
    new = updated.copy()
    old["date"] = old["date"].astype(str).str[:10]
    new["date"] = new["date"].astype(str).str[:10]
    added = sorted(set(new["date"]) - set(old["date"]))
    ordered = list(old["date"]) + added
    tail = set(ordered[-forward_horizon:])
    head = old[~old["date"].isin(tail)]
    refreshed = new[new["date"].isin(tail | set(added))]
    return pd.concat([head, refreshed], ignore_index=True).sort_values("date").reset_index(drop=True)


def _reaction_dates(bar_dates: pd.Series, earnings: pd.DataFrame) -> list[str]:
    days = [str(value)[:10] for value in bar_dates]
    if earnings is None or earnings.empty or "earnings_date" not in earnings.columns:
        return []
    reactions: list[str] = []
    for _, row in earnings.iterrows():
        event_day = parse_date(row["earnings_date"]).isoformat()
        timing = normalize_timing(row.get("timing", "unknown"))
        if timing == "BMO":
            chosen = next((day for day in days if day >= event_day), None)
        else:
            chosen = next((day for day in days if day > event_day), None)
        if chosen is not None:
            reactions.append(chosen)
    return reactions


def _implied_rows(
    bars: pd.DataFrame,
    chain: pd.DataFrame,
    earnings: pd.DataFrame,
    targets: Sequence[int],
) -> pd.DataFrame:
    closes = dict(zip(bars["date"], pd.to_numeric(bars["close"], errors="coerce"), strict=False))
    grouped: dict[str, pd.DataFrame] = {}
    if chain is not None and not chain.empty and "asof" in chain.columns:
        dated = chain.copy()
        dated["asof"] = dated["asof"].astype(str).str[:10]
        grouped = {str(key): value for key, value in dated.groupby("asof")}
    rows: list[dict[str, object]] = []
    for day in bars["date"]:
        spot = float(closes.get(day) or 0.0)
        day_chain = grouped.get(day, pd.DataFrame())
        if "underlying_last" in day_chain.columns:
            underlying = pd.to_numeric(day_chain["underlying_last"], errors="coerce")
            underlying = underlying[(underlying > 0) & underlying.notna()]
            if not underlying.empty:
                spot = float(underlying.median())
        term = atm_iv_term_structure(day_chain, spot)
        row: dict[str, object] = {"date": day}
        for target in targets:
            row[f"iv{target}"] = constant_maturity_iv(term, target)
        iv, expiry, dte = front_iv(term)
        row["iv_front"] = iv
        row["front_expiry"] = expiry
        row["front_dte"] = dte
        front_rows = term[term["expiry"] == expiry] if expiry is not None and not term.empty else pd.DataFrame()
        row["front_straddle"] = None if front_rows.empty else front_rows.iloc[0]["straddle"]
        context = context_fields(day, earnings)
        event_move = None
        event_vol = None
        next_earnings = context["next_earnings_date"]
        if next_earnings is not None and spot > 0:
            event = event_implied_vol(term, next_earnings, spot, asof=day)
            event_vol = event.event_vol
            event_move = event.implied_event_move
        row["event_vol"] = event_vol
        row["implied_event_move"] = event_move
        rows.append(row)
    return pd.DataFrame(rows)


def _context_columns(day: str, earnings: pd.DataFrame) -> dict[str, object]:
    context = context_fields(day, earnings)
    return {
        "days_to_next_earnings": context["days_to_next_earnings"],
        "days_since_last_earnings": context["days_since_last_earnings"],
    }


def _iv_rank(iv: pd.Series, lookback: int) -> pd.Series:
    low = iv.rolling(lookback, min_periods=lookback).min()
    high = iv.rolling(lookback, min_periods=lookback).max()
    span = (high - low).replace(0.0, np.nan)
    return (iv - low) / span


def _iv_percentile(iv: pd.Series, lookback: int) -> pd.Series:
    def _share(values: np.ndarray) -> float:
        current = values[-1]
        if not np.isfinite(current):
            return np.nan
        return float(np.sum(values < current) / len(values))

    return iv.rolling(lookback, min_periods=lookback).apply(_share, raw=True)
