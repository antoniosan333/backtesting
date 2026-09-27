"""Implied-volatility term structure and constant-maturity series."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from lambdaclass.options.expected_move import expected_moves
from lambdaclass.options.pricing import parse_option_date


@dataclass(frozen=True)
class EventImpliedVol:
    event_vol: float | None
    implied_event_move: float | None
    reason: str | None


def atm_iv_term_structure(chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    """One row per expiry with interpolated at-the-money implied volatility."""
    columns = ["expiry", "dte", "atm_iv", "straddle", "n_quotes", "quality"]
    if chain is None or chain.empty or "expiry" not in chain.columns:
        return pd.DataFrame(columns=columns)
    moves = expected_moves(chain, spot)
    if moves.empty:
        return pd.DataFrame(columns=columns)
    counts = chain.assign(_expiry=chain["expiry"].astype(str).str[:10]).groupby("_expiry").size()
    out = pd.DataFrame(
        {
            "expiry": moves["expiry"].astype(str).str[:10],
            "dte": pd.to_numeric(moves["dte"], errors="coerce"),
            "atm_iv": pd.to_numeric(moves["atm_iv"], errors="coerce"),
            "straddle": pd.to_numeric(moves["straddle"], errors="coerce"),
            "n_quotes": moves["expiry"].astype(str).str[:10].map(counts).fillna(0).astype(int),
            "quality": moves["quality"].astype(str),
        }
    )
    out.loc[out["atm_iv"] <= 0, "atm_iv"] = pd.NA
    return out.sort_values("dte").reset_index(drop=True)


def constant_maturity_iv(term: pd.DataFrame, target_dte: float) -> float | None:
    """Interpolate total variance between the expiries around ``target_dte``."""
    if term is None or term.empty:
        return None
    points = term.dropna(subset=["dte", "atm_iv"]).copy()
    points = points[(points["dte"] > 0) & (points["atm_iv"] > 0)].sort_values("dte")
    if points.empty:
        return None
    target = float(target_dte)
    below = points[points["dte"] <= target]
    above = points[points["dte"] >= target]
    if below.empty or above.empty:
        return None
    left = below.iloc[-1]
    right = above.iloc[0]
    if float(right["dte"]) == float(left["dte"]):
        return float(left["atm_iv"])
    left_t = float(left["dte"]) / 365.0
    right_t = float(right["dte"]) / 365.0
    target_t = target / 365.0
    left_var = float(left["atm_iv"]) ** 2 * left_t
    right_var = float(right["atm_iv"]) ** 2 * right_t
    weight = (target_t - left_t) / (right_t - left_t)
    variance = left_var + weight * (right_var - left_var)
    if variance <= 0 or target_t <= 0:
        return None
    return math.sqrt(variance / target_t)


def front_iv(term: pd.DataFrame) -> tuple[float | None, str | None, float | None]:
    """ATM IV of the nearest expiry with at least one day left."""
    if term is None or term.empty:
        return None, None, None
    points = term.dropna(subset=["dte", "atm_iv"])
    points = points[(points["dte"] >= 1) & (points["atm_iv"] > 0)].sort_values("dte")
    if points.empty:
        return None, None, None
    row = points.iloc[0]
    return float(row["atm_iv"]), str(row["expiry"]), float(row["dte"])


def event_implied_vol(
    term: pd.DataFrame,
    earnings_date: str,
    spot: float,
    *,
    asof: str | None = None,
) -> EventImpliedVol:
    """Separate the earnings jump from the variance between two expiries."""
    if term is None or term.empty:
        return EventImpliedVol(None, None, "MISSING_EXPIRY")
    event_day = parse_option_date(earnings_date)
    if asof is not None and parse_option_date(asof) > event_day:
        return EventImpliedVol(None, None, "PAST_EVENT")
    points = term.dropna(subset=["expiry", "dte", "atm_iv"]).copy()
    points = points[(points["dte"] > 0) & (points["atm_iv"] > 0)]
    if points.empty:
        return EventImpliedVol(None, None, "MISSING_EXPIRY")
    points["expiry_date"] = points["expiry"].map(lambda value: parse_option_date(str(value)))
    after = points[points["expiry_date"] >= event_day].sort_values("dte")
    if len(after) < 2:
        return EventImpliedVol(None, None, "MISSING_EXPIRY")
    near = after.iloc[0]
    far = after.iloc[1]
    near_t = float(near["dte"]) / 365.0
    far_t = float(far["dte"]) / 365.0
    if far_t <= near_t:
        return EventImpliedVol(None, None, "MISSING_EXPIRY")
    near_var = float(near["atm_iv"]) ** 2 * near_t
    far_var = float(far["atm_iv"]) ** 2 * far_t
    ambient_per_year = (far_var - near_var) / (far_t - near_t)
    event_var = near_var - ambient_per_year * near_t
    if event_var <= 1e-8:
        return EventImpliedVol(None, None, "NOT_INVERTED")
    event_vol = math.sqrt(event_var)
    return EventImpliedVol(event_vol, float(spot) * event_vol, None)
