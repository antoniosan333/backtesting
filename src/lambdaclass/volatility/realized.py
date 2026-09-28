"""Annualized realized-volatility estimators.

Windows are trailing and right-aligned. Values stay missing until the window is full.
Variances are daily; the returned series is multiplied by ``sqrt(252)``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _as_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _dates(dates: pd.Series | None, length: int) -> np.ndarray:
    if dates is None:
        return np.array([""] * length, dtype=object)
    return pd.Series(dates).astype(str).str[:10].to_numpy()


def _excluded(exclude_dates: Iterable[str] | None) -> set[str]:
    if not exclude_dates:
        return set()
    return {str(value)[:10] for value in exclude_dates}


def close_to_close(
    close: pd.Series,
    window: int,
    *,
    dates: pd.Series | None = None,
    exclude_dates: Iterable[str] | None = None,
) -> pd.Series:
    """Annualized close-to-close volatility over ``window`` log returns."""
    if window < 2:
        raise ValueError("window must be at least 2")
    prices = _as_float(close)
    returns = np.full(len(prices), np.nan)
    valid_price = np.isfinite(prices) & (prices > 0)
    returns[1:] = np.where(
        valid_price[1:] & valid_price[:-1],
        np.log(prices[1:] / prices[:-1]),
        np.nan,
    )
    excluded = _excluded(exclude_dates)
    if excluded:
        day = _dates(dates, len(prices))
        returns = np.where(np.isin(day, list(excluded)), np.nan, returns)
    return _rolling_std(returns, window, index=close.index, allow_holes=bool(excluded))


def realized_forward(close: pd.Series, horizon: int) -> pd.Series:
    """Annualized close-to-close volatility over the next ``horizon`` sessions."""
    if horizon < 2:
        raise ValueError("horizon must be at least 2")
    trailing = close_to_close(close, horizon)
    return trailing.shift(-horizon)


def parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Annualized Parkinson volatility from the high-low range."""
    if window < 2:
        raise ValueError("window must be at least 2")
    hi = _as_float(high)
    lo = _as_float(low)
    usable = (hi > 0) & (lo > 0) & (hi >= lo)
    variance = np.where(usable, np.log(hi / lo) ** 2 / (4.0 * math.log(2.0)), np.nan)
    return _rolling_mean(variance, window, index=high.index)


def garman_klass(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    """Annualized Garman-Klass volatility from the open, high, low, and close."""
    if window < 2:
        raise ValueError("window must be at least 2")
    op = _as_float(open_)
    hi = _as_float(high)
    lo = _as_float(low)
    cl = _as_float(close)
    usable = (op > 0) & (hi > 0) & (lo > 0) & (cl > 0) & (hi >= lo)
    log_hl = np.log(hi / lo)
    log_co = np.log(cl / op)
    variance = 0.5 * log_hl**2 - (2.0 * math.log(2.0) - 1.0) * log_co**2
    variance = np.where(usable, variance, np.nan)
    return _rolling_mean(variance, window, index=close.index)


def yang_zhang(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    """Annualized Yang-Zhang volatility, including the overnight gap."""
    if window < 2:
        raise ValueError("window must be at least 2")
    op = _as_float(open_)
    hi = _as_float(high)
    lo = _as_float(low)
    cl = _as_float(close)
    overnight = np.full(len(cl), np.nan)
    usable_gap = (op > 0) & (cl > 0)
    overnight[1:] = np.where(usable_gap[1:] & (cl[:-1] > 0), np.log(op[1:] / cl[:-1]), np.nan)
    open_close = np.where(usable_gap, np.log(cl / op), np.nan)
    range_ok = usable_gap & (hi > 0) & (lo > 0) & (hi >= lo)
    rogers = np.log(hi / cl) * np.log(hi / op) + np.log(lo / cl) * np.log(lo / op)
    rogers = np.where(range_ok, rogers, np.nan)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    out = np.full(len(cl), np.nan)
    for end in range(window, len(cl) + 1):
        start = end - window
        overnight_w = overnight[start:end]
        open_w = open_close[start:end]
        range_w = rogers[start:end]
        if not (np.isfinite(overnight_w).all() and np.isfinite(open_w).all() and np.isfinite(range_w).all()):
            continue
        variance = (
            np.var(overnight_w, ddof=1)
            + k * np.var(open_w, ddof=1)
            + (1.0 - k) * np.mean(range_w)
        )
        out[end - 1] = math.sqrt(max(variance, 0.0) * TRADING_DAYS_PER_YEAR)
    return pd.Series(out, index=close.index, dtype=float)


def _rolling_std(values: np.ndarray, window: int, *, index: pd.Index, allow_holes: bool) -> pd.Series:
    out = np.full(len(values), np.nan)
    for end in range(window, len(values) + 1):
        chunk = values[end - window : end]
        valid = chunk[np.isfinite(chunk)]
        if len(valid) < 2:
            continue
        if not allow_holes and len(valid) < window:
            continue
        out[end - 1] = float(np.std(valid, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
    return pd.Series(out, index=index, dtype=float)


def _rolling_mean(variance: np.ndarray, window: int, *, index: pd.Index) -> pd.Series:
    out = np.full(len(variance), np.nan)
    for end in range(window, len(variance) + 1):
        chunk = variance[end - window : end]
        if not np.isfinite(chunk).all():
            continue
        out[end - 1] = math.sqrt(max(float(np.mean(chunk)), 0.0) * TRADING_DAYS_PER_YEAR)
    return pd.Series(out, index=index, dtype=float)
