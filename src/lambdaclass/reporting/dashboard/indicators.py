"""Pure technical indicators and equity-curve transforms used by the dashboard.

Functions take pandas inputs, return pandas outputs, and never read from disk.
Formulas are spelled out in docstrings so the dashboard does not silently
diverge from strategy code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over ``window`` rows. Right-aligned, no future leak."""
    if window <= 0:
        raise ValueError("window must be positive")
    return series.astype(float).rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int, adjust: bool = False) -> pd.Series:
    """Exponential moving average. ``adjust=False`` matches recursive EMA used in TA libraries."""
    if window <= 0:
        raise ValueError("window must be positive")
    return series.astype(float).ewm(span=window, adjust=adjust, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]. NaN until ``window`` deltas accumulate."""
    if window <= 0:
        raise ValueError("window must be positive")
    closes = series.astype(float)
    delta = closes.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(100.0).where(avg_loss != 0, 100.0)


@dataclass
class BollingerBands:
    middle: pd.Series
    upper: pd.Series
    lower: pd.Series


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> BollingerBands:
    closes = series.astype(float)
    mid = sma(closes, window)
    std = closes.rolling(window=window, min_periods=window).std(ddof=0)
    return BollingerBands(middle=mid, upper=mid + num_std * std, lower=mid - num_std * std)


def equity_drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown as ``(equity - running_max) / running_max``, in [-1, 0]."""
    eq = equity.astype(float)
    running_max = eq.cummax()
    return ((eq - running_max) / running_max.replace(0, float("nan"))).fillna(0.0)


def rolling_sharpe(
    equity: pd.Series, window: int = 21, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Annualized rolling Sharpe over ``window`` periods of pct returns."""
    if window <= 1:
        raise ValueError("window must be > 1")
    returns = equity.astype(float).pct_change().fillna(0.0)
    mean = returns.rolling(window=window, min_periods=window).mean()
    std = returns.rolling(window=window, min_periods=window).std(ddof=0)
    return (mean / std.replace(0.0, float("nan"))) * math.sqrt(periods_per_year)


def monthly_returns_table(
    equity: pd.DataFrame, date_col: str = "date", equity_col: str = "equity"
) -> pd.DataFrame:
    """Year x Month pivot of monthly compounded returns.

    Resamples ``equity`` to month-end last value, then computes ``last/first - 1``
    per calendar month. Returns a DataFrame indexed by year with month columns 1..12.
    Empty input yields an empty DataFrame with no columns.
    """
    if equity is None or equity.empty or equity_col not in equity.columns:
        return pd.DataFrame()
    df = equity[[date_col, equity_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    if df.empty:
        return pd.DataFrame()
    monthly_first = df[equity_col].resample("MS").first()
    monthly_last = df[equity_col].resample("ME").last()
    monthly_first.index = pd.DatetimeIndex(monthly_first.index).to_period("M")
    monthly_last.index = pd.DatetimeIndex(monthly_last.index).to_period("M")
    common = monthly_first.index.intersection(monthly_last.index)
    if len(common) == 0:
        return pd.DataFrame()
    pct = (monthly_last.loc[common] / monthly_first.loc[common]) - 1.0
    period_index = pd.PeriodIndex(common, freq="M")
    table = pd.DataFrame(
        {
            "year": period_index.year,
            "month": period_index.month,
            "ret": pct.to_numpy(),
        }
    )
    pivot = table.pivot(index="year", columns="month", values="ret")
    return pivot.reindex(columns=range(1, 13))


def derived_run_stats(
    metrics: dict[str, float],
    trades: pd.DataFrame,
    num_trades_value: int,
    holding_value: float,
    win_rate_value: float,
) -> dict[str, float]:
    """Combine engine-emitted metrics with dashboard-derived stats into a single dict."""
    out: dict[str, float] = {k: float(v) for k, v in metrics.items()}
    out["num_trades"] = float(num_trades_value)
    out["avg_holding_days"] = float(holding_value)
    out["win_rate_per_trade"] = float(win_rate_value)
    return out
