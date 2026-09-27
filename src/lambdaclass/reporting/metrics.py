from __future__ import annotations

import math

import pandas as pd

_EMPTY_METRICS = {
    "cagr": 0.0,
    "max_drawdown": 0.0,
    "sharpe": 0.0,
    "hit_rate": 0.0,
    "up_bar_ratio": 0.0,
    "total_return": 0.0,
    "annualized_volatility": 0.0,
    "sortino": 0.0,
    "calmar": 0.0,
}


def _infer_periods_per_year(equity_curve: pd.DataFrame) -> int:
    if "date" in equity_curve:
        dates = pd.to_datetime(equity_curve["date"], errors="coerce")
    elif isinstance(equity_curve.index, pd.DatetimeIndex):
        dates = pd.Series(equity_curve.index)
    else:
        return 252
    spacing = dates.dropna().sort_values().diff().dt.total_seconds().dropna() / 86_400
    positive_spacing = spacing[spacing > 0]
    if positive_spacing.empty:
        return 252
    median_days = float(positive_spacing.median())
    if median_days <= 3:
        return 252
    if median_days <= 10:
        return 52
    return 12


def compute_metrics(
    equity_curve: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int | None = None,
) -> dict[str, float]:
    if equity_curve.empty or "equity" not in equity_curve:
        return dict(_EMPTY_METRICS)
    if periods_per_year is not None and periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    annual_periods = periods_per_year or _infer_periods_per_year(equity_curve)
    equity = (
        pd.to_numeric(equity_curve["equity"], errors="coerce").replace([math.inf, -math.inf], pd.NA).dropna()
    )
    if equity.empty:
        return dict(_EMPTY_METRICS)

    previous = equity.shift(1)
    returns = (equity / previous - 1.0).where(previous != 0).replace([math.inf, -math.inf], pd.NA)
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    first_equity = float(equity.iloc[0])
    last_equity = float(equity.iloc[-1])
    total_return = last_equity / first_equity - 1.0 if first_equity != 0 else 0.0
    intervals = len(equity) - 1
    cagr = (
        float((last_equity / first_equity) ** (annual_periods / intervals) - 1.0)
        if intervals > 0 and first_equity > 0 and last_equity > 0
        else 0.0
    )

    volatility = float(returns.std(ddof=0)) if not returns.empty else 0.0
    annualized_volatility = volatility * math.sqrt(annual_periods)
    periodic_risk_free = (
        (1.0 + risk_free_rate) ** (1.0 / annual_periods) - 1.0 if risk_free_rate > -1.0 else -1.0
    )
    excess_returns = returns - periodic_risk_free
    mean_excess = float(excess_returns.mean()) if not excess_returns.empty else 0.0
    sharpe = mean_excess / volatility * math.sqrt(annual_periods) if volatility > 0 else 0.0
    downside = excess_returns.clip(upper=0.0)
    downside_deviation = float(math.sqrt(float((downside**2).mean()))) if not downside.empty else 0.0
    sortino = mean_excess / downside_deviation * math.sqrt(annual_periods) if downside_deviation > 0 else 0.0

    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max.where(rolling_max > 0)
    max_drawdown = float(drawdowns.fillna(0.0).min())
    up_bar_ratio = float((returns > 0).mean()) if not returns.empty else 0.0
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0
    return {
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "hit_rate": up_bar_ratio,
        "up_bar_ratio": up_bar_ratio,
        "total_return": total_return,
        "annualized_volatility": annualized_volatility,
        "sortino": sortino,
        "calmar": calmar,
    }


def num_trades(trades: pd.DataFrame) -> int:
    return 0 if trades is None or trades.empty else int(len(trades))


def avg_holding_days(trades: pd.DataFrame) -> float:
    """Average days between paired buy/sell rows in chronological order."""
    if trades is None or trades.empty:
        return 0.0
    if "date" not in trades.columns or "action" not in trades.columns:
        return 0.0
    frame = trades.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    holding: list[float] = []
    last_buy: pd.Timestamp | None = None
    for action, when in zip(frame["action"].astype(str).str.lower(), frame["date"], strict=True):
        if action == "buy" and last_buy is None:
            last_buy = when
        elif action == "sell" and last_buy is not None:
            holding.append((when - last_buy).days)
            last_buy = None
    return float(sum(holding) / len(holding)) if holding else 0.0


def win_rate_per_trade(trades: pd.DataFrame) -> float:
    """Fraction of buy/sell pairs whose sell price exceeds the buy price."""
    if trades is None or trades.empty:
        return 0.0
    if "action" not in trades.columns or "price" not in trades.columns:
        return 0.0
    wins = 0
    pairs = 0
    last_buy: float | None = None
    actions = trades["action"].astype(str).str.lower()
    prices = pd.to_numeric(trades["price"], errors="coerce").fillna(0.0).astype(float)
    for action, price in zip(actions, prices, strict=True):
        if action == "buy" and last_buy is None:
            last_buy = price
        elif action == "sell" and last_buy is not None:
            pairs += 1
            if price > last_buy:
                wins += 1
            last_buy = None
    return float(wins / pairs) if pairs else 0.0
