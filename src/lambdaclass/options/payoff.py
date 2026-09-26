"""Option-position payoff, breakeven, and risk calculations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np

from lambdaclass.options.presets import Leg
from lambdaclass.options.pricing import (
    greeks,
    option_flag,
    parse_option_date,
    years_between,
)


def find_breakevens(spot_grid: np.ndarray, pnl: np.ndarray) -> list[float]:
    """Find zero crossings using linear interpolation."""
    if len(spot_grid) < 2 or len(pnl) != len(spot_grid):
        return []
    breakevens: list[float] = []
    for index in range(len(pnl) - 1):
        pnl_start = float(pnl[index])
        pnl_end = float(pnl[index + 1])
        if pnl_start == 0.0:
            breakevens.append(float(spot_grid[index]))
        if pnl_start == 0.0 or pnl_end == 0.0:
            continue
        if pnl_start * pnl_end < 0.0:
            spot_start = float(spot_grid[index])
            spot_end = float(spot_grid[index + 1])
            fraction = -pnl_start / (pnl_end - pnl_start)
            breakevens.append(spot_start + fraction * (spot_end - spot_start))
    if len(pnl) and float(pnl[-1]) == 0.0:
        breakevens.append(float(spot_grid[-1]))
    return sorted(set(round(value, 8) for value in breakevens))


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    """Vectorized normal CDF (Abramowitz-Stegun 7.1.26)."""
    absolute = np.abs(values)
    scale = 1.0 / (1.0 + 0.2316419 * absolute)
    polynomial = (
        (((1.330274429 * scale - 1.821255978) * scale + 1.781477937) * scale - 0.356563782) * scale
        + 0.319381530
    ) * scale
    density = np.exp(-0.5 * absolute * absolute) / math.sqrt(2.0 * math.pi)
    positive = 1.0 - density * polynomial
    return np.where(values >= 0.0, positive, 1.0 - positive)


def _option_values(
    side: str,
    spots: np.ndarray,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
) -> np.ndarray:
    strike = float(strike)
    if time_to_expiry <= 0.0:
        if option_flag(side) == "c":
            return np.maximum(spots - strike, 0.0)
        return np.maximum(strike - spots, 0.0)
    sigma = max(float(volatility), 1e-12)
    root_time = math.sqrt(time_to_expiry)
    d1 = (np.log(spots / strike) + (float(rate) + 0.5 * sigma * sigma) * time_to_expiry) / (sigma * root_time)
    d2 = d1 - sigma * root_time
    discounted_strike = strike * math.exp(-float(rate) * time_to_expiry)
    if option_flag(side) == "c":
        return spots * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    return discounted_strike * _normal_cdf(-d2) - spots * _normal_cdf(-d1)


def _spot_grid(
    legs: Sequence[Leg],
    spot: float,
    grid_pct: float,
    grid_n: int,
) -> np.ndarray:
    spot = float(spot)
    grid_pct = float(grid_pct)
    lower = max(spot * (1.0 - grid_pct), 1e-6)
    upper = spot * (1.0 + grid_pct)
    strikes = [float(leg.strike) for leg in legs]
    if strikes:
        half_margin = grid_pct / 2.0
        lower = min(lower, max(min(strikes) * (1.0 - half_margin), 1e-6))
        upper = max(upper, max(strikes) * (1.0 + half_margin))
    if upper <= lower:
        upper = lower * 1.01
    return np.linspace(lower, upper, int(grid_n), dtype=float)


def position_pnl(
    legs: Sequence[Leg],
    *,
    eval_date: str | date,
    r: float,
    iv_shift: float = 0.0,
    spot: float,
    grid_pct: float = 0.30,
    grid_n: int = 401,
) -> dict[str, Any]:
    """Aggregate P&L curves, breakevens, tail risk, and Greeks."""
    if not legs:
        raise ValueError("legs must be non-empty")
    evaluation_date = parse_option_date(eval_date)
    spot = float(spot)
    spot_grid = _spot_grid(legs, spot, grid_pct, grid_n)
    times_to_expiry = [years_between(evaluation_date, leg.expiry_date()) for leg in legs]

    pnl_expiry = np.zeros_like(spot_grid)
    pnl_now = np.zeros_like(spot_grid)
    for leg, time_to_expiry in zip(legs, times_to_expiry, strict=True):
        multiplier = float(leg.quantity) * 100.0
        expiry_values = _option_values(
            leg.side,
            spot_grid,
            leg.strike,
            0.0,
            r,
            leg.iv + iv_shift,
        )
        mark_values = _option_values(
            leg.side,
            spot_grid,
            leg.strike,
            time_to_expiry,
            r,
            leg.iv + iv_shift,
        )
        pnl_expiry += multiplier * (expiry_values - leg.mid_price)
        pnl_now += multiplier * (mark_values - leg.mid_price)

    net_premium = float(sum(-leg.quantity * 100.0 * leg.mid_price for leg in legs))
    breakevens = find_breakevens(spot_grid, pnl_expiry)

    if len(spot_grid) > 1:
        left_slope = float((pnl_expiry[1] - pnl_expiry[0]) / (spot_grid[1] - spot_grid[0]))
        right_slope = float((pnl_expiry[-1] - pnl_expiry[-2]) / (spot_grid[-1] - spot_grid[-2]))
    else:
        left_slope = right_slope = 0.0
    tolerance = 15.0
    tail_length = min(25, max(3, len(pnl_expiry) // 10))
    pnl_scale = abs(float(np.max(pnl_expiry))) + abs(float(np.min(pnl_expiry)))
    flat_threshold = max(50.0, 0.02 * pnl_scale)
    flat_right = float(np.max(pnl_expiry[-tail_length:]) - np.min(pnl_expiry[-tail_length:])) < flat_threshold
    flat_left = float(np.max(pnl_expiry[:tail_length]) - np.min(pnl_expiry[:tail_length])) < flat_threshold
    unbounded_up = abs(right_slope) > tolerance and not flat_right
    unbounded_down = abs(left_slope) > tolerance and not flat_left

    max_profit = float(np.max(pnl_expiry))
    max_loss = float(np.min(pnl_expiry))
    if right_slope < -tolerance and not flat_right:
        max_loss = -math.inf
    if right_slope > tolerance and not flat_right:
        max_profit = math.inf
    if left_slope > tolerance and not flat_left:
        max_loss = -math.inf
    if left_slope < -tolerance and not flat_left:
        max_profit = math.inf

    net_greeks = {
        "delta": 0.0,
        "gamma": 0.0,
        "theta": 0.0,
        "vega": 0.0,
        "rho": 0.0,
    }
    for leg, time_to_expiry in zip(legs, times_to_expiry, strict=True):
        leg_greeks = greeks(
            leg.side,
            spot,
            leg.strike,
            time_to_expiry,
            r,
            max(leg.iv + iv_shift, 1e-6),
        )
        multiplier = float(leg.quantity) * 100.0
        for greek_name in net_greeks:
            net_greeks[greek_name] += multiplier * leg_greeks[greek_name]

    return {
        "S_grid": spot_grid,
        "pnl_expiry": pnl_expiry,
        "pnl_now": pnl_now,
        "net_premium": net_premium,
        "breakevens": breakevens,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "unbounded_up": unbounded_up,
        "unbounded_down": unbounded_down,
        "net_greeks": net_greeks,
    }


__all__ = ["find_breakevens", "position_pnl"]
