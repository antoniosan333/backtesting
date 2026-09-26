"""Black–Scholes pricing and quote helpers (pure; no Streamlit).

Uses ``vollib`` analytical BSM. Convention notes:
- ``theta`` is per calendar day (vollib already divides by 365).
- ``vega`` / ``rho`` are per 1-percentage-point move (÷100).
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from vollib.black_scholes import black_scholes
from vollib.black_scholes.greeks.analytical import delta, gamma, rho, theta, vega

Side = str  # "call" | "put"


def option_flag(side: Side) -> str:
    s = str(side).lower().strip()
    if s.startswith("c"):
        return "c"
    if s.startswith("p"):
        return "p"
    raise ValueError(f"side must be call or put, got {side!r}")


def parse_option_date(d: str | date | datetime) -> date:
    """Parse ``YYYY-MM-DD`` or pass through ``date``/``datetime``."""
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    s = str(d).strip()[:10]
    return date.fromisoformat(s)


def years_between(from_date: date, to_date: date) -> float:
    if to_date <= from_date:
        return 0.0
    return max((to_date - from_date).days, 0) / 365.0


def intrinsic_value(side: Side, strike: float, spot: float) -> float:
    """Intrinsic value per share (not ×100)."""
    k = float(strike)
    s = float(spot)
    if option_flag(side) == "c":
        return max(s - k, 0.0)
    return max(k - s, 0.0)


def black_scholes_price(side: Side, spot: float, strike: float, t: float, r: float, sigma: float) -> float:
    """BSM price per share. ``t`` in years."""
    if t <= 0.0:
        return intrinsic_value(side, strike, spot)
    sig = max(float(sigma), 1e-12)
    return float(black_scholes(option_flag(side), float(spot), float(strike), float(t), float(r), sig))


def greeks(side: Side, spot: float, strike: float, t: float, r: float, sigma: float) -> dict[str, float]:
    """Per-share Greeks (vollib analytical).

    ``theta`` is calendar-day decay (already ÷365). ``vega`` / ``rho`` are per
    1 percentage-point move in vol / rate (already ÷100).
    """
    if t <= 0.0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    sig = max(float(sigma), 1e-12)
    fl = option_flag(side)
    s = float(spot)
    k = float(strike)
    return {
        "delta": float(delta(fl, s, k, float(t), float(r), sig)),
        "gamma": float(gamma(fl, s, k, float(t), float(r), sig)),
        "theta": float(theta(fl, s, k, float(t), float(r), sig)),
        "vega": float(vega(fl, s, k, float(t), float(r), sig)),
        "rho": float(rho(fl, s, k, float(t), float(r), sig)),
    }


def safe_option_mid(row: pd.Series) -> float:
    bid = float(pd.to_numeric(row.get("bid"), errors="coerce") or 0.0)
    ask = float(pd.to_numeric(row.get("ask"), errors="coerce") or 0.0)
    last = float(pd.to_numeric(row.get("last_price"), errors="coerce") or 0.0)
    if bid > 0.0 and ask > 0.0 and bid <= ask:
        return (bid + ask) / 2.0
    return last
