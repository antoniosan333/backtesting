"""Options strategy math for the Streamlit dashboard (pure, no Streamlit).

Chain-only legs: strikes, IVs, and mids come from normalized OptionsDX rows.
Pricing and Greeks come from ``lambdaclass.options.pricing`` (vollib BSM).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from lambdaclass.options.pricing import (
    black_scholes_price,
    greeks,
    intrinsic_value,
    parse_option_date,
    safe_option_mid,
    years_between,
)

Side = str  # "call" | "put"
ParamKind = Literal["int", "float", "choice"]


def _normalize_side_cell(side: object) -> str:
    s = str(side).lower().strip()
    if s.startswith("c"):
        return "call"
    if s.startswith("p"):
        return "put"
    return s


def _row_iv(row: pd.Series) -> float:
    iv = float(pd.to_numeric(row.get("implied_volatility"), errors="coerce") or 0.0)
    return max(iv, 1e-6)


@dataclass(frozen=True)
class Leg:
    contract_symbol: str
    side: Side
    strike: float
    expiry: str
    iv: float
    mid_price: float
    quantity: int

    def expiry_date(self):
        return parse_option_date(self.expiry)


def leg_from_row(row: pd.Series, *, quantity: int) -> Leg:
    return Leg(
        contract_symbol=str(row["contract_symbol"]),
        side=_normalize_side_cell(row["side"]),
        strike=float(row["strike"]),
        expiry=str(row["expiry"]).strip(),
        iv=_row_iv(row),
        mid_price=float(safe_option_mid(row)),
        quantity=int(quantity),
    )


def snap_to_chain(chain: pd.DataFrame, side: Side, expiry: str, target_strike: float) -> Leg:
    """Pick the chain row with ``side`` and ``expiry`` whose strike is nearest to ``target_strike``."""
    if chain is None or chain.empty:
        raise ValueError("chain is empty")
    want = _normalize_side_cell(side)
    exp = str(expiry).strip()
    df = chain.copy()
    df["_side_n"] = df["side"].map(_normalize_side_cell)
    df["_exp"] = df["expiry"].astype(str).str.strip()
    sub = df[(df["_side_n"] == want) & (df["_exp"] == exp)].copy()
    if sub.empty:
        raise ValueError(f"No rows for side={want!r} expiry={exp!r}")
    sub = sub.reset_index(drop=True)
    strikes = sub["strike"].astype(float).to_numpy()
    j = int(np.abs(strikes - float(target_strike)).argmin())
    row = sub.iloc[j]
    return leg_from_row(row, quantity=1)


def find_breakevens(S_grid: np.ndarray, pnl: np.ndarray) -> list[float]:
    """Linear interpolation where ``pnl`` crosses zero."""
    if len(S_grid) < 2 or len(pnl) != len(S_grid):
        return []
    out: list[float] = []
    for i in range(len(pnl) - 1):
        y0, y1 = float(pnl[i]), float(pnl[i + 1])
        if y0 == 0.0:
            out.append(float(S_grid[i]))
        if y0 == 0.0 or y1 == 0.0:
            continue
        if y0 * y1 < 0.0:
            x0, x1 = float(S_grid[i]), float(S_grid[i + 1])
            t = -y0 / (y1 - y0)
            out.append(x0 + t * (x1 - x0))
    if len(pnl) and float(pnl[-1]) == 0.0:
        out.append(float(S_grid[-1]))
    return sorted(set(round(x, 8) for x in out))


def _leg_pnl_at(
    leg: Leg,
    spot: float,
    *,
    t_eval_years: float,
    r: float,
    iv_shift: float,
    expiry_mode: bool,
) -> float:
    """PnL contribution in dollars for this leg at ``spot``."""
    sig = max(leg.iv + iv_shift, 1e-6)
    if expiry_mode:
        v = intrinsic_value(leg.side, leg.strike, spot)
    else:
        v = black_scholes_price(leg.side, spot, leg.strike, t_eval_years, r, sig)
    return float(leg.quantity) * 100.0 * (v - leg.mid_price)


def _spot_grid(legs: Sequence[Leg], spot: float, grid_pct: float, grid_n: int) -> np.ndarray:
    """Spot grid covering spot ± pct and all leg strikes with margin."""
    spot = float(spot)
    pct = float(grid_pct)
    lo = max(spot * (1.0 - pct), 1e-6)
    hi = spot * (1.0 + pct)
    strikes = [float(leg.strike) for leg in legs]
    if strikes:
        k_min = min(strikes)
        k_max = max(strikes)
        half = pct / 2.0
        lo = min(lo, max(k_min * (1.0 - half), 1e-6))
        hi = max(hi, k_max * (1.0 + half))
    if hi <= lo:
        hi = lo * 1.01
    return np.linspace(lo, hi, int(grid_n), dtype=float)


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
    """Aggregate P&L curves and risk stats.

    ``T_eval`` per leg: calendar days from ``eval_date`` to that leg's expiry, /365.
    Grid always spans spot ± ``grid_pct`` union all leg strikes with half-pct margin.
    """
    if not legs:
        raise ValueError("legs must be non-empty")
    eval_d = parse_option_date(eval_date)
    spot = float(spot)
    S_grid = _spot_grid(legs, spot, grid_pct, grid_n)

    t_years: list[float] = []
    for leg in legs:
        t_years.append(years_between(eval_d, leg.expiry_date()))

    pnl_expiry = np.zeros_like(S_grid)
    pnl_now = np.zeros_like(S_grid)
    for j, s in enumerate(S_grid):
        pe = 0.0
        pn = 0.0
        for leg, t_ev in zip(legs, t_years, strict=True):
            pe += _leg_pnl_at(leg, float(s), t_eval_years=0.0, r=r, iv_shift=iv_shift, expiry_mode=True)
            pn += _leg_pnl_at(leg, float(s), t_eval_years=t_ev, r=r, iv_shift=iv_shift, expiry_mode=False)
        pnl_expiry[j] = pe
        pnl_now[j] = pn

    # Negative = debit (paid), positive = credit (received)
    net_premium = float(sum(-leg.quantity * 100.0 * leg.mid_price for leg in legs))

    breakevens = find_breakevens(S_grid, pnl_expiry)

    d_left = float((pnl_expiry[1] - pnl_expiry[0]) / (S_grid[1] - S_grid[0])) if len(S_grid) > 1 else 0.0
    d_right = float((pnl_expiry[-1] - pnl_expiry[-2]) / (S_grid[-1] - S_grid[-2])) if len(S_grid) > 1 else 0.0
    tol = 15.0
    k_tail = min(25, max(3, len(pnl_expiry) // 10))
    span_r = float(np.max(pnl_expiry[-k_tail:]) - np.min(pnl_expiry[-k_tail:]))
    span_l = float(np.max(pnl_expiry[:k_tail]) - np.min(pnl_expiry[:k_tail]))
    flat_right = span_r < max(50.0, 0.02 * (abs(float(np.max(pnl_expiry))) + abs(float(np.min(pnl_expiry)))))
    flat_left = span_l < max(50.0, 0.02 * (abs(float(np.max(pnl_expiry))) + abs(float(np.min(pnl_expiry)))))
    unbounded_up = bool(d_right < -tol and not flat_right) or bool(d_right > tol and not flat_right)
    unbounded_down = bool(d_left > tol and not flat_left) or bool(d_left < -tol and not flat_left)

    max_profit = float(np.max(pnl_expiry))
    max_loss = float(np.min(pnl_expiry))
    if d_right < -tol and not flat_right:
        max_loss = -math.inf
    if d_right > tol and not flat_right:
        max_profit = math.inf
    if d_left > tol and not flat_left:
        max_loss = -math.inf
    if d_left < -tol and not flat_left:
        max_profit = math.inf

    net_greeks = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    for leg, t_ev in zip(legs, t_years, strict=True):
        g = greeks(leg.side, spot, leg.strike, t_ev, r, max(leg.iv + iv_shift, 1e-6))
        m = float(leg.quantity) * 100.0
        for k in net_greeks:
            net_greeks[k] += m * g[k]

    return {
        "S_grid": S_grid,
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


# --- Presets (chain-only) ---


def _snap(chain: pd.DataFrame, side: Side, expiry: str, k: float, qty: int) -> Leg:
    leg = snap_to_chain(chain, side, expiry, k)
    if qty == leg.quantity:
        return leg
    return Leg(
        contract_symbol=leg.contract_symbol,
        side=leg.side,
        strike=leg.strike,
        expiry=leg.expiry,
        iv=leg.iv,
        mid_price=leg.mid_price,
        quantity=qty,
    )


def preset_long_call(chain: pd.DataFrame, expiry: str, spot: float, *, lots: int = 1) -> list[Leg]:
    return [_snap(chain, "call", expiry, spot, lots)]


def preset_long_put(chain: pd.DataFrame, expiry: str, spot: float, *, lots: int = 1) -> list[Leg]:
    return [_snap(chain, "put", expiry, spot, lots)]


def preset_short_call(chain: pd.DataFrame, expiry: str, spot: float, *, lots: int = 1) -> list[Leg]:
    return [_snap(chain, "call", expiry, spot, -abs(int(lots)))]


def preset_short_put(chain: pd.DataFrame, expiry: str, spot: float, *, lots: int = 1) -> list[Leg]:
    return [_snap(chain, "put", expiry, spot, -abs(int(lots)))]


def preset_vertical_spread(
    chain: pd.DataFrame,
    expiry: str,
    spot: float,
    *,
    side: Side = "call",
    width: float,
    lots: int = 1,
) -> list[Leg]:
    """Debit vertical: long lower strike, short higher strike (calls); puts: long higher, short lower."""
    w = abs(float(width))
    if str(side).lower().startswith("c"):
        lo_k = spot - w * 0.5
        hi_k = spot + w * 0.5
        return [
            _snap(chain, "call", expiry, lo_k, abs(int(lots))),
            _snap(chain, "call", expiry, hi_k, -abs(int(lots))),
        ]
    lo_k = spot - w * 0.5
    hi_k = spot + w * 0.5
    return [
        _snap(chain, "put", expiry, hi_k, abs(int(lots))),
        _snap(chain, "put", expiry, lo_k, -abs(int(lots))),
    ]


def preset_ratio_spread(
    chain: pd.DataFrame,
    expiry: str,
    spot: float,
    *,
    side: Side = "call",
    width: float,
    long_qty: int = 1,
    short_qty: int = 2,
) -> list[Leg]:
    """Buy ``long_qty`` near strike, sell ``short_qty`` further OTM."""
    w = abs(float(width))
    if str(side).lower().startswith("c"):
        k_long = spot - w * 0.25
        k_short = spot + w * 0.75
        return [
            _snap(chain, "call", expiry, k_long, abs(int(long_qty))),
            _snap(chain, "call", expiry, k_short, -abs(int(short_qty))),
        ]
    k_long = spot + w * 0.25
    k_short = spot - w * 0.75
    return [
        _snap(chain, "put", expiry, k_long, abs(int(long_qty))),
        _snap(chain, "put", expiry, k_short, -abs(int(short_qty))),
    ]


def preset_straddle(chain: pd.DataFrame, expiry: str, spot: float, *, lots: int = 1) -> list[Leg]:
    q = abs(int(lots))
    return [
        _snap(chain, "call", expiry, spot, q),
        _snap(chain, "put", expiry, spot, q),
    ]


def preset_strangle(chain: pd.DataFrame, expiry: str, spot: float, *, offset: float, lots: int = 1) -> list[Leg]:
    off = abs(float(offset))
    q = abs(int(lots))
    return [
        _snap(chain, "put", expiry, spot - off, q),
        _snap(chain, "call", expiry, spot + off, q),
    ]


def preset_butterfly(chain: pd.DataFrame, expiry: str, spot: float, *, width: float, lots: int = 1) -> list[Leg]:
    """Long call K-w, short 2× call K, long call K+w (same expiry)."""
    w = abs(float(width))
    q = abs(int(lots))
    return [
        _snap(chain, "call", expiry, spot - w, q),
        _snap(chain, "call", expiry, spot, -2 * q),
        _snap(chain, "call", expiry, spot + w, q),
    ]


def preset_iron_condor(
    chain: pd.DataFrame,
    expiry: str,
    spot: float,
    *,
    width_inner: float,
    width_outer: float,
    lots: int = 1,
) -> list[Leg]:
    """Long put far OTM, short put closer, short call closer, long call far OTM."""
    wi = abs(float(width_inner))
    wo = abs(float(width_outer))
    if wo <= wi:
        wo = wi + 1.0
    q = abs(int(lots))
    return [
        _snap(chain, "put", expiry, spot - wo, q),
        _snap(chain, "put", expiry, spot - wi, -q),
        _snap(chain, "call", expiry, spot + wi, -q),
        _snap(chain, "call", expiry, spot + wo, q),
    ]


def preset_iron_butterfly(
    chain: pd.DataFrame,
    expiry: str,
    spot: float,
    *,
    width: float,
    lots: int = 1,
) -> list[Leg]:
    """Short straddle at center, long OTM call and put."""
    w = abs(float(width))
    q = abs(int(lots))
    return [
        _snap(chain, "put", expiry, spot, -q),
        _snap(chain, "call", expiry, spot, -q),
        _snap(chain, "put", expiry, spot - w, q),
        _snap(chain, "call", expiry, spot + w, q),
    ]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: ParamKind
    label: str
    default: Any
    min_value: float | int | None = None
    step: float | int | None = None
    choices: tuple[str, ...] | None = None


PresetFactory = Callable[..., list[Leg]]


@dataclass(frozen=True)
class PresetSpec:
    label: str
    factory: PresetFactory
    params: tuple[ParamSpec, ...]


PRESETS: dict[str, PresetSpec] = {
    "long_call": PresetSpec(
        "Long call",
        preset_long_call,
        (ParamSpec("lots", "int", "Lots", 1, min_value=1, step=1),),
    ),
    "long_put": PresetSpec(
        "Long put",
        preset_long_put,
        (ParamSpec("lots", "int", "Lots", 1, min_value=1, step=1),),
    ),
    "short_call": PresetSpec(
        "Short call (naked)",
        preset_short_call,
        (ParamSpec("lots", "int", "Lots", 1, min_value=1, step=1),),
    ),
    "short_put": PresetSpec(
        "Short put (naked)",
        preset_short_put,
        (ParamSpec("lots", "int", "Lots", 1, min_value=1, step=1),),
    ),
    "vertical": PresetSpec(
        "Vertical spread",
        preset_vertical_spread,
        (
            ParamSpec("side", "choice", "Side", "call", choices=("call", "put")),
            ParamSpec("width", "float", "Width ($)", 5.0, min_value=0.5, step=0.5),
            ParamSpec("lots", "int", "Lots", 1, min_value=1, step=1),
        ),
    ),
    "ratio": PresetSpec(
        "Ratio spread",
        preset_ratio_spread,
        (
            ParamSpec("side", "choice", "Side", "call", choices=("call", "put")),
            ParamSpec("width", "float", "Width ($)", 10.0, min_value=0.5, step=0.5),
            ParamSpec("long_qty", "int", "Long qty", 1, min_value=1, step=1),
            ParamSpec("short_qty", "int", "Short qty", 2, min_value=1, step=1),
        ),
    ),
    "straddle": PresetSpec(
        "Straddle",
        preset_straddle,
        (ParamSpec("lots", "int", "Lots", 1, min_value=1, step=1),),
    ),
    "strangle": PresetSpec(
        "Strangle",
        preset_strangle,
        (
            ParamSpec("offset", "float", "OTM offset ($)", 5.0, min_value=0.5, step=0.5),
            ParamSpec("lots", "int", "Lots", 1, min_value=1, step=1),
        ),
    ),
    "butterfly": PresetSpec(
        "Call butterfly",
        preset_butterfly,
        (
            ParamSpec("width", "float", "Wing width ($)", 5.0, min_value=0.5, step=0.5),
            ParamSpec("lots", "int", "Lots", 1, min_value=1, step=1),
        ),
    ),
    "iron_condor": PresetSpec(
        "Iron condor",
        preset_iron_condor,
        (
            ParamSpec("width_inner", "float", "Inner width ($)", 5.0, min_value=0.5, step=0.5),
            ParamSpec("width_outer", "float", "Outer width ($)", 12.0, min_value=1.0, step=0.5),
            ParamSpec("lots", "int", "Lots", 1, min_value=1, step=1),
        ),
    ),
    "iron_butterfly": PresetSpec(
        "Iron butterfly",
        preset_iron_butterfly,
        (
            ParamSpec("width", "float", "Wing width ($)", 5.0, min_value=0.5, step=0.5),
            ParamSpec("lots", "int", "Lots", 1, min_value=1, step=1),
        ),
    ),
}


def build_preset_legs(
    preset_key: str,
    chain: pd.DataFrame,
    expiry: str,
    spot: float,
    params: dict[str, Any],
) -> list[Leg]:
    """Dispatch preset factories with typed kwargs from ``params``."""
    if preset_key not in PRESETS:
        raise KeyError(preset_key)
    spec = PRESETS[preset_key]
    kwargs: dict[str, Any] = {}
    for p in spec.params:
        raw = params.get(p.name, p.default)
        if p.kind == "int":
            kwargs[p.name] = int(raw)
        elif p.kind == "float":
            kwargs[p.name] = float(raw)
        else:
            kwargs[p.name] = str(raw)
    return spec.factory(chain, expiry, spot, **kwargs)


def legs_to_dataframe(legs: Sequence[Leg]) -> pd.DataFrame:
    """Serialize legs for ``st.data_editor``."""
    rows = [
        {
            "contract_symbol": leg.contract_symbol,
            "side": leg.side,
            "strike": leg.strike,
            "expiry": leg.expiry,
            "iv": leg.iv,
            "mid_price": leg.mid_price,
            "quantity": leg.quantity,
            "remove": False,
        }
        for leg in legs
    ]
    return pd.DataFrame(rows)


def legs_from_dataframe(df: pd.DataFrame) -> list[Leg]:
    """Rebuild ``Leg`` list from editor output (skips rows marked remove)."""
    legs: list[Leg] = []
    for _, row in df.iterrows():
        if bool(row.get("remove", False)):
            continue
        legs.append(
            Leg(
                contract_symbol=str(row["contract_symbol"]),
                side=_normalize_side_cell(row["side"]),
                strike=float(row["strike"]),
                expiry=str(row["expiry"]).strip(),
                iv=float(row["iv"]),
                mid_price=float(row["mid_price"]),
                quantity=int(row["quantity"]),
            )
        )
    return legs
