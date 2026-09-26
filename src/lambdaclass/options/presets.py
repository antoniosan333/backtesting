"""Option legs and chain-backed strategy presets."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from lambdaclass.options.pricing import numeric_value, parse_option_date, safe_option_mid

Side = str  # "call" | "put"
ParamKind = Literal["int", "float", "choice"]


def _normalize_side_cell(side: object) -> str:
    normalized = str(side).lower().strip()
    if normalized.startswith("c"):
        return "call"
    if normalized.startswith("p"):
        return "put"
    return normalized


def _row_iv(row: pd.Series) -> float:
    iv = numeric_value(row.get("implied_volatility"))
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

    def expiry_date(self) -> date:
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
    """Pick the row for ``side`` and ``expiry`` nearest to ``target_strike``."""
    if chain is None or chain.empty:
        raise ValueError("chain is empty")
    wanted_side = _normalize_side_cell(side)
    wanted_expiry = str(expiry).strip()
    candidates = chain.copy()
    candidates["_side_n"] = candidates["side"].map(_normalize_side_cell)
    candidates["_exp"] = candidates["expiry"].astype(str).str.strip()
    candidates = candidates[
        (candidates["_side_n"] == wanted_side) & (candidates["_exp"] == wanted_expiry)
    ].reset_index(drop=True)
    if candidates.empty:
        raise ValueError(f"No rows for side={wanted_side!r} expiry={wanted_expiry!r}")
    strikes = candidates["strike"].astype(float).to_numpy()
    row = candidates.iloc[int(np.abs(strikes - float(target_strike)).argmin())]
    return leg_from_row(row, quantity=1)


def _snap(chain: pd.DataFrame, side: Side, expiry: str, strike: float, quantity: int) -> Leg:
    leg = snap_to_chain(chain, side, expiry, strike)
    if quantity == leg.quantity:
        return leg
    return Leg(
        contract_symbol=leg.contract_symbol,
        side=leg.side,
        strike=leg.strike,
        expiry=leg.expiry,
        iv=leg.iv,
        mid_price=leg.mid_price,
        quantity=quantity,
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
    """Build a debit vertical spread."""
    width = abs(float(width))
    quantity = abs(int(lots))
    lower_strike = spot - width * 0.5
    upper_strike = spot + width * 0.5
    if str(side).lower().startswith("c"):
        return [
            _snap(chain, "call", expiry, lower_strike, quantity),
            _snap(chain, "call", expiry, upper_strike, -quantity),
        ]
    return [
        _snap(chain, "put", expiry, upper_strike, quantity),
        _snap(chain, "put", expiry, lower_strike, -quantity),
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
    """Buy ``long_qty`` near strike and sell ``short_qty`` further OTM."""
    width = abs(float(width))
    if str(side).lower().startswith("c"):
        return [
            _snap(chain, "call", expiry, spot - width * 0.25, abs(int(long_qty))),
            _snap(chain, "call", expiry, spot + width * 0.75, -abs(int(short_qty))),
        ]
    return [
        _snap(chain, "put", expiry, spot + width * 0.25, abs(int(long_qty))),
        _snap(chain, "put", expiry, spot - width * 0.75, -abs(int(short_qty))),
    ]


def preset_straddle(chain: pd.DataFrame, expiry: str, spot: float, *, lots: int = 1) -> list[Leg]:
    quantity = abs(int(lots))
    return [
        _snap(chain, "call", expiry, spot, quantity),
        _snap(chain, "put", expiry, spot, quantity),
    ]


def preset_strangle(
    chain: pd.DataFrame,
    expiry: str,
    spot: float,
    *,
    offset: float,
    lots: int = 1,
) -> list[Leg]:
    offset = abs(float(offset))
    quantity = abs(int(lots))
    return [
        _snap(chain, "put", expiry, spot - offset, quantity),
        _snap(chain, "call", expiry, spot + offset, quantity),
    ]


def preset_butterfly(
    chain: pd.DataFrame,
    expiry: str,
    spot: float,
    *,
    width: float,
    lots: int = 1,
) -> list[Leg]:
    """Build a long call butterfly."""
    width = abs(float(width))
    quantity = abs(int(lots))
    return [
        _snap(chain, "call", expiry, spot - width, quantity),
        _snap(chain, "call", expiry, spot, -2 * quantity),
        _snap(chain, "call", expiry, spot + width, quantity),
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
    """Build a long-wing, short-body iron condor."""
    inner = abs(float(width_inner))
    outer = abs(float(width_outer))
    if outer <= inner:
        outer = inner + 1.0
    quantity = abs(int(lots))
    return [
        _snap(chain, "put", expiry, spot - outer, quantity),
        _snap(chain, "put", expiry, spot - inner, -quantity),
        _snap(chain, "call", expiry, spot + inner, -quantity),
        _snap(chain, "call", expiry, spot + outer, quantity),
    ]


def preset_iron_butterfly(
    chain: pd.DataFrame,
    expiry: str,
    spot: float,
    *,
    width: float,
    lots: int = 1,
) -> list[Leg]:
    """Build a short straddle with long OTM wings."""
    width = abs(float(width))
    quantity = abs(int(lots))
    return [
        _snap(chain, "put", expiry, spot, -quantity),
        _snap(chain, "call", expiry, spot, -quantity),
        _snap(chain, "put", expiry, spot - width, quantity),
        _snap(chain, "call", expiry, spot + width, quantity),
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
    """Dispatch a preset factory with typed parameters."""
    if preset_key not in PRESETS:
        raise KeyError(preset_key)
    spec = PRESETS[preset_key]
    kwargs: dict[str, Any] = {}
    for param in spec.params:
        raw = params.get(param.name, param.default)
        if param.kind == "int":
            kwargs[param.name] = int(raw)
        elif param.kind == "float":
            kwargs[param.name] = float(raw)
        else:
            kwargs[param.name] = str(raw)
    return spec.factory(chain, expiry, spot, **kwargs)


def legs_to_dataframe(legs: Sequence[Leg]) -> pd.DataFrame:
    """Serialize legs for a dataframe editor."""
    return pd.DataFrame(
        [
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
    )


def legs_from_dataframe(df: pd.DataFrame) -> list[Leg]:
    """Rebuild legs from editor output, skipping rows marked for removal."""
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


__all__ = [
    "PRESETS",
    "Leg",
    "ParamSpec",
    "PresetFactory",
    "PresetSpec",
    "Side",
    "build_preset_legs",
    "leg_from_row",
    "legs_from_dataframe",
    "legs_to_dataframe",
    "preset_butterfly",
    "preset_iron_butterfly",
    "preset_iron_condor",
    "preset_long_call",
    "preset_long_put",
    "preset_ratio_spread",
    "preset_short_call",
    "preset_short_put",
    "preset_straddle",
    "preset_strangle",
    "preset_vertical_spread",
    "snap_to_chain",
]
