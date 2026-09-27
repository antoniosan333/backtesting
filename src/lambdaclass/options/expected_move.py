"""Vendor-neutral expected-move calculations from an option chain."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import math

import pandas as pd

from lambdaclass.options.pricing import parse_option_date


@dataclass(frozen=True)
class ExpectedMove:
    expiry: str
    dte: float
    spot: float
    atm_strike: float
    atm_iv: float | None
    straddle: float | None
    iv_1sd: float | None
    tos: float | None
    straddle_1sd: float | None
    weighted: float | None
    quality: str


def _number(value: object) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    result = float(parsed)
    return result if math.isfinite(result) else None


def _quote_mid(row: pd.Series, max_spread_pct: float) -> tuple[float | None, str | None]:
    bid = _number(row.get("bid"))
    ask = _number(row.get("ask"))
    if bid is None or ask is None or bid <= 0.0 or ask <= 0.0:
        return None, "INVALID_PRICE"
    if bid > ask:
        return None, "CROSSED_MARKET"
    mid = (bid + ask) / 2.0
    if (ask - bid) / mid > max_spread_pct:
        return None, "WIDE_SPREAD"
    return mid, None


def _preferred_spot(chain: pd.DataFrame, fallback: float) -> float:
    if "underlying_last" in chain:
        values = pd.to_numeric(chain["underlying_last"], errors="coerce")
        values = values[(values > 0) & values.notna()]
        if not values.empty:
            return float(values.median())
    return float(fallback)


def _dte(chain: pd.DataFrame, expiry: str) -> float:
    if "dte" in chain:
        values = pd.to_numeric(chain["dte"], errors="coerce")
        values = values[(values >= 0) & values.notna()]
        if not values.empty:
            return float(values.median())
    if "asof" in chain and not chain.empty:
        asof = parse_option_date(str(chain.iloc[0]["asof"]))
        return float(max((parse_option_date(expiry) - asof).days, 0))
    return 0.0


def _iv_at_strike(chain: pd.DataFrame, strike: float) -> float | None:
    rows = chain[chain["strike"] == strike]
    values = pd.to_numeric(rows.get("implied_volatility"), errors="coerce")
    values = values[(values > 0) & values.notna()]
    return float(values.mean()) if not values.empty else None


def interpolated_atm_iv(chain: pd.DataFrame, spot: float) -> float | None:
    strikes = sorted(float(value) for value in pd.to_numeric(chain["strike"], errors="coerce").dropna().unique())
    points = [(strike, _iv_at_strike(chain, strike)) for strike in strikes]
    points = [(strike, iv) for strike, iv in points if iv is not None]
    if not points:
        return None
    below = [point for point in points if point[0] <= spot]
    above = [point for point in points if point[0] >= spot]
    if below and above:
        low_k, low_iv = below[-1]
        high_k, high_iv = above[0]
        if high_k == low_k:
            return low_iv
        weight = (spot - low_k) / (high_k - low_k)
        return low_iv + weight * (high_iv - low_iv)
    return min(points, key=lambda point: abs(point[0] - spot))[1]


def _side_row(chain: pd.DataFrame, strike: float, side: str) -> pd.Series | None:
    matches = chain[(chain["strike"] == strike) & (chain["side"] == side)]
    return None if matches.empty else matches.iloc[0]


def _pair_price(
    chain: pd.DataFrame,
    call_strike: float,
    put_strike: float,
    max_spread_pct: float,
) -> tuple[float | None, list[str]]:
    reasons: list[str] = []
    call = _side_row(chain, call_strike, "call")
    put = _side_row(chain, put_strike, "put")
    if call is None:
        reasons.append("MISSING_CALL")
    if put is None:
        reasons.append("MISSING_PUT")
    if reasons:
        return None, reasons
    call_mid, call_reason = _quote_mid(call, max_spread_pct)
    put_mid, put_reason = _quote_mid(put, max_spread_pct)
    reasons.extend(reason for reason in (call_reason, put_reason) if reason)
    if call_mid is None or put_mid is None:
        return None, reasons
    return call_mid + put_mid, reasons


def expected_move_for_expiry(
    chain: pd.DataFrame,
    expiry: str,
    spot: float,
    *,
    skew_factor: float = 0.85,
    max_spread_pct: float = 0.5,
) -> ExpectedMove:
    """Calculate expected-move variants for one expiration."""
    expiry_text = str(expiry)[:10]
    subset = chain[chain["expiry"].astype(str).str[:10] == expiry_text].copy()
    if subset.empty:
        raise ValueError(f"expiry {expiry_text!r} is not present in the chain")
    subset["side"] = subset["side"].astype(str).str.lower()
    subset["strike"] = pd.to_numeric(subset["strike"], errors="coerce")
    subset = subset.dropna(subset=["strike"])
    resolved_spot = _preferred_spot(subset, spot)
    strikes = sorted(float(value) for value in subset["strike"].unique())
    if not strikes:
        raise ValueError("chain has no valid strikes")
    atm_strike = min(strikes, key=lambda strike: (abs(strike - resolved_spot), strike))
    dte = _dte(subset, expiry_text)
    atm_iv = interpolated_atm_iv(subset, resolved_spot)
    straddle, reasons = _pair_price(subset, atm_strike, atm_strike, max_spread_pct)

    iv_1sd = None
    if atm_iv is not None:
        iv_1sd = resolved_spot * atm_iv * math.sqrt(max(dte, 1.0) / 365.0)

    below = [strike for strike in strikes if strike < atm_strike]
    above = [strike for strike in strikes if strike > atm_strike]
    weighted = None
    if straddle is not None and len(below) >= 2 and len(above) >= 2:
        first, first_reasons = _pair_price(
            subset, above[0], below[-1], max_spread_pct
        )
        second, second_reasons = _pair_price(
            subset, above[1], below[-2], max_spread_pct
        )
        reasons.extend(first_reasons)
        reasons.extend(second_reasons)
        if first is not None and second is not None:
            weighted = 0.6 * straddle + 0.3 * first + 0.1 * second
    elif straddle is not None:
        reasons.append("MISSING_OTM_WINGS")

    quality = "|".join(sorted(set(reasons))) if reasons else "ok"
    return ExpectedMove(
        expiry=expiry_text,
        dte=dte,
        spot=resolved_spot,
        atm_strike=atm_strike,
        atm_iv=atm_iv,
        straddle=straddle,
        iv_1sd=iv_1sd,
        tos=None if iv_1sd is None else iv_1sd * skew_factor,
        straddle_1sd=None if straddle is None else straddle * 1.25,
        weighted=weighted,
        quality=quality,
    )


def expected_moves(
    chain: pd.DataFrame,
    spot: float,
    *,
    expiries: list[str] | tuple[str, ...] | None = None,
    skew_factor: float = 0.85,
    max_spread_pct: float = 0.5,
) -> pd.DataFrame:
    """Return one expected-move row per expiration."""
    if chain is None or chain.empty or "expiry" not in chain:
        return pd.DataFrame(columns=ExpectedMove.__dataclass_fields__)
    available = sorted(chain["expiry"].dropna().astype(str).str[:10].unique())
    selected = available if expiries is None else [str(value)[:10] for value in expiries if str(value)[:10] in available]
    rows = [
        asdict(
            expected_move_for_expiry(
                chain,
                expiry,
                spot,
                skew_factor=skew_factor,
                max_spread_pct=max_spread_pct,
            )
        )
        for expiry in selected
    ]
    return pd.DataFrame(rows, columns=ExpectedMove.__dataclass_fields__)


def expected_move_for_horizon(
    chain: pd.DataFrame,
    spot: float,
    target_dte: float,
    **kwargs: float,
) -> ExpectedMove:
    """Select the nearest expiration at or beyond a target DTE."""
    moves = expected_moves(chain, spot, **kwargs)
    if moves.empty:
        raise ValueError("chain has no expirations")
    candidates = moves[moves["dte"] >= float(target_dte)]
    row = candidates.sort_values("dte").iloc[0] if not candidates.empty else moves.sort_values("dte").iloc[-1]
    return ExpectedMove(**row.to_dict())
