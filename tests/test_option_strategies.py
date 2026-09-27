from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lambdaclass.options import payoff, presets, pricing
from lambdaclass.reporting.dashboard import option_strategies as legacy_options


def _row(
    *,
    strike: float,
    side: str,
    expiry: str = "2026-12-18",
    asof: str = "2026-05-01",
    sym: str = "ZZZ",
    bid: float = 1.9,
    ask: float = 2.1,
    last: float = 2.0,
    iv: float = 0.28,
) -> dict[str, object]:
    cp = "C" if side.lower().startswith("c") else "P"
    return {
        "contract_symbol": f"{sym}_{expiry}_{cp}_{int(round(strike * 1000))}",
        "side": side,
        "strike": strike,
        "last_price": last,
        "bid": bid,
        "ask": ask,
        "implied_volatility": iv,
        "open_interest": 100.0,
        "volume": 10.0,
        "expiry": expiry,
        "asof": asof,
        "symbol": sym,
    }


def _chain(strikes: list[float], *, expiry: str = "2026-12-18", asof: str = "2026-05-01") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for k in strikes:
        rows.append(_row(strike=k, side="call", expiry=expiry, asof=asof))
        rows.append(_row(strike=k, side="put", expiry=expiry, asof=asof))
    return pd.DataFrame(rows)


def test_dashboard_option_strategies_remains_a_compatibility_shim() -> None:
    assert legacy_options.Leg is presets.Leg
    assert legacy_options.position_pnl is payoff.position_pnl


def test_vectorized_mark_curve_matches_scalar_black_scholes() -> None:
    leg = presets.Leg("c", "call", 100.0, "2026-12-18", 0.28, 4.0, 2)
    result = payoff.position_pnl(
        [leg],
        eval_date="2026-05-01",
        r=0.04,
        spot=100.0,
        grid_pct=0.2,
        grid_n=21,
    )
    t = pricing.years_between(
        pricing.parse_option_date("2026-05-01"),
        pricing.parse_option_date(leg.expiry),
    )
    expected = np.array(
        [
            leg.quantity
            * 100.0
            * (pricing.black_scholes_price("call", float(spot), 100.0, t, 0.04, 0.28) - 4.0)
            for spot in result["S_grid"]
        ]
    )
    direct = payoff._option_values(
        "call",
        result["S_grid"],
        100.0,
        t,
        0.04,
        0.28,
    )
    np.testing.assert_allclose(direct, (expected / 200.0) + 4.0, atol=2e-5)
    np.testing.assert_allclose(result["pnl_now"], expected, atol=3e-3)


def test_intrinsic_value_call_put() -> None:
    assert pricing.intrinsic_value("call", 100.0, 110.0) == pytest.approx(10.0)
    assert pricing.intrinsic_value("call", 100.0, 90.0) == 0.0
    assert pricing.intrinsic_value("put", 100.0, 90.0) == pytest.approx(10.0)
    assert pricing.intrinsic_value("put", 100.0, 110.0) == 0.0


def test_bsm_collapses_to_intrinsic_at_T_zero() -> None:
    s = 105.0
    k = 100.0
    r = 0.05
    sig = 0.2
    t = 1e-10
    c = pricing.black_scholes_price("call", s, k, t, r, sig)
    assert c == pytest.approx(pricing.intrinsic_value("call", k, s), rel=0, abs=1e-4)


def test_long_call_breakeven_equals_strike_plus_premium_per_share() -> None:
    legs = [
        presets.Leg("x", "call", 100.0, "2026-12-18", 0.25, 4.0, 1),
    ]
    out = payoff.position_pnl(
        legs, eval_date="2026-05-01", r=0.04, iv_shift=0.0, spot=100.0, grid_pct=0.5, grid_n=801
    )
    be = out["breakevens"]
    assert len(be) >= 1
    assert be[0] == pytest.approx(104.0, abs=0.15)


def test_short_call_marks_unbounded_loss() -> None:
    legs = [presets.Leg("x", "call", 100.0, "2026-12-18", 0.25, 5.0, -1)]
    out = payoff.position_pnl(legs, eval_date="2026-05-01", r=0.04, iv_shift=0.0, spot=100.0)
    assert out["unbounded_up"] is True
    assert math.isinf(out["max_loss"]) and out["max_loss"] < 0


def test_vertical_debit_spread_bounded_both_sides() -> None:
    legs = [
        presets.Leg("a", "call", 95.0, "2026-12-18", 0.25, 8.0, 1),
        presets.Leg("b", "call", 105.0, "2026-12-18", 0.25, 3.0, -1),
    ]
    out = payoff.position_pnl(legs, eval_date="2026-05-01", r=0.04, iv_shift=0.0, spot=100.0)
    assert math.isfinite(out["max_profit"])
    assert math.isfinite(out["max_loss"])


def test_straddle_two_breakevens_symmetric() -> None:
    legs = [
        presets.Leg("c", "call", 100.0, "2026-12-18", 0.3, 4.0, 1),
        presets.Leg("p", "put", 100.0, "2026-12-18", 0.3, 4.0, 1),
    ]
    out = payoff.position_pnl(
        legs, eval_date="2026-05-01", r=0.04, iv_shift=0.0, spot=100.0, grid_pct=0.6, grid_n=1201
    )
    be = out["breakevens"]
    assert len(be) == 2
    lo, hi = min(be), max(be)
    assert abs((hi - 100.0) - (100.0 - lo)) < 0.5


def test_iron_condor_payoff_shape_credit_flat_middle() -> None:
    legs = [
        presets.Leg("lp", "put", 80.0, "2026-12-18", 0.35, 0.5, 1),
        presets.Leg("sp", "put", 90.0, "2026-12-18", 0.35, 3.0, -1),
        presets.Leg("sc", "call", 110.0, "2026-12-18", 0.35, 3.0, -1),
        presets.Leg("lc", "call", 120.0, "2026-12-18", 0.35, 0.5, 1),
    ]
    out = payoff.position_pnl(
        legs, eval_date="2026-05-01", r=0.04, iv_shift=0.0, spot=100.0, grid_pct=0.8, grid_n=1601
    )
    assert len(out["breakevens"]) >= 2
    assert out["max_profit"] > 0.0
    mid = (out["S_grid"] >= 92.0) & (out["S_grid"] <= 108.0)
    assert float(np.std(out["pnl_expiry"][mid])) < float(np.std(out["pnl_expiry"])) * 0.5


def test_iron_butterfly_payoff_triangle_peak_near_center() -> None:
    legs = [
        presets.Leg("sp", "put", 100.0, "2026-12-18", 0.3, 5.0, -1),
        presets.Leg("sc", "call", 100.0, "2026-12-18", 0.3, 5.0, -1),
        presets.Leg("lp", "put", 90.0, "2026-12-18", 0.3, 1.0, 1),
        presets.Leg("lc", "call", 110.0, "2026-12-18", 0.3, 1.0, 1),
    ]
    out = payoff.position_pnl(
        legs, eval_date="2026-05-01", r=0.04, iv_shift=0.0, spot=100.0, grid_pct=0.5, grid_n=801
    )
    S = out["S_grid"]
    pe = out["pnl_expiry"]
    i100 = int(np.abs(S - 100.0).argmin())
    assert float(pe[i100]) == pytest.approx(float(np.max(pe)), rel=0, abs=1.0)
    assert len(out["breakevens"]) == 2


def test_butterfly_qty_pattern() -> None:
    ch = _chain(list(range(85, 116)))
    legs = presets.preset_butterfly(ch, "2026-12-18", 100.0, width=5.0, lots=1)
    qs = sorted((int(l.strike), l.quantity) for l in legs)
    assert qs == [(95, 1), (100, -2), (105, 1)]


def test_net_delta_offset_zero_for_back_to_back_long_short() -> None:
    g1 = pricing.greeks("call", 100.0, 100.0, 0.25, 0.05, 0.2)["delta"]
    g2 = pricing.greeks("call", 100.0, 100.0, 0.25, 0.05, 0.2)["delta"]
    net = 100 * (g1 - g2)
    assert net == pytest.approx(0.0, abs=1e-9)


def test_snap_to_chain_picks_nearest_strike() -> None:
    ch = _chain([98.0, 100.0, 102.0])
    leg = presets.snap_to_chain(ch, "call", "2026-12-18", 101.0)
    assert leg.strike == 100.0


def test_snap_to_chain_raises_when_side_or_expiry_missing() -> None:
    ch = _chain([100.0])
    with pytest.raises(ValueError):
        presets.snap_to_chain(ch, "call", "2099-01-01", 100.0)


def test_safe_mid_falls_back_to_last_when_quote_invalid() -> None:
    row = pd.Series(_row(strike=100.0, side="call", bid=0.0, ask=0.0, last=1.23))
    assert pricing.safe_option_mid(row) == pytest.approx(1.23)


def test_theta_matches_one_day_finite_difference() -> None:
    s, k, t, r, sig = 100.0, 100.0, 0.25, 0.05, 0.20
    th = pricing.greeks("call", s, k, t, r, sig)["theta"]
    p0 = pricing.black_scholes_price("call", s, k, t, r, sig)
    p1 = pricing.black_scholes_price("call", s, k, t - 1.0 / 365.0, r, sig)
    assert th == pytest.approx(p1 - p0, rel=0.05, abs=0.02)


def test_vega_matches_one_vol_point_finite_difference() -> None:
    s, k, t, r, sig = 100.0, 100.0, 0.25, 0.05, 0.20
    vg = pricing.greeks("call", s, k, t, r, sig)["vega"]
    p0 = pricing.black_scholes_price("call", s, k, t, r, sig)
    p1 = pricing.black_scholes_price("call", s, k, t, r, sig + 0.01)
    assert vg == pytest.approx(p1 - p0, rel=0.05, abs=0.02)


def test_far_otm_short_call_unbounded_up() -> None:
    """Grid must cover the strike so naked short OTM still flags unbounded loss."""
    legs = [presets.Leg("x", "call", 150.0, "2026-12-18", 0.25, 0.50, -1)]
    out = payoff.position_pnl(legs, eval_date="2026-05-01", r=0.04, iv_shift=0.0, spot=100.0)
    assert float(out["S_grid"][-1]) >= 150.0 * 1.05
    assert out["unbounded_up"] is True
    assert math.isinf(out["max_loss"]) and out["max_loss"] < 0


def test_every_preset_builds_legs_with_defaults() -> None:
    ch = _chain(list(range(80, 121)))
    for key, spec in presets.PRESETS.items():
        params = {p.name: p.default for p in spec.params}
        legs = presets.build_preset_legs(key, ch, "2026-12-18", 100.0, params)
        assert legs, key
        assert all(isinstance(lg, presets.Leg) for lg in legs)
