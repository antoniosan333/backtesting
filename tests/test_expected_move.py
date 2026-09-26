from __future__ import annotations

import math

import pandas as pd
import pytest

from lambdaclass.options.expected_move import (
    expected_move_for_expiry,
    expected_move_for_horizon,
    expected_moves,
)
from lambdaclass.options.pricing import black_scholes_price


def _quote(
    *,
    side: str,
    strike: float,
    mid: float,
    iv: float = 0.20,
    expiry: str = "2026-02-06",
    asof: str = "2026-01-02",
    dte: float = 35.0,
    spot: float = 100.0,
    spread: float = 0.02,
) -> dict[str, object]:
    return {
        "contract_symbol": f"XYZ_{expiry}_{side}_{strike}",
        "side": side,
        "strike": strike,
        "last_price": mid,
        "bid": mid - spread / 2,
        "ask": mid + spread / 2,
        "implied_volatility": iv,
        "expiry": expiry,
        "asof": asof,
        "underlying_last": spot,
        "dte": dte,
    }


def test_flat_vol_bsm_chain_produces_consistent_expected_moves() -> None:
    spot = 100.0
    sigma = 0.20
    dte = 35.0
    t = dte / 365.0
    rows = []
    for strike in (95.0, 100.0, 105.0):
        for side in ("call", "put"):
            mid = black_scholes_price(side, spot, strike, t, 0.0, sigma)
            rows.append(_quote(side=side, strike=strike, mid=mid, iv=sigma, dte=dte))

    move = expected_move_for_expiry(pd.DataFrame(rows), "2026-02-06", spot)

    theoretical = spot * sigma * math.sqrt(t)
    assert move.atm_strike == 100.0
    assert move.atm_iv == pytest.approx(sigma)
    assert move.iv_1sd == pytest.approx(theoretical)
    assert move.tos == pytest.approx(theoretical * 0.85)
    assert move.straddle == pytest.approx(theoretical * math.sqrt(2 / math.pi), rel=0.03)
    assert move.straddle_1sd == pytest.approx(theoretical, rel=0.03)
    assert move.quality == "ok"


def test_weighted_expected_move_matches_published_example() -> None:
    rows = [
        _quote(side="call", strike=121, mid=2.20),
        _quote(side="put", strike=121, mid=2.20),
        _quote(side="call", strike=122, mid=1.73),
        _quote(side="put", strike=120, mid=1.73),
        _quote(side="call", strike=123, mid=1.33),
        _quote(side="put", strike=119, mid=1.33),
    ]

    move = expected_move_for_expiry(pd.DataFrame(rows), "2026-02-06", 121.0)

    assert move.straddle == pytest.approx(4.40)
    assert move.weighted == pytest.approx(3.944)


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        ([_quote(side="call", strike=100, mid=2.0)], "MISSING_PUT"),
        (
            [
                {**_quote(side="call", strike=100, mid=2.0), "bid": 2.2, "ask": 2.0},
                _quote(side="put", strike=100, mid=2.0),
            ],
            "CROSSED_MARKET",
        ),
        (
            [
                {**_quote(side="call", strike=100, mid=2.0), "bid": 1.0, "ask": 3.0},
                _quote(side="put", strike=100, mid=2.0),
            ],
            "WIDE_SPREAD",
        ),
    ],
)
def test_invalid_atm_quote_returns_no_straddle(rows: list[dict[str, object]], reason: str) -> None:
    move = expected_move_for_expiry(pd.DataFrame(rows), "2026-02-06", 100.0)

    assert move.straddle is None
    assert reason in move.quality


def test_batch_and_horizon_selection() -> None:
    chain = pd.DataFrame(
        [
            _quote(side=side, strike=100, mid=2.0, expiry=expiry, dte=dte)
            for expiry, dte in (("2026-01-09", 7.0), ("2026-02-06", 35.0))
            for side in ("call", "put")
        ]
    )

    result = expected_moves(chain, 100.0)
    selected = expected_move_for_horizon(chain, 100.0, 30.0)

    assert result["expiry"].tolist() == ["2026-01-09", "2026-02-06"]
    assert selected.expiry == "2026-02-06"
