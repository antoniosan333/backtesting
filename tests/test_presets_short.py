import pandas as pd
from lambdaclass.options.presets import (
    preset_short_straddle,
    preset_short_strangle,
    PRESETS,
)


def _dummy_chain():
    rows = []
    for side in ("call", "put"):
        for strike in (95, 100, 105):
            rows.append({
                "contract_symbol": f"TEST_{strike}{side[0].upper()}",
                "side": side,
                "strike": strike,
                "expiry": "2026-12-19",
                "implied_volatility": 0.35,
                "bid": 3.0,
                "ask": 3.2,
                "underlying_last": 100.0,
            })
    return pd.DataFrame(rows)


def test_preset_short_straddle():
    chain = _dummy_chain()
    legs = preset_short_straddle(chain, "2026-12-19", 100.0, lots=1)
    assert len(legs) == 2
    assert all(leg.quantity < 0 for leg in legs)
    sides = {leg.side for leg in legs}
    assert sides == {"call", "put"}


def test_preset_short_strangle():
    chain = _dummy_chain()
    legs = preset_short_strangle(chain, "2026-12-19", 100.0, offset=5.0, lots=1)
    assert len(legs) == 2
    assert all(leg.quantity < 0 for leg in legs)


def test_short_presets_in_registry():
    assert "short_straddle" in PRESETS
    assert "short_strangle" in PRESETS