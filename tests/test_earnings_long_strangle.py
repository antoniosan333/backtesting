import importlib.util
from pathlib import Path

import pandas as pd
from lambdaclass.strategies.base import StrategyContext


def _load_strategy(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StrategyImpl()


def _dummy_chain():
    """Minimal options chain with strikes around spot=100 for OTM strangle."""
    rows = []
    for side in ("call", "put"):
        for strike in (90, 95, 100, 105, 110):
            rows.append(
                {
                    "contract_symbol": f"TEST_{strike}{side[0].upper()}",
                    "side": side,
                    "strike": float(strike),
                    "expiry": "2026-12-19",
                    "implied_volatility": 0.35,
                    "bid": 3.0,
                    "ask": 3.2,
                    "underlying_last": 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_earnings_long_strangle_enters_before_earnings():
    """Strategy enters a 2-leg OTM strangle when days_to_next_earnings == enter_days_before."""
    strat = _load_strategy("elsg", "strategies/2026-09/earnings_long_strangle.py")
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=5,
        options_chain=_dummy_chain(),
    )
    decision = strat.on_bar(ctx)
    assert len(decision.option_legs) == 2  # call + put
    sides = {leg.side for leg in decision.option_legs}
    assert sides == {"call", "put"}
    # OTM strangle: call strike > spot, put strike < spot
    call_leg = next(leg for leg in decision.option_legs if leg.side == "call")
    put_leg = next(leg for leg in decision.option_legs if leg.side == "put")
    assert float(call_leg.strike) > 100.0
    assert float(put_leg.strike) < 100.0
    assert decision.metadata.get("reason") == "earnings_entry"


def test_earnings_long_strangle_holds_when_not_entry_day():
    """Strategy holds (no legs) when days_to_next_earnings != enter_days_before."""
    strat = _load_strategy("elsg", "strategies/2026-09/earnings_long_strangle.py")
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=10,
        options_chain=_dummy_chain(),
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []