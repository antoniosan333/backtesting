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
    """Minimal options chain with ATM call+put at strike 100."""
    rows = []
    for side in ("call", "put"):
        for strike in (95, 100, 105):
            rows.append(
                {
                    "contract_symbol": f"TEST_{strike}{side[0].upper()}",
                    "side": side,
                    "strike": strike,
                    "expiry": "2026-12-19",
                    "implied_volatility": 0.35,
                    "bid": 3.0,
                    "ask": 3.2,
                    "underlying_last": 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_long_straddle_skips_when_iv_rank_too_high():
    strat = _load_strategy("els", "strategies/2026-08/earnings_long_straddle.py")
    strat.params["max_iv_rank"] = 30.0  # only enter when IV rank <= 30
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=5,
        iv_rank_252=60.0,
        options_chain=_dummy_chain(),
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []  # filtered out


def test_long_straddle_enters_when_iv_rank_low():
    strat = _load_strategy("els", "strategies/2026-08/earnings_long_straddle.py")
    strat.params["max_iv_rank"] = 30.0
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=5,
        iv_rank_252=15.0,
        options_chain=_dummy_chain(),
    )
    decision = strat.on_bar(ctx)
    assert len(decision.option_legs) == 2  # enters


def test_long_straddle_skips_when_iv_rank_too_low():
    strat = _load_strategy("els", "strategies/2026-08/earnings_long_straddle.py")
    strat.params["min_iv_rank"] = 50.0  # only enter when IV rank >= 50
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=5,
        iv_rank_252=20.0,
        options_chain=_dummy_chain(),
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []  # filtered out


def test_long_straddle_enters_when_iv_rank_high():
    strat = _load_strategy("els", "strategies/2026-08/earnings_long_straddle.py")
    strat.params["min_iv_rank"] = 50.0
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=5,
        iv_rank_252=70.0,
        options_chain=_dummy_chain(),
    )
    decision = strat.on_bar(ctx)
    assert len(decision.option_legs) == 2  # enters


def test_long_straddle_enters_when_no_filter():
    strat = _load_strategy("els", "strategies/2026-08/earnings_long_straddle.py")
    # Both max_iv_rank and min_iv_rank are None by default
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=5,
        iv_rank_252=99.0,
        options_chain=_dummy_chain(),
    )
    decision = strat.on_bar(ctx)
    assert len(decision.option_legs) == 2  # no filter, enters


# ---------------------------------------------------------------------------
# earnings_short_iron_condor — same IV rank filter pattern (4 legs)
# ---------------------------------------------------------------------------

def _dummy_chain_wide():
    """Options chain wide enough for iron condor wings (strikes 70..130)."""
    rows = []
    for side in ("call", "put"):
        for strike in (70, 90, 100, 110, 130):
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


def test_short_iron_condor_skips_when_iv_rank_too_low():
    """min_iv_rank=50 but iv_rank=20 → skip (only sell premium when IV expensive)."""
    strat = _load_strategy("esic", "strategies/2026-08/earnings_short_iron_condor.py")
    strat.params["min_iv_rank"] = 50.0
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=1,
        iv_rank_252=20.0,
        options_chain=_dummy_chain_wide(),
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []  # filtered out


def test_short_iron_condor_enters_when_iv_rank_high():
    """min_iv_rank=50 and iv_rank=70 → enter the 4-leg iron condor."""
    strat = _load_strategy("esic", "strategies/2026-08/earnings_short_iron_condor.py")
    strat.params["min_iv_rank"] = 50.0
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=1,
        iv_rank_252=70.0,
        options_chain=_dummy_chain_wide(),
    )
    decision = strat.on_bar(ctx)
    assert len(decision.option_legs) == 4  # iron condor has 4 legs
    assert decision.metadata.get("reason") == "earnings_entry"


def test_short_iron_condor_skips_when_iv_rank_too_high():
    """max_iv_rank=30 but iv_rank=60 → skip."""
    strat = _load_strategy("esic", "strategies/2026-08/earnings_short_iron_condor.py")
    strat.params["max_iv_rank"] = 30.0
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=1,
        iv_rank_252=60.0,
        options_chain=_dummy_chain_wide(),
    )
    decision = strat.on_bar(ctx)
    assert decision.option_legs == []  # filtered out


def test_short_iron_condor_enters_when_no_filter():
    """No IV rank filter set → enters regardless of iv_rank."""
    strat = _load_strategy("esic", "strategies/2026-08/earnings_short_iron_condor.py")
    ctx = StrategyContext(
        row=pd.Series({"date": "2026-09-20", "close": 100.0}),
        cash=10000,
        position=0,
        days_to_next_earnings=1,
        iv_rank_252=99.0,
        options_chain=_dummy_chain_wide(),
    )
    decision = strat.on_bar(ctx)
    assert len(decision.option_legs) == 4  # no filter, enters