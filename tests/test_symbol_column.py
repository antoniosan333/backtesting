import pandas as pd
from lambdaclass.volatility.earnings_cycle import align_events, event_table


def test_align_events_includes_symbol():
    vol_df = pd.DataFrame({
        "date": [f"2026-01-{d:02d}" for d in range(1, 32)],
        "close": [100.0] * 31,
        "iv_front": [0.3] * 31,
        "iv30": [0.3] * 31,
        "implied_event_move": [3.0] * 31,
        "front_straddle": [3.0] * 31,
    })
    earnings = pd.DataFrame({
        "earnings_date": ["2026-01-15"],
        "timing": ["AMC"],
    })
    aligned = align_events(vol_df, earnings, pre_days=10, post_days=5, symbol="AAPL")
    assert "symbol" in aligned.columns
    assert (aligned["symbol"] == "AAPL").all()


def test_align_events_without_symbol():
    vol_df = pd.DataFrame({
        "date": [f"2026-01-{d:02d}" for d in range(1, 32)],
        "close": [100.0] * 31,
        "iv_front": [0.3] * 31,
        "iv30": [0.3] * 31,
        "implied_event_move": [3.0] * 31,
        "front_straddle": [3.0] * 31,
    })
    earnings = pd.DataFrame({
        "earnings_date": ["2026-01-15"],
        "timing": ["AMC"],
    })
    aligned = align_events(vol_df, earnings, pre_days=10, post_days=5)
    assert not aligned.empty
    # symbol column should not be present when not passed
    assert "symbol" not in aligned.columns


def test_event_table_preserves_symbol():
    vol_df = pd.DataFrame({
        "date": [f"2026-01-{d:02d}" for d in range(1, 32)],
        "close": [100.0 + d for d in range(31)],
        "iv_front": [0.3] * 31,
        "iv30": [0.3] * 31,
        "implied_event_move": [3.0] * 31,
        "front_straddle": [3.0] * 31,
    })
    earnings = pd.DataFrame({
        "earnings_date": ["2026-01-15"],
        "timing": ["AMC"],
    })
    aligned = align_events(vol_df, earnings, pre_days=20, post_days=5, symbol="MSFT")
    events = event_table(aligned)
    assert not events.empty
    assert "symbol" in events.columns
    assert (events["symbol"] == "MSFT").all()