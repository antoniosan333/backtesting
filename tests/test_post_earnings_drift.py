# tests/test_post_earnings_drift.py
import pandas as pd
from lambdaclass.volatility.post_earnings_drift import analyze_drift


def test_analyze_drift_basic():
    vol_series = pd.DataFrame({
        "date": [f"2026-01-{d:02d}" for d in range(1, 32)],
        "close": [100.0 + d for d in range(31)],
        "iv30": [0.35] * 31,
        "rv_fwd21": [0.40] * 31,
        "days_to_next_earnings": list(range(15, -16, -1)),
        "days_since_last_earnings": [None] * 15 + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    })
    earnings = pd.DataFrame({
        "earnings_date": ["2026-01-16"],
        "timing": ["AMC"],
    })
    result = analyze_drift(vol_series, earnings)
    assert "pre_iv30" in result.columns
    assert "post_rv21" in result.columns
    assert "vrp" in result.columns  # vol risk premium = iv30 - rv_fwd21
    assert len(result) == 1  # one event


def test_analyze_drift_empty():
    result = analyze_drift(pd.DataFrame(), pd.DataFrame())
    assert result.empty