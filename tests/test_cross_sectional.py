# tests/test_cross_sectional.py
import pandas as pd
from lambdaclass.volatility.cross_sectional import rank_symbols


def test_rank_symbols_basic():
    event_table = pd.DataFrame({
        "earnings_date": ["2026-01-01", "2026-01-01", "2026-04-01", "2026-04-01"],
        "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
        "crush_pct": [0.30, 0.15, 0.25, 0.20],
        "ramp_pct": [0.10, 0.05, 0.12, 0.08],
        "realized_over_implied": [1.2, 0.8, 1.1, 0.9],
    })
    ranking = rank_symbols(event_table)
    assert "symbol" in ranking.columns
    assert "median_crush" in ranking.columns
    assert "median_ramp" in ranking.columns
    assert "median_realized_over_implied" in ranking.columns
    assert "score" in ranking.columns
    # AAPL has higher realized_over_implied → better for long straddle
    assert ranking.iloc[0]["symbol"] == "AAPL"


def test_rank_symbols_empty():
    ranking = rank_symbols(pd.DataFrame())
    assert ranking.empty