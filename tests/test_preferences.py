from pathlib import Path

from lambdaclass.config import Preferences


def test_preferences_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "preferences.toml"
    prefs = Preferences()
    prefs.defaults.starting_capital = 250_000
    prefs.save(path)
    loaded = Preferences.load(path)
    assert loaded.defaults.starting_capital == 250_000
    assert loaded.defaults.data_adapter == "yfinance"
    assert loaded.expected_move.skew_factor == 0.85
    assert loaded.expected_move.horizons_dte == [0, 7, 30]
    assert loaded.volatility.history_start == "2019-01-01"
    assert loaded.volatility.lookback_days == 252
