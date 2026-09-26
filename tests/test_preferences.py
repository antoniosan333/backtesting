import tomllib
from pathlib import Path

from lambdaclass.config import Preferences, build_snapshot_payload, snapshot_preferences


def test_preferences_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "preferences.toml"
    prefs = Preferences()
    prefs.defaults.starting_capital = 250_000
    prefs.save(path)
    loaded = Preferences.load(path)
    assert loaded.defaults.starting_capital == 250_000
    assert loaded.defaults.data_adapter == "yfinance"


def test_snapshot_omits_none_cli_overrides_from_file_and_hash_payload(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.toml"
    prefs = Preferences()

    snapshot_preferences(
        prefs,
        strategy_params={},
        cli_overrides={"start": None, "end": None, "symbol": "SPY", "options_source": "optionsdx"},
        output_path=path,
    )

    snapshot = tomllib.loads(path.read_text(encoding="utf-8"))
    assert snapshot["cli_overrides"] == {"symbol": "SPY", "options_source": "optionsdx"}
    payload = build_snapshot_payload(
        prefs,
        strategy_params={},
        cli_overrides={"start": None, "end": None, "symbol": "SPY", "options_source": "optionsdx"},
    )
    assert payload["cli_overrides"] == {"symbol": "SPY", "options_source": "optionsdx"}
