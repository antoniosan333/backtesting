import tomllib
from pathlib import Path

import pytest

from lambdaclass.config import Preferences, _dump_toml, build_snapshot_payload, snapshot_preferences


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


def test_environment_strings_are_coerced_by_pydantic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAMBDACLASS__DEFAULTS__STARTING_CAPITAL", "123456.5")
    monkeypatch.setenv("LAMBDACLASS__DEFAULTS__ALLOW_NEGATIVE_CASH", "true")
    monkeypatch.setenv("LAMBDACLASS__RISK__MAX_OPEN_POSITIONS", "7")

    prefs = Preferences.load(tmp_path / "missing.toml")

    assert prefs.defaults.starting_capital == 123456.5
    assert prefs.defaults.allow_negative_cash is True
    assert prefs.risk.max_open_positions == 7


def test_toml_format_error_identifies_unsupported_key_and_value() -> None:
    with pytest.raises(TypeError, match=r"section\.unsupported.*list"):
        _dump_toml({"section": {"unsupported": []}})
