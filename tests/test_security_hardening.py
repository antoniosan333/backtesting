from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from lambdaclass.cli import _validate_strategy_name, app
from lambdaclass.config import Preferences, snapshot_preferences
from lambdaclass.symbols import validate_symbol

runner = CliRunner()


def test_strategy_name_validation_rejects_path_characters() -> None:
    with pytest.raises(typer.BadParameter):
        _validate_strategy_name("../escape")


@pytest.mark.parametrize("symbol", ["SPY", "BRK.B", "^VIX"])
def test_symbol_validation_accepts_supported_ticker_formats(symbol: str) -> None:
    assert validate_symbol(symbol.lower()) == symbol


@pytest.mark.parametrize(
    "symbol",
    ["", "../SPY", "SPY/../../secret", r"SPY\secret", " SPY", "SPY ", "SP Y", "A" * 13],
)
def test_symbol_validation_rejects_unsafe_or_unreasonable_values(symbol: str) -> None:
    with pytest.raises(ValueError, match="symbol"):
        validate_symbol(symbol)


@pytest.mark.parametrize(
    ("command", "extra_args"),
    [
        ("fetch", ["--start", "2024-01-01"]),
        ("fetch-earnings", []),
        ("run", ["alpha", "--symbol"]),
    ],
)
def test_cli_commands_reject_invalid_symbols(command: str, extra_args: list[str]) -> None:
    args = [command, *extra_args, "../SPY"] if command == "run" else [command, "../SPY", *extra_args]

    result = runner.invoke(app, args)

    assert result.exit_code != 0
    assert "symbol" in result.output.lower()


def test_snapshot_redacts_sensitive_keys(tmp_path: Path) -> None:
    destination = tmp_path / "config.snapshot.toml"
    snapshot_preferences(
        preferences=Preferences(),
        strategy_params={"api_key": "secret-123", "units": 2},
        cli_overrides={"password": "pw", "start": "2020-01-01"},
        output_path=destination,
    )
    content = destination.read_text(encoding="utf-8")
    assert "***REDACTED***" in content
    assert "secret-123" not in content
    assert "pw" not in content
