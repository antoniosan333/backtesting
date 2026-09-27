from __future__ import annotations

import tomllib
from datetime import UTC, date, datetime, tzinfo
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import lambdaclass.cli as cli
from lambdaclass.config import Preferences
from lambdaclass.storage.duckdb_store import DuckDBStore

runner = CliRunner()


class _FixedDate(date):
    @classmethod
    def today(cls) -> _FixedDate:
        return cls(2030, 1, 2)


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> _FixedDatetime:
        assert tz is UTC
        return cls(2031, 2, 3, 4, 5, 6, tzinfo=UTC)


class _RecordingAdapter:
    def __init__(self) -> None:
        self.bar_calls: list[tuple[str, date, date]] = []
        self.chain_calls: list[tuple[str, date]] = []

    def get_stock_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        self.bar_calls.append((symbol, start, end))
        return pd.DataFrame()

    def get_option_chain(self, symbol: str, asof: date) -> pd.DataFrame:
        self.chain_calls.append((symbol, asof))
        return pd.DataFrame()


class _RecordingStore:
    instances: list[_RecordingStore] = []

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.chain_writes = 0
        self.__class__.instances.append(self)

    def write_bars(self, symbol: str, bars: pd.DataFrame) -> Path:
        return self.data_root / "stocks" / f"{symbol.upper()}.parquet"

    def write_chain(self, symbol: str, chain: pd.DataFrame) -> Path:
        self.chain_writes += 1
        return self.data_root / "options" / f"{symbol.upper()}.parquet"


def test_fetch_resolves_default_end_at_call_time_and_stamps_chain_with_today(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _RecordingAdapter()
    _RecordingStore.instances.clear()
    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "_get_adapter", lambda name: adapter)
    monkeypatch.setattr(cli, "DuckDBStore", _RecordingStore)
    monkeypatch.setattr(cli, "date", _FixedDate)

    cli.fetch_data("spy", start="2029-12-01", end=None)

    assert adapter.bar_calls == [("SPY", date(2029, 12, 1), date(2030, 1, 2))]
    assert adapter.chain_calls == [("SPY", date(2030, 1, 2))]
    assert _RecordingStore.instances[-1].chain_writes == 1


def test_fetch_skips_current_chain_for_historical_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter = _RecordingAdapter()
    _RecordingStore.instances.clear()
    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "_get_adapter", lambda name: adapter)
    monkeypatch.setattr(cli, "DuckDBStore", _RecordingStore)
    monkeypatch.setattr(cli, "date", _FixedDate)

    cli.fetch_data("spy", start="2020-01-01", end="2020-01-31")

    assert adapter.chain_calls == []
    assert _RecordingStore.instances[-1].chain_writes == 0
    assert "historical" in capsys.readouterr().err.lower()


def test_run_snapshot_hash_inputs_include_symbol_and_options_source_without_scaffolding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefs = Preferences()
    prefs.reporting.save_html = False
    prefs.save(tmp_path / "config" / "preferences.toml")
    strategy_dir = tmp_path / "strategies" / "2020-01"
    strategy_dir.mkdir(parents=True)
    (strategy_dir / "alpha.py").write_text(
        "from lambdaclass.strategies.base import Strategy, StrategyDecision\n"
        "\n"
        "class StrategyImpl(Strategy):\n"
        '    name = "alpha"\n'
        "    params = {}\n"
        "\n"
        "    def on_bar(self, context):\n"
        '        return StrategyDecision(action="hold", quantity=0)\n',
        encoding="utf-8",
    )
    store = DuckDBStore(tmp_path / "data")
    store.write_bars(
        "QQQ",
        pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.0, 101.0],
                "volume": [1000, 1100],
            }
        ),
    )
    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "datetime", _FixedDatetime)

    result = runner.invoke(cli.app, ["run", "alpha", "--symbol", "qqq", "--options-source", "yfinance"])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "strategies" / "2031-02").exists()
    snapshots = list((tmp_path / "runs" / "2031-02" / "alpha").glob("*/config.snapshot.toml"))
    assert len(snapshots) == 1
    snapshot = tomllib.loads(snapshots[0].read_text(encoding="utf-8"))
    assert snapshot["cli_overrides"]["symbol"] == "QQQ"
    assert snapshot["cli_overrides"]["options_source"] == "yfinance"
    assert "start" not in snapshot["cli_overrides"]
    assert "end" not in snapshot["cli_overrides"]


def test_fetch_earnings_help_has_force_option() -> None:
    result = runner.invoke(cli.app, ["fetch-earnings", "--help"])

    assert result.exit_code == 0
    assert "--force" in result.output


def test_init_does_not_create_unused_cache_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)

    cli.init_project()

    assert not (tmp_path / "data" / "cache").exists()
