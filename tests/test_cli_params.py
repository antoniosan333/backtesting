from __future__ import annotations

import tomllib
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import lambdaclass.cli as cli

runner = CliRunner()

STRATEGY_SOURCE = """
from lambdaclass.strategies.base import Strategy, StrategyDecision


class StrategyImpl(Strategy):
    name = "tunable"
    params = {"units": 1, "enabled": True}

    def on_bar(self, context):
        if self.params["enabled"] and context.position == 0:
            return StrategyDecision(action="buy", quantity=self.params["units"])
        return StrategyDecision(action="hold")
"""


@pytest.fixture
def project(make_project: Callable[..., Path]) -> Path:
    return make_project("tunable", STRATEGY_SOURCE, [100.0, 101.0])


def _single_run_dir(root: Path) -> Path:
    run_dirs = [path.parent for path in (root / "runs").glob("*/tunable/*/metrics.json")]
    assert len(run_dirs) == 1
    return run_dirs[0]


def test_run_applies_param_override_to_backtest_and_snapshot(project: Path) -> None:
    result = runner.invoke(cli.app, ["run", "tunable", "--param", "units=3"])

    assert result.exit_code == 0, result.output
    run_dir = _single_run_dir(project)
    snapshot = tomllib.loads((run_dir / "config.snapshot.toml").read_text(encoding="utf-8"))
    assert snapshot["strategy_params"] == {"units": "3", "enabled": "True"}
    trades = pd.read_csv(run_dir / "trades.csv")
    assert trades["quantity"].tolist() == [3]
    assert {"metrics.json", "run.log", "equity.parquet"} <= {path.name for path in run_dir.iterdir()}


def test_run_with_different_params_writes_distinct_runs(project: Path) -> None:
    first = runner.invoke(cli.app, ["run", "tunable"])
    second = runner.invoke(cli.app, ["run", "tunable", "--param", "enabled=false"])

    assert first.exit_code == 0 and second.exit_code == 0
    hashes = {
        path.parent.name.rsplit("-", 1)[-1] for path in (project / "runs").glob("*/tunable/*/metrics.json")
    }
    assert len(hashes) == 2


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ("unitz=3", "Unknown param(s) unitz"),
        ("units=2.5", "units expects an integer"),
        ("units", "expects key=value"),
    ],
)
def test_run_rejects_invalid_param(project: Path, argument: str, message: str) -> None:
    result = runner.invoke(cli.app, ["run", "tunable", "--param", argument])

    assert result.exit_code == 2
    assert message in result.output
    assert not (project / "runs").exists()
