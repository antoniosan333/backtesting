from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import lambdaclass.cli as cli
from lambdaclass.runs.sweep import SweepError, expand_grid, parse_grid, rank_results
from lambdaclass.strategies.params import ParamError

runner = CliRunner()

STRATEGY_SOURCE = """
from lambdaclass.strategies.base import Strategy, StrategyDecision


class StrategyImpl(Strategy):
    name = "grid"
    params = {"units": 1, "entry_bar": 0, "explode": False}

    def __init__(self):
        self._bar = 0

    def on_bar(self, context):
        if self.params["explode"]:
            raise RuntimeError("boom")
        bar = self._bar
        self._bar += 1
        if bar == self.params["entry_bar"] and context.position == 0:
            return StrategyDecision(action="buy", quantity=self.params["units"])
        return StrategyDecision(action="hold")
"""


def test_parse_grid_lists_and_inclusive_ranges() -> None:
    assert parse_grid(["units=1,2, 3", "width=5:10:2.5", "mode=a:b"]) == {
        "units": ["1", "2", "3"],
        "width": ["5", "7.5", "10"],
        "mode": ["a:b"],
    }


def test_parse_grid_range_has_no_float_drift() -> None:
    assert parse_grid(["x=0.1:0.3:0.1"]) == {"x": ["0.1", "0.2", "0.3"]}


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([], "at least one --grid"),
        (["x=1,,2"], "empty value"),
        (["x=5:1:1"], "stop below start"),
        (["x=1:5:0"], "positive step"),
        (["x=0:100000:1"], "expands to"),
        (["x=1", "x=2"], "more than once"),
    ],
)
def test_parse_grid_rejects_invalid(items: list[str], message: str) -> None:
    with pytest.raises((SweepError, ParamError), match=message):
        parse_grid(items)


def test_expand_grid_coerces_dedupes_and_orders() -> None:
    combos = expand_grid(
        {"units": ["1", "1.0", "2"], "flag": ["true"]}, {"units": 1, "flag": False}, max_combos=10
    )

    assert combos == [{"units": 1, "flag": True}, {"units": 2, "flag": True}]


def test_expand_grid_enforces_cap_and_known_keys() -> None:
    with pytest.raises(SweepError, match="4 combinations"):
        expand_grid({"a": ["1", "2"], "b": ["1", "2"]}, {"a": 1, "b": 1}, max_combos=3)
    with pytest.raises(ParamError, match="Unknown param"):
        expand_grid({"c": ["1"]}, {"a": 1}, max_combos=3)


def test_rank_results_skips_failures_and_honors_direction() -> None:
    results = pd.DataFrame(
        {
            "combo": [0, 1, 2],
            "sharpe": [0.5, float("nan"), 1.5],
            "status": ["ok", "error", "ok"],
        }
    )

    assert rank_results(results, "sharpe")["combo"].tolist() == [2, 0]
    assert rank_results(results, "sharpe", minimize=True)["combo"].tolist() == [0, 2]
    assert rank_results(results, "missing").empty


@pytest.fixture
def project(make_project: Callable[..., Path]) -> Path:
    return make_project("grid", STRATEGY_SOURCE, [100.0, 102.0, 101.0, 105.0, 107.0], save_html=True)


def _sweep_dir(root: Path) -> Path:
    sweep_dirs = list(root.glob("runs/*/grid/_sweeps/*"))
    assert len(sweep_dirs) == 1
    return sweep_dirs[0]


def test_sweep_writes_one_run_per_combination_and_summary(project: Path) -> None:
    result = runner.invoke(cli.app, ["sweep", "grid", "--grid", "units=1,2", "--grid", "entry_bar=0:1:1"])

    assert result.exit_code == 0, result.output
    run_dirs = sorted(path.parent for path in project.glob("runs/*/grid/*/metrics.json"))
    assert len(run_dirs) == 4
    assert not any((run_dir / "report.html").exists() for run_dir in run_dirs)
    sweep_dir = _sweep_dir(project)
    summary = pd.read_parquet(sweep_dir / "sweep.parquet")
    assert summary[["param_units", "param_entry_bar"]].values.tolist() == [[1, 0], [1, 1], [2, 0], [2, 1]]
    assert set(summary["status"]) == {"ok"}
    assert sorted(Path(path) for path in summary["run_dir"]) == run_dirs
    assert (summary["total_return"] > 0).all()
    manifest = json.loads((sweep_dir / "sweep.json").read_text(encoding="utf-8"))
    assert manifest["combinations"] == 4
    assert manifest["completed"] == 4
    assert manifest["grid"] == {"units": ["1", "2"], "entry_bar": ["0", "1"]}
    log = (run_dirs[0] / "run.log").read_text(encoding="utf-8")
    assert f"sweep_id={manifest['sweep_id']}" in log
    assert "Top 4 by sharpe" in result.output
    assert "out of sample" in result.output


def test_sweep_combo_matches_equivalent_single_run(project: Path) -> None:
    sweep = runner.invoke(cli.app, ["sweep", "grid", "--grid", "units=2", "--param", "entry_bar=1"])
    single = runner.invoke(cli.app, ["run", "grid", "--param", "units=2", "--param", "entry_bar=1"])

    assert sweep.exit_code == 0, sweep.output
    assert single.exit_code == 0, single.output
    hashes = {path.parent.name.rsplit("-", 1)[-1] for path in project.glob("runs/*/grid/*/metrics.json")}
    assert len(hashes) == 1


def test_sweep_parallel_matches_sequential(project: Path) -> None:
    grid = ["--grid", "units=1,2,3", "--grid", "entry_bar=0,2"]
    sequential = runner.invoke(cli.app, ["sweep", "grid", *grid, "--jobs", "1"])
    assert sequential.exit_code == 0, sequential.output
    first = pd.read_parquet(_sweep_dir(project) / "sweep.parquet")
    for path in project.glob("runs/*/grid/_sweeps/*"):
        for child in path.iterdir():
            child.unlink()
        path.rmdir()

    parallel = runner.invoke(cli.app, ["sweep", "grid", *grid, "--jobs", "2"])

    assert parallel.exit_code == 0, parallel.output
    second = pd.read_parquet(_sweep_dir(project) / "sweep.parquet")
    metric_columns = ["param_units", "param_entry_bar", "total_return", "sharpe", "max_drawdown"]
    pd.testing.assert_frame_equal(first[metric_columns], second[metric_columns])


def test_sweep_records_failed_combination_and_continues(project: Path) -> None:
    result = runner.invoke(cli.app, ["sweep", "grid", "--grid", "explode=false,true"])

    assert result.exit_code == 0, result.output
    summary = pd.read_parquet(_sweep_dir(project) / "sweep.parquet")
    assert summary["status"].tolist() == ["ok", "error"]
    assert "RuntimeError: boom" in summary.loc[1, "error"]
    assert "1 combination(s) failed" in result.output


def test_sweep_fail_fast_stops_and_exits_non_zero(project: Path) -> None:
    result = runner.invoke(cli.app, ["sweep", "grid", "--grid", "explode=true,false", "--fail-fast"])

    assert result.exit_code == 1
    summary = pd.read_parquet(_sweep_dir(project) / "sweep.parquet")
    assert summary["status"].tolist() == ["error"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--grid", "units=1,2", "--param", "units=3"], "given in both --grid and --param"),
        (["--grid", "unitz=1"], "Unknown param(s) unitz"),
        (["--grid", "units=1,2,3", "--max-combos", "2"], "3 combinations"),
    ],
)
def test_sweep_rejects_invalid_grid_before_running(project: Path, arguments: list[str], message: str) -> None:
    result = runner.invoke(cli.app, ["sweep", "grid", *arguments])

    assert result.exit_code == 2
    assert message in result.output
    assert not (project / "runs").exists()
