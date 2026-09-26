from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

import lambdaclass.cli as cli
import lambdaclass.reporting.dashboard.app as dashboard_app
from lambdaclass.reporting.dashboard import charts, loader

runner = CliRunner()

STRATEGY_SOURCE = """
from lambdaclass.strategies.base import Strategy, StrategyDecision


class StrategyImpl(Strategy):
    name = "grid"
    params = {"units": 1, "entry_bar": 0, "tag": "a"}

    def __init__(self):
        self._bar = 0

    def on_bar(self, context):
        bar = self._bar
        self._bar += 1
        if bar == self.params["entry_bar"] and context.position == 0:
            return StrategyDecision(action="buy", quantity=self.params["units"])
        return StrategyDecision(action="hold")
"""


@pytest.fixture
def swept(make_project: Callable[..., Path]) -> Path:
    root = make_project("grid", STRATEGY_SOURCE, [100.0, 102.0, 101.0, 105.0, 107.0])
    result = runner.invoke(
        cli.app,
        ["sweep", "grid", "--grid", "units=1,2", "--grid", "entry_bar=0,1,2", "--grid", "tag=a,b"],
    )
    assert result.exit_code == 0, result.output
    return root


def test_list_and_load_sweep(swept: Path) -> None:
    sweep_dirs = loader.list_sweeps(swept / "runs")

    assert len(sweep_dirs) == 1
    bundle = loader.load_sweep(sweep_dirs[0])
    assert bundle.strategy == "grid"
    assert bundle.manifest["combinations"] == 12
    assert loader.sweep_param_names(bundle.results) == ["units", "entry_bar", "tag"]
    metrics = loader.sweep_metric_names(bundle.results)
    assert "sharpe" in metrics and "total_return" in metrics
    assert "combo" not in metrics


def test_sweep_dirs_are_not_listed_as_runs(swept: Path) -> None:
    runs = loader.list_runs(swept / "runs")

    assert len(runs) == 12
    assert all(run.parent.name == "grid" for run in runs)


def test_sweep_pivot_holds_other_params_fixed(swept: Path) -> None:
    results = loader.load_sweep(loader.list_sweeps(swept / "runs")[0]).results

    pivot = loader.sweep_pivot(results, x="units", y="entry_bar", metric="total_return", fixed={"tag": "a"})

    assert pivot.shape == (3, 2)
    assert list(pivot.columns) == [1, 2]
    assert list(pivot.index) == [0, 1, 2]
    # Doubling units doubles the stock P&L on the same entry bar.
    assert pivot.loc[0, 2] == pytest.approx(2 * pivot.loc[0, 1])


def test_sweep_pivot_single_axis_and_empty_cases() -> None:
    results = pd.DataFrame(
        {
            "combo": [0, 1, 2],
            "param_units": [1, 2, 2],
            "sharpe": [0.1, 0.4, 0.6],
            "status": ["ok", "ok", "error"],
        }
    )

    single = loader.sweep_pivot(results, x="units", y=None, metric="sharpe")
    assert single.loc["sharpe"].tolist() == [0.1, 0.4]
    assert loader.sweep_pivot(results, x="units", y=None, metric="missing").empty
    assert loader.sweep_pivot(pd.DataFrame(), x="units", y=None, metric="sharpe").empty


def test_sweep_heatmap_figure() -> None:
    pivot = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=[0, 1], columns=[5.0, 10.0])

    fig = charts.sweep_heatmap(pivot, metric="sharpe", x_label="width", y_label="lots")

    assert fig.data[0].type == "heatmap"
    assert list(fig.data[0].x) == ["5.0", "10.0"]
    assert "sharpe by width × lots" in fig.layout.title.text
    assert len(charts.sweep_heatmap(pd.DataFrame(), metric="m", x_label="x", y_label=None).data) == 0


def test_dashboard_renders_sweeps_tab(swept: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(swept)
    app = AppTest.from_file(dashboard_app.__file__, default_timeout=60)

    app.run()

    assert not app.exception
    tab_labels = [tab.label for tab in app.tabs]
    assert "Sweeps" in tab_labels
    sweep_select = next(box for box in app.selectbox if box.label == "Sweep")
    assert "12 combos" in sweep_select.options[0]
    assert any(frame.value.shape[0] == 12 for frame in app.dataframe)
    assert [box.label for box in app.selectbox if box.label.startswith("Hold ")] == ["Hold tag at"]
