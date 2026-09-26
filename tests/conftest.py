from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

import lambdaclass.cli as cli
from lambdaclass.config import Preferences
from lambdaclass.storage.duckdb_store import DuckDBStore

ProjectFactory = Callable[..., Path]


@pytest.fixture
def make_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectFactory:
    """Build a throwaway project (prefs, one strategy file, SPY bars) and point the CLI at it."""

    def build(
        strategy_name: str,
        source: str,
        closes: list[float],
        *,
        save_html: bool = False,
    ) -> Path:
        prefs = Preferences()
        prefs.reporting.save_html = save_html
        prefs.risk.max_position_pct = 1.0
        prefs.defaults.slippage_bps = 0.0
        prefs.save(tmp_path / "config" / "preferences.toml")
        strategy_dir = tmp_path / "strategies" / "2024-01"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / f"{strategy_name}.py").write_text(source, encoding="utf-8")
        dates = pd.bdate_range("2024-01-02", periods=len(closes)).strftime("%Y-%m-%d")
        DuckDBStore(tmp_path / "data").write_bars(
            "SPY",
            pd.DataFrame(
                {
                    "date": dates,
                    "open": closes,
                    "high": closes,
                    "low": closes,
                    "close": closes,
                    "volume": [1_000] * len(closes),
                    "dividends": [0.0] * len(closes),
                }
            ),
        )
        monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
        return tmp_path

    return build
