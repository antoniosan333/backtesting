from pathlib import Path

from lambdaclass.config import Preferences
from lambdaclass.storage.duckdb_store import DuckDBStore
from lambdaclass.strategies.base import Strategy


def test_removed_configuration_and_storage_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    store = DuckDBStore(tmp_path / "data")

    assert not hasattr(Preferences().defaults, "timezone")
    assert not hasattr(store, "cache_dir")
    assert not hasattr(Strategy, "on_chain")
    assert "pydantic-settings" not in (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "TIMEZONE" not in (root / ".env.example").read_text(encoding="utf-8")


def test_dashboard_offers_report_download_without_server_browser_open() -> None:
    root = Path(__file__).parents[1]
    source = (root / "src/lambdaclass/reporting/dashboard/app.py").read_text(encoding="utf-8")

    assert "webbrowser" not in source
    assert '"Download report.html"' in source
    assert "validate_symbol(symbol_input)" in source
    assert "st.sidebar.error(str(exc))" in source
    assert "st.stop()" in source
