"""Read-only loaders for backtest run artifacts and market data.

All functions are pure and accept absolute filesystem paths so they can be
unit-tested without Streamlit. ``app.py`` wraps the public entry points with
``@st.cache_data`` keyed by ``(path, mtime_ns)`` for snappy reruns.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from lambdaclass.data_adapters.optionsdx_chain_loader import (
    CHAIN_COLUMNS,
    load_normalized_optionsdx_chain,
)
from lambdaclass.reporting import metrics as reporting_metrics
from lambdaclass.runs.layout import PARAM_PREFIX, SWEEP_MANIFEST_FILE, SWEEP_RESULTS_FILE, SWEEPS_DIRNAME
from lambdaclass.symbols import validate_symbol

ARTIFACT_NAMES = (
    "metrics.json",
    "equity.parquet",
    "trades.csv",
    "expected_moves.parquet",
    "config.snapshot.toml",
)


@dataclass(frozen=True)
class RunBundle:
    run_dir: Path
    strategy: str
    run_id: str
    metrics: dict[str, float]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    option_trades: pd.DataFrame
    expected_moves: pd.DataFrame
    snapshot: dict[str, Any]
    symbol: str
    start: str
    end: str
    config_hash: str


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "metrics.json").is_file()


def _strategy_for_run(run_dir: Path) -> str:
    return run_dir.parent.name


def list_runs(runs_root: Path, limit: int | None = None) -> list[Path]:
    """Return run directories under ``runs/<YYYY-MM>/<strategy>/<run_id>/`` sorted newest first."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    candidates = [p for p in runs_root.glob("*/*/*") if _is_run_dir(p)]
    candidates.sort(key=lambda p: (p / "metrics.json").stat().st_mtime, reverse=True)
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


def list_strategies(runs_root: Path) -> list[str]:
    """Distinct strategy names with at least one run, alphabetical."""
    return sorted({_strategy_for_run(r) for r in list_runs(runs_root)})


def runs_for_strategy(runs_root: Path, strategy: str) -> list[Path]:
    return [r for r in list_runs(runs_root) if _strategy_for_run(r) == strategy]


def run_dir_mtime_ns(run_dir: Path) -> int:
    """Max mtime over the four core artifacts. Used for cache invalidation."""
    mtimes: list[int] = []
    for name in ARTIFACT_NAMES:
        p = run_dir / name
        if p.exists():
            mtimes.append(p.stat().st_mtime_ns)
    return max(mtimes) if mtimes else 0


def _load_snapshot(snapshot_path: Path) -> dict[str, Any]:
    if not snapshot_path.is_file():
        return {}
    return tomllib.loads(snapshot_path.read_text(encoding="utf-8"))


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "action", "quantity", "price", "cash_after"])


def _empty_option_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "action",
            "contract_symbol",
            "side",
            "strike",
            "expiry",
            "quantity",
            "mid_price",
            "premium",
            "commission",
            "cash_after",
        ]
    )


def _optional_snapshot_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "none" else text


def load_run(run_dir: Path) -> RunBundle:
    run_dir = Path(run_dir).resolve()
    if not _is_run_dir(run_dir):
        raise FileNotFoundError(f"Not a run directory: {run_dir}")

    metrics_path = run_dir / "metrics.json"
    equity_path = run_dir / "equity.parquet"
    trades_path = run_dir / "trades.csv"
    option_trades_path = run_dir / "option_trades.csv"
    expected_moves_path = run_dir / "expected_moves.parquet"
    snapshot_path = run_dir / "config.snapshot.toml"

    metrics: dict[str, float] = json.loads(metrics_path.read_text(encoding="utf-8"))
    equity_curve = pd.read_parquet(equity_path) if equity_path.is_file() else pd.DataFrame()
    if trades_path.is_file():
        trades = pd.read_csv(trades_path)
        if trades.empty:
            trades = _empty_trades()
    else:
        trades = _empty_trades()
    if option_trades_path.is_file():
        option_trades = pd.read_csv(option_trades_path)
        if option_trades.empty:
            option_trades = _empty_option_trades()
    else:
        option_trades = _empty_option_trades()
    expected_moves = (
        pd.read_parquet(expected_moves_path)
        if expected_moves_path.is_file()
        else pd.DataFrame()
    )
    snapshot = _load_snapshot(snapshot_path)

    cli_overrides = snapshot.get("cli_overrides", {}) if isinstance(snapshot, dict) else {}
    snap_meta = snapshot.get("meta", {}) if isinstance(snapshot, dict) else {}
    snapshot_start = _optional_snapshot_value(cli_overrides.get("start"))
    snapshot_end = _optional_snapshot_value(cli_overrides.get("end"))

    if not equity_curve.empty:
        equity_dates = equity_curve["date"].astype(str)
        start = snapshot_start or str(equity_dates.iloc[0])
        end = snapshot_end or str(equity_dates.iloc[-1])
    else:
        start = snapshot_start
        end = snapshot_end

    symbol = _optional_snapshot_value(cli_overrides.get("symbol"))
    log_path = run_dir / "run.log"
    if not symbol and log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("symbol="):
                symbol = line.split("=", 1)[1].strip()
                break

    return RunBundle(
        run_dir=run_dir,
        strategy=_strategy_for_run(run_dir),
        run_id=run_dir.name,
        metrics=metrics,
        equity_curve=equity_curve,
        trades=trades,
        option_trades=option_trades,
        expected_moves=expected_moves,
        snapshot=snapshot,
        symbol=symbol.upper(),
        start=start,
        end=end,
        config_hash=str(snap_meta.get("config_hash", "")),
    )


def load_bars(data_dir: Path, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Read ``data/stocks/<SYMBOL>.parquet`` via DuckDB. Empty frame if missing."""
    symbol = validate_symbol(symbol)
    bars_path = Path(data_dir) / "stocks" / f"{symbol}.parquet"
    if not bars_path.is_file():
        return pd.DataFrame()
    query = "SELECT * FROM read_parquet(?)"
    params: list[str] = [str(bars_path)]
    clauses: list[str] = []
    if start:
        clauses.append("date >= ?")
        params.append(start)
    if end:
        clauses.append("date <= ?")
        params.append(end)
    if clauses:
        query = f"{query} WHERE {' AND '.join(clauses)}"
    query += " ORDER BY date"
    with duckdb.connect() as con:
        return con.execute(query, params).df()


def load_chain(normalized_root: Path, symbol: str, dates: list[str]) -> pd.DataFrame:
    """Filter normalized OptionsDX Parquet to ``dates`` (YYYY-MM-DD) for ``symbol``."""
    symbol = validate_symbol(symbol)
    if not dates:
        return pd.DataFrame(columns=CHAIN_COLUMNS)
    return load_normalized_optionsdx_chain(Path(normalized_root), symbol, dates)


def chain_expiries(chain: pd.DataFrame) -> list[str]:
    """Distinct expiry strings from a loaded chain, sorted."""
    if chain is None or chain.empty or "expiry" not in chain.columns:
        return []
    return sorted({str(x).strip() for x in chain["expiry"].dropna().unique()})


def chain_spot_estimate(bars: pd.DataFrame, asof: str) -> float:
    """Last ``close`` on or before ``asof`` (YYYY-MM-DD)."""
    if bars is None or bars.empty or "close" not in bars.columns:
        return 0.0
    df = bars[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])
    cut = str(asof)[:10]
    sub = df[df["date"] <= cut]
    if sub.empty:
        return float(df["close"].iloc[-1])
    return float(sub["close"].iloc[-1])


def load_earnings(data_dir: Path, symbol: str) -> pd.DataFrame:
    """Read ``data/earnings/<SYMBOL>.parquet``; empty frame if missing."""
    symbol = validate_symbol(symbol)
    path = Path(data_dir) / "earnings" / f"{symbol}.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=["symbol", "earnings_date", "timing", "source", "fetched_at"])
    return pd.read_parquet(path)


def load_events(run_dir: Path) -> pd.DataFrame:
    """Read ``events.parquet`` from a run directory if present."""
    path = Path(run_dir) / "events.parquet"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def aggregate_metrics(runs_root: Path, run_dirs: list[Path] | None = None) -> pd.DataFrame:
    """One-row-per-run table for cross-run metric comparison.

    Uses DuckDB ``read_json_auto`` over each run's ``metrics.json`` and stitches in
    identifiers from the directory layout.
    """
    if run_dirs is None:
        run_dirs = list_runs(runs_root)
    if not run_dirs:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    with duckdb.connect() as con:
        for run_dir in run_dirs:
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.is_file():
                continue
            try:
                df = con.execute("SELECT * FROM read_json_auto(?)", [str(metrics_path)]).df()
            except duckdb.Error:
                continue
            if df.empty:
                continue
            row: dict[str, Any] = {col: df.iloc[0][col] for col in df.columns}
            row["strategy"] = _strategy_for_run(run_dir)
            row["run_id"] = run_dir.name
            row["run_dir"] = str(run_dir)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    columns = (
        ["strategy", "run_id"]
        + sorted({k for r in rows for k in r} - {"strategy", "run_id", "run_dir"})
        + ["run_dir"]
    )
    return pd.DataFrame(rows)[columns]


@dataclass(frozen=True)
class SweepBundle:
    sweep_dir: Path
    sweep_id: str
    strategy: str
    manifest: dict[str, Any]
    results: pd.DataFrame


def list_sweeps(runs_root: Path) -> list[Path]:
    """Sweep directories under ``runs/<YYYY-MM>/<strategy>/_sweeps/<sweep_id>/``, newest first."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    results = runs_root.glob(f"*/*/{SWEEPS_DIRNAME}/*/{SWEEP_RESULTS_FILE}")
    return [path.parent for path in sorted(results, key=lambda path: path.stat().st_mtime, reverse=True)]


def load_sweep(sweep_dir: Path) -> SweepBundle:
    sweep_dir = Path(sweep_dir)
    manifest_path = sweep_dir / SWEEP_MANIFEST_FILE
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    results_path = sweep_dir / SWEEP_RESULTS_FILE
    results = pd.read_parquet(results_path) if results_path.is_file() else pd.DataFrame()
    return SweepBundle(
        sweep_dir=sweep_dir,
        sweep_id=sweep_dir.name,
        strategy=str(manifest.get("strategy") or sweep_dir.parent.parent.name),
        manifest=manifest,
        results=results,
    )


def sweep_param_names(results: pd.DataFrame) -> list[str]:
    """Swept param names (without the column prefix), in grid order."""
    return [
        column.removeprefix(PARAM_PREFIX) for column in results.columns if column.startswith(PARAM_PREFIX)
    ]


def sweep_metric_names(results: pd.DataFrame) -> list[str]:
    """Numeric result columns other than params and the combination index."""
    excluded = {"combo"}
    return [
        column
        for column in results.columns
        if column not in excluded
        and not column.startswith(PARAM_PREFIX)
        and pd.api.types.is_numeric_dtype(results[column])
        and not pd.api.types.is_bool_dtype(results[column])
    ]


def sweep_pivot(
    results: pd.DataFrame,
    *,
    x: str,
    y: str | None,
    metric: str,
    fixed: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Metric table over one or two swept params, holding the other params at ``fixed``.

    With ``y=None`` the result has a single row named after ``metric``.
    """
    if results.empty or "status" not in results.columns or metric not in results.columns:
        return pd.DataFrame()
    frame = results[results["status"] == "ok"]
    for name, value in (fixed or {}).items():
        frame = frame[frame[f"{PARAM_PREFIX}{name}"] == value]
    if frame.empty:
        return pd.DataFrame()
    x_column = f"{PARAM_PREFIX}{x}"
    if y is None:
        series = frame.groupby(x_column, sort=True)[metric].mean()
        return series.to_frame(metric).T
    return frame.pivot_table(index=f"{PARAM_PREFIX}{y}", columns=x_column, values=metric, aggfunc="mean")


def num_trades(trades: pd.DataFrame) -> int:
    """Compatibility wrapper for the shared reporting metric."""
    return reporting_metrics.num_trades(trades)


def avg_holding_days(trades: pd.DataFrame) -> float:
    """Compatibility wrapper for the shared reporting metric."""
    return reporting_metrics.avg_holding_days(trades)


def win_rate_per_trade(trades: pd.DataFrame) -> float:
    """Compatibility wrapper for the shared reporting metric."""
    return reporting_metrics.win_rate_per_trade(trades)
