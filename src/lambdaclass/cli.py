from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import typer

from lambdaclass.cli_universe import universe_app
from lambdaclass.config import (
    DEFAULT_PREFERENCES,
    Preferences,
    compute_config_hash,
    load_project_preferences,
    preferences_path,
)
from lambdaclass.data_adapters.optionsdx_normalize import NormalizeOptions, run_normalize
from lambdaclass.data_adapters.yfinance_adapter import YFinanceAdapter
from lambdaclass.data_adapters.dolthub_chain import fetch_dolthub_chain, fetch_dolthub_vol_history
from lambdaclass.data_adapters.edgar_earnings import fetch_earnings_frame
from lambdaclass.earnings.calendar import normalize_earnings_frame
from lambdaclass.reporting import dashboard as reporting_dashboard
from lambdaclass.retry import with_retry
from lambdaclass.runs.layout import PARAM_PREFIX, SWEEPS_DIRNAME
from lambdaclass.runs.runner import RunInputs, execute_run, load_run_inputs
from lambdaclass.runs.sweep import (
    SweepContext,
    SweepError,
    SweepSpec,
    expand_grid,
    parse_grid,
    rank_results,
    run_sweep,
    write_sweep_summary,
)
from lambdaclass.state import load_json, save_json
from lambdaclass.storage.duckdb_store import DuckDBStore
from lambdaclass.storage.optionsdx_reader import read_atm_slice
from lambdaclass.volatility.earnings_cycle import align_events, cycle_summary, event_table, normalize
from lambdaclass.volatility.series import build_vol_series, build_vol_series_from_history, merge_vol_cache
from lambdaclass.volatility.universe import PILOT_SYMBOLS, load_stock_universe
from lambdaclass.strategies.base import Strategy
from lambdaclass.strategies.loader import StrategyLoadError, load_strategy_class
from lambdaclass.strategies.params import ParamError, apply_params, parse_assignments, resolve_params
from lambdaclass.strategies.scaffolder import scaffold_strategy
from lambdaclass.symbols import validate_symbol

app = typer.Typer(help="LambdaClass stock + options backtesting CLI")
STRATEGY_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
FetchResult = TypeVar("FetchResult")


def _repo_root() -> Path:
    return Path.cwd()


def _warn_chain_coverage(options_chain: pd.DataFrame, bar_dates: pd.Series, source: str) -> None:
    """Warn when the loaded chain cannot serve the backtest window.

    A `yfinance` chain is a single snapshot stamped with the fetch date, so it
    typically overlaps at most one bar; the engine would then reject every
    option order rather than fill it.
    """
    if options_chain.empty:
        typer.secho(
            f"Warning: no options chain rows loaded from '{source}'. Any option orders will be rejected.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        return
    covered = set(options_chain["asof"].astype(str)) & set(bar_dates.astype(str))
    if not covered:
        typer.secho(
            f"Warning: the '{source}' chain has no as-of date matching any bar in this window "
            f"({len(options_chain)} rows loaded). Any option orders will be rejected.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    elif len(covered) < len(bar_dates):
        typer.secho(
            f"Warning: options chain covers {len(covered)} of {len(bar_dates)} bars.",
            fg=typer.colors.YELLOW,
            err=True,
        )


def _preferences_path(root: Path) -> Path:
    return preferences_path(root)


def _load_preferences(root: Path) -> Preferences:
    return load_project_preferences(root)


def _get_adapter(name: str) -> YFinanceAdapter:
    if name == "yfinance":
        return YFinanceAdapter()
    raise typer.BadParameter(f"Unsupported data adapter: {name}")


def _git_short_sha(root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return output or "nogit"
    except Exception:
        return "nogit"


def _run_id(config_hash: str, root: Path) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{_git_short_sha(root)}-{config_hash}"


def _validate_strategy_name(strategy_name: str) -> str:
    if not STRATEGY_NAME_PATTERN.match(strategy_name):
        raise typer.BadParameter(
            "Strategy name must start with a letter and contain only letters, numbers, and underscores."
        )
    return strategy_name


def _validate_symbol(symbol: str) -> str:
    try:
        return validate_symbol(symbol)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="symbol") from exc


def _find_strategy_file(strategies_dir: Path, strategy_name: str) -> Path:
    matches = sorted(strategies_dir.glob(f"*/{strategy_name}.py"))
    if not matches:
        raise typer.BadParameter(f"Strategy `{strategy_name}` not found under {strategies_dir}")
    resolved = matches[-1].resolve()
    root = strategies_dir.resolve()
    if root not in resolved.parents:
        raise typer.BadParameter("Strategy path escapes strategies directory.")
    return resolved


def _load_strategy_class(path: Path) -> type[Strategy]:
    try:
        return load_strategy_class(path)
    except StrategyLoadError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _instantiate_strategy(strategy_cls: type[Strategy]) -> Strategy:
    strategy = strategy_cls()
    if not isinstance(strategy, Strategy):
        raise typer.BadParameter("StrategyImpl must inherit lambdaclass.strategies.base.Strategy")
    return strategy


def _load_strategy(path: Path) -> Strategy:
    return _instantiate_strategy(_load_strategy_class(path))


def _resolve_strategy_class(root: Path, prefs: Preferences, strategy_name: str) -> type[Strategy]:
    strategies_dir = root / prefs.paths.strategies_dir
    validated_strategy = _validate_strategy_name(strategy_name)
    return _load_strategy_class(_find_strategy_file(strategies_dir, validated_strategy))


def _load_inputs(
    root: Path,
    prefs: Preferences,
    *,
    symbol: str,
    start: str | None,
    end: str | None,
    options_source: str | None,
) -> RunInputs:
    optionsdx_root = Path(prefs.optionsdx.output_dir)
    if not optionsdx_root.is_absolute():
        optionsdx_root = (root / optionsdx_root).resolve()
    try:
        inputs = load_run_inputs(
            DuckDBStore(root / prefs.paths.data_dir),
            symbol=symbol,
            start=start,
            end=end,
            options_source=options_source or prefs.defaults.options_chain_source,
            optionsdx_root=optionsdx_root,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if inputs.dividends_backfilled:
        typer.secho(
            f"Warning: {symbol} bars predate dividend capture; dividends are treated as zero. "
            f"Re-run `lambdaclass fetch {symbol}` to include them.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    _warn_chain_coverage(inputs.options_chain, inputs.bars["date"], inputs.options_source)
    return inputs


def _run_overrides(inputs: RunInputs, *, start: str | None, end: str | None) -> dict[str, str | None]:
    return {
        "start": start,
        "end": end,
        "symbol": inputs.symbol,
        "options_source": inputs.options_source,
    }


def _fetch_with_retry(
    fetch_fn: Callable[[], FetchResult],
    retries: int = 3,
    delay_seconds: float = 1.0,
) -> FetchResult:
    try:
        return with_retry(fetch_fn, retries=retries, delay_seconds=delay_seconds)
    except Exception as exc:  # pragma: no cover - defensive runtime handling
        raise typer.BadParameter(f"Data fetch failed after {retries} attempts: {exc}") from exc


@app.command("init")
def init_project(force: bool = typer.Option(False, help="Overwrite existing preferences file")) -> None:
    root = _repo_root()
    config_path = _preferences_path(root)
    data_dir = root / "data"
    strategies_dir = root / "strategies"
    runs_dir = root / "runs"
    state_dir = root / "state"
    for directory in [
        root / "config",
        data_dir / "stocks",
        data_dir / "options",
        data_dir / "earnings",
        strategies_dir,
        runs_dir,
        state_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    if force or not config_path.exists():
        DEFAULT_PREFERENCES.save(config_path)
        typer.echo(f"Wrote preferences file: {config_path}")
    else:
        typer.echo(f"Preferences file already exists: {config_path}")
    typer.echo("Project folders initialized.")


@app.command("fetch")
def fetch_data(
    symbol: str,
    start: str = typer.Option(..., help="Start date in YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End date in YYYY-MM-DD (default: today)"),
) -> None:
    root = _repo_root()
    symbol = _validate_symbol(symbol)
    prefs = _load_preferences(root)
    store = DuckDBStore(root / prefs.paths.data_dir)
    adapter = _get_adapter(prefs.defaults.data_adapter)
    today = date.today()
    start_dt = date.fromisoformat(start)
    end_dt = date.fromisoformat(end) if end is not None else today
    bars = _fetch_with_retry(lambda: adapter.get_stock_bars(symbol, start_dt, end_dt))
    bars_path = store.write_bars(symbol, bars)
    chain = pd.DataFrame()
    chain_path: Path | None = None
    if end_dt < today:
        typer.secho(
            "Warning: skipping options chain fetch for a historical end date; "
            "yfinance only provides the current chain.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    else:
        chain = _fetch_with_retry(lambda: adapter.get_option_chain(symbol, today))
        chain_path = store.write_chain(symbol, chain)
    markers_path = root / "state" / "fetch_markers.json"
    markers = load_json(markers_path)
    markers[symbol.upper()] = {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "bars_path": str(bars_path),
        "options_path": str(chain_path),
        "updated_at": datetime.now(tz=UTC).isoformat(),
    }
    save_json(markers_path, markers)
    typer.echo(
        f"Fetched {symbol.upper()} bars={len(bars)} options={len(chain)} "
        f"from {start_dt.isoformat()} to {end_dt.isoformat()}"
    )


@app.command("fetch-earnings")
def fetch_earnings(
    symbol: str,
    csv: str | None = typer.Option(None, help="Optional CSV with earnings_date [, timing] columns"),
    source: str = typer.Option("yfinance", help="yfinance or edgar"),
    force: bool = typer.Option(False, help="Overwrite existing rows for matching dates (via dedupe keep last)"),
) -> None:
    """Fetch or import earnings calendar into data/earnings/<SYMBOL>.parquet."""
    root = _repo_root()
    symbol = _validate_symbol(symbol)
    prefs = _load_preferences(root)
    store = DuckDBStore(root / prefs.paths.data_dir)
    if csv:
        path = Path(csv)
        if not path.is_file():
            raise typer.BadParameter(f"CSV not found: {path}")
        raw = pd.read_csv(path)
        frame = normalize_earnings_frame(raw, symbol=symbol, source="csv")
        source_name = "csv"
    elif source.strip().lower() == "edgar":
        frame = fetch_earnings_frame(symbol)
        source_name = "edgar"
    elif source.strip().lower() == "yfinance":
        adapter = _get_adapter(prefs.defaults.data_adapter)
        raw = _fetch_with_retry(lambda: adapter.get_earnings_dates(symbol))
        frame = normalize_earnings_frame(raw, symbol=symbol, source="yfinance")
        source_name = "yfinance"
    else:
        raise typer.BadParameter("source must be yfinance, edgar, or omitted when --csv is set")
    out_path = store.write_earnings(symbol, frame)
    markers_path = root / "state" / "earnings_fetch_markers.json"
    markers = load_json(markers_path)
    markers[symbol] = {
        "rows": int(len(frame)),
        "path": str(out_path),
        "source": source_name,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    save_json(markers_path, markers)
    typer.echo(f"Earnings calendar {symbol}: {len(frame)} rows → {out_path}")


@app.command("new-strategy")
def new_strategy(name: str) -> None:
    root = _repo_root()
    prefs = _load_preferences(root)
    validated_name = _validate_strategy_name(name)
    strategy_file = scaffold_strategy(root / prefs.paths.strategies_dir, validated_name)
    typer.echo(f"Strategy scaffold ready: {strategy_file}")


@app.command("run")
def run_strategy(
    strategy_name: str,
    symbol: str = typer.Option("SPY", help="Underlying symbol with fetched data"),
    start: str | None = typer.Option(None, help="Optional YYYY-MM-DD start override"),
    end: str | None = typer.Option(None, help="Optional YYYY-MM-DD end override"),
    options_source: str | None = typer.Option(
        None,
        help="Options chain: yfinance (data/options Parquet) or optionsdx (normalized under [optionsdx].output_dir). Default: [defaults].options_chain_source",
    ),
    fail_on_rejected_orders: bool = typer.Option(
        False,
        help="Exit non-zero if any stock or option order was rejected",
    ),
    param: list[str] | None = typer.Option(
        None,
        "--param",
        help="Override a strategy param as key=value (repeatable); typed from the strategy's default",
    ),
) -> None:
    root = _repo_root()
    symbol = _validate_symbol(symbol)
    prefs = _load_preferences(root)
    strategy_cls = _resolve_strategy_class(root, prefs, strategy_name)
    strategy = _instantiate_strategy(strategy_cls)
    try:
        apply_params(strategy, parse_assignments(param or []))
    except ParamError as exc:
        raise typer.BadParameter(str(exc), param_hint="--param") from exc
    inputs = _load_inputs(root, prefs, symbol=symbol, start=start, end=end, options_source=options_source)
    month = datetime.now(tz=UTC).strftime("%Y-%m")
    strategy_runs = root / prefs.paths.runs_dir / month / strategy.name
    summary = execute_run(
        strategy,
        inputs,
        prefs,
        cli_overrides=_run_overrides(inputs, start=start, end=end),
        run_dir_for=lambda config_hash: strategy_runs / _run_id(config_hash, root),
    )
    run_dir = summary.run_dir
    typer.echo(f"Run complete: {run_dir}")
    typer.echo(f"Metrics: {run_dir / 'metrics.json'}")
    typer.echo(f"Trades: {run_dir / 'trades.csv'}")
    typer.echo(f"Equity: {run_dir / 'equity.parquet'}")
    if not summary.events.empty:
        typer.echo(f"Events: {run_dir / 'events.parquet'}")
    rejected = summary.result.rejected_orders
    if not rejected.empty:
        by_reason = ", ".join(
            f"{reason}={count}" for reason, count in rejected["reason"].value_counts().items()
        )
        typer.secho(
            f"Warning: {len(rejected)} order(s) rejected ({by_reason}). "
            f"See {run_dir / 'rejected_orders.csv'}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        if fail_on_rejected_orders:
            raise typer.Exit(code=1)


@app.command("sweep")
def sweep_strategy(
    strategy_name: str,
    grid: list[str] | None = typer.Option(
        None,
        "--grid",
        help="Param values as key=a,b,c or key=start:stop:step (inclusive); repeatable",
    ),
    param: list[str] | None = typer.Option(
        None,
        "--param",
        help="Fixed param override for every combination, key=value; repeatable",
    ),
    symbol: str = typer.Option("SPY", help="Underlying symbol with fetched data"),
    start: str | None = typer.Option(None, help="Optional YYYY-MM-DD start override"),
    end: str | None = typer.Option(None, help="Optional YYYY-MM-DD end override"),
    options_source: str | None = typer.Option(
        None,
        help="Options chain: yfinance or optionsdx. Default: [defaults].options_chain_source",
    ),
    metric: str = typer.Option("sharpe", help="Metric used to rank combinations"),
    minimize: bool = typer.Option(False, help="Rank ascending (lower metric is better)"),
    top: int = typer.Option(5, min=0, help="Combinations to print after the sweep"),
    jobs: int = typer.Option(1, min=1, help="Worker processes (each loads the data once)"),
    max_combos: int = typer.Option(200, min=1, help="Refuse grids with more combinations than this"),
    html: bool = typer.Option(False, help="Write report.html for every combination"),
    fail_fast: bool = typer.Option(False, help="Stop at the first failing combination and exit non-zero"),
) -> None:
    """Run a strategy over every combination of --grid values."""
    root = _repo_root()
    symbol = _validate_symbol(symbol)
    prefs = _load_preferences(root)
    strategies_dir = root / prefs.paths.strategies_dir
    strategy_path = _find_strategy_file(strategies_dir, _validate_strategy_name(strategy_name))
    strategy_cls = _load_strategy_class(strategy_path)
    probe = _instantiate_strategy(strategy_cls)
    try:
        fixed_raw = parse_assignments(param or [])
        resolved = resolve_params(probe.params, fixed_raw)
        parsed_grid = parse_grid(grid or [])
        overlap = sorted(set(parsed_grid) & set(fixed_raw))
        if overlap:
            raise SweepError(f"{', '.join(overlap)} given in both --grid and --param")
        combos = expand_grid(parsed_grid, probe.params, max_combos=max_combos)
    except (ParamError, SweepError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    fixed_overrides = {key: resolved[key] for key in fixed_raw}

    inputs = _load_inputs(root, prefs, symbol=symbol, start=start, end=end, options_source=options_source)
    now = datetime.now(tz=UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    git_sha = _git_short_sha(root)
    strategy_runs = root / prefs.paths.runs_dir / now.strftime("%Y-%m") / probe.name
    manifest: dict[str, Any] = {
        "strategy": probe.name,
        "strategy_file": str(strategy_path.relative_to(root.resolve()))
        if root.resolve() in strategy_path.parents
        else strategy_path.name,
        "symbol": inputs.symbol,
        "start": start,
        "end": end,
        "options_source": inputs.options_source,
        "grid": parsed_grid,
        "fixed_params": fixed_overrides,
        "combinations": len(combos),
        "metric": metric,
        "minimize": minimize,
        "git_sha": git_sha,
        "created_at": now.isoformat(),
        "preferences": prefs.model_dump(mode="json"),
    }
    sweep_id = f"{stamp}-{git_sha}-{compute_config_hash(manifest | {'created_at': None})}"
    manifest["sweep_id"] = sweep_id
    optionsdx_root = Path(prefs.optionsdx.output_dir)
    if not optionsdx_root.is_absolute():
        optionsdx_root = (root / optionsdx_root).resolve()
    spec = SweepSpec(
        strategy_path=strategy_path,
        data_dir=root / prefs.paths.data_dir,
        optionsdx_root=optionsdx_root,
        symbol=inputs.symbol,
        start=start,
        end=end,
        options_source=inputs.options_source,
        preferences_payload=prefs.model_dump(mode="json"),
        strategy_runs_dir=strategy_runs,
        run_prefix=f"{stamp}-{git_sha}",
        sweep_id=sweep_id,
        fixed_params=fixed_overrides,
        write_html=html,
    )
    context = SweepContext(spec=spec, strategy_cls=strategy_cls, inputs=inputs, preferences=prefs)
    typer.echo(
        f"Sweep {sweep_id}: {len(combos)} combination(s) of {probe.name} on {inputs.symbol}, jobs={jobs}"
    )
    completed = 0

    def report(row: dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        label = " ".join(
            f"{key.removeprefix(PARAM_PREFIX)}={value}"
            for key, value in row.items()
            if key.startswith(PARAM_PREFIX)
        )
        if row["status"] == "ok":
            value = row.get(metric)
            shown = f"{value:.4f}" if isinstance(value, int | float) else "n/a"
            typer.echo(f"[{completed}/{len(combos)}] {label} {metric}={shown}")
        else:
            typer.secho(
                f"[{completed}/{len(combos)}] {label} failed: {row['error']}", fg=typer.colors.RED, err=True
            )

    results = run_sweep(context, combos, jobs=jobs, fail_fast=fail_fast, on_result=report)
    sweep_dir = write_sweep_summary(strategy_runs / SWEEPS_DIRNAME / sweep_id, results, manifest)
    typer.echo(f"Sweep summary: {sweep_dir / 'sweep.parquet'}")

    ranked = rank_results(results, metric, minimize=minimize)
    if metric not in results.columns:
        typer.secho(
            f"Warning: metric '{metric}' not found in results; nothing ranked.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    elif top and not ranked.empty:
        param_columns = [column for column in ranked.columns if column.startswith(PARAM_PREFIX)]
        typer.echo(f"Top {min(top, len(ranked))} by {metric} ({'lowest' if minimize else 'highest'} first):")
        for _, row in ranked.head(top).iterrows():
            label = " ".join(f"{column.removeprefix(PARAM_PREFIX)}={row[column]}" for column in param_columns)
            typer.echo(f"  {metric}={row[metric]:.4f} {label} -> {row['run_dir']}")
    if len(ranked) > 1:
        typer.echo("Note: the best of many combinations is an optimistic estimate; confirm it out of sample.")
    failed = int((results["status"] == "error").sum()) if not results.empty else 0
    if failed:
        typer.secho(
            f"Warning: {failed} combination(s) failed; see {sweep_dir / 'sweep.parquet'}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        if fail_fast:
            raise typer.Exit(code=1)


@app.command("list-runs")
def list_runs(limit: int = typer.Option(20, help="Max run directories to show")) -> None:
    root = _repo_root()
    prefs = _load_preferences(root)
    runs_root = root / prefs.paths.runs_dir
    if not runs_root.exists():
        typer.echo("No runs directory found yet.")
        return
    run_dirs = sorted([path.parent for path in runs_root.glob("**/metrics.json")], reverse=True)
    if not run_dirs:
        typer.echo("No runs available.")
        return
    for run_dir in run_dirs[:limit]:
        metrics_file = run_dir / "metrics.json"
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        typer.echo(
            f"{run_dir} | total_return={metrics.get('total_return', 0):.4f} "
            f"sharpe={metrics.get('sharpe', 0):.4f} drawdown={metrics.get('max_drawdown', 0):.4f}"
        )


@app.command("compare")
def compare_runs(strategy_name: str, limit: int = typer.Option(5, help="Latest runs to compare")) -> None:
    root = _repo_root()
    prefs = _load_preferences(root)
    runs_root = root / prefs.paths.runs_dir
    metrics_files = sorted(
        runs_root.glob(f"**/{strategy_name}/*/metrics.json"),
        reverse=True,
    )
    if not metrics_files:
        typer.echo(f"No runs found for strategy `{strategy_name}`")
        return
    for metrics_file in metrics_files[:limit]:
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        run_dir = metrics_file.parent
        typer.echo(
            f"{run_dir.name}: return={metrics.get('total_return', 0):.4f}, "
            f"sharpe={metrics.get('sharpe', 0):.4f}, hit_rate={metrics.get('hit_rate', 0):.4f}"
        )


@app.command("normalize-optionsdx")
def normalize_optionsdx(
    input_dir: str | None = typer.Option(None, help="Root folder containing OptionsDX *.txt files"),
    output_dir: str | None = typer.Option(None, help="Output root for normalized Parquet partitions"),
    reports_dir: str | None = typer.Option(None, help="Output folder for per-file and run JSON reports"),
    dry_run: bool = typer.Option(False, help="Parse and score only; do not write Parquet, reports, or state"),
    fail_on_errors: bool = typer.Option(False, help="Exit non-zero if any parse errors occur"),
    fail_on_gates: bool = typer.Option(False, help="Exit non-zero if any rate gate fails"),
    max_negative_iv_rate: float | None = typer.Option(
        None,
        help="If set, fail when a file's NEGATIVE_IV row rate exceeds this threshold (0-1)",
    ),
    max_crossed_market_rate: float | None = typer.Option(
        None,
        help="If set, fail when a file's crossed-market row rate exceeds this threshold (0-1)",
    ),
) -> None:
    root = _repo_root()
    prefs = _load_preferences(root)
    ox = prefs.optionsdx
    inp = Path(input_dir or ox.input_dir)
    if not inp.is_absolute():
        inp = (root / inp).resolve()
    out = Path(output_dir or ox.output_dir)
    if not out.is_absolute():
        out = (root / out).resolve()
    rep = Path(reports_dir or ox.reports_dir)
    if not rep.is_absolute():
        rep = (root / rep).resolve()
    state_path = root / "state" / "optionsdx_normalize_state.json"
    opts = NormalizeOptions(
        input_root=inp,
        output_root=out,
        reports_dir=rep,
        state_path=state_path,
        dry_run=dry_run,
        fail_on_errors=fail_on_errors,
        max_negative_iv_rate=max_negative_iv_rate,
        max_crossed_market_rate=max_crossed_market_rate,
    )
    summary = run_normalize(opts)
    typer.echo(json.dumps(summary["totals"], indent=2, sort_keys=True))
    if summary["gate_failures"]:
        for msg in summary["gate_failures"]:
            typer.echo(f"GATE: {msg}", err=True)
    if fail_on_errors and summary["totals"].get("parse_errors", 0) > 0:
        raise typer.Exit(code=1)
    if fail_on_gates and summary["gate_failures"]:
        raise typer.Exit(code=1)


@app.command("dashboard")
def dashboard_cmd(
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8501, help="Streamlit port"),
    headless: bool = typer.Option(
        True,
        "--headless/--no-headless",
        help="Headless server (use --no-headless to auto-open a browser tab)",
    ),
) -> None:
    """Launch the read-only Streamlit run review dashboard."""
    app_path = Path(reporting_dashboard.__path__[0]) / "app.py"
    app_path = app_path.resolve()
    if not app_path.is_file():
        typer.echo(f"Dashboard entry missing at {app_path}", err=True)
        raise typer.Exit(code=1)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true" if headless else "false",
    ]
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError as exc:
        typer.echo(
            "Streamlit is not on PATH. Install the package dependencies (e.g. `pip install -e .`).",
            err=True,
        )
        raise typer.Exit(code=1) from exc


app.add_typer(universe_app, name="universe")


# ---------------------------------------------------------------------------
# DoltHub + volatility commands
# ---------------------------------------------------------------------------

@app.command("fetch-dolthub")
def fetch_dolthub(
    symbol: str,
    start: str = typer.Option("2020-01-01", help="Start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD (default: today)"),
    vol_only: bool = typer.Option(False, "--vol-only", help="Fetch only volatility_history (faster)"),
) -> None:
    """Fetch option chain or vol history from DoltHub for one symbol."""
    root = _repo_root()
    sym = _validate_symbol(symbol)
    prefs = _load_preferences(root)
    store = DuckDBStore(root / prefs.paths.data_dir)
    end_date = end or date.today().isoformat()

    if vol_only:
        typer.echo(f"Fetching vol history {sym} {start} → {end_date} from DoltHub...")
        vol_df = fetch_dolthub_vol_history(sym, start, end_date)
        if vol_df.empty:
            typer.echo(f"No vol history rows for {sym}.")
            return
        vol_dir = root / prefs.paths.data_dir / "cache" / "vol"
        vol_dir.mkdir(parents=True, exist_ok=True)
        vol_path = vol_dir / f"{sym}.parquet"
        vol_df.to_parquet(vol_path, index=False)
        typer.echo(f"Vol history {sym}: {len(vol_df)} rows → {vol_path}")
    else:
        typer.echo(f"Fetching chain {sym} {start} → {end_date} from DoltHub...")
        chain_df = fetch_dolthub_chain(sym, start, end_date)
        if chain_df.empty:
            typer.echo(f"No chain rows for {sym}.")
            return
        # Merge underlying_last from stock bars if available
        bars = store.read_bars(sym)
        if not bars.empty:
            close_map = dict(zip(bars["date"].astype(str).str[:10], pd.to_numeric(bars["close"], errors="coerce")))
            chain_df["underlying_last"] = chain_df["asof"].map(close_map)
        else:
            chain_df["underlying_last"] = 0.0
        # Compute DTE
        chain_df["dte"] = (
            pd.to_datetime(chain_df["expiry"], errors="coerce") -
            pd.to_datetime(chain_df["asof"], errors="coerce")
        ).dt.days
        store.write_chain(sym, chain_df)
        typer.echo(f"Chain {sym}: {len(chain_df)} rows → data/options/{sym}.parquet")


@app.command("fetch-dolthub-universe")
def fetch_dolthub_universe(
    start: str = typer.Option("2020-01-01", help="Start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD (default: today)"),
    vol_only: bool = typer.Option(False, "--vol-only", help="Fetch only volatility_history"),
    pilot: bool = typer.Option(False, "--pilot", help="Use 25-symbol pilot list instead of full universe"),
    delay: float = typer.Option(2.0, help="Seconds between symbols (rate limiting)"),
) -> None:
    """Fetch DoltHub data for every symbol in the weekly options universe."""
    import time

    root = _repo_root()
    prefs = _load_preferences(root)
    store = DuckDBStore(root / prefs.paths.data_dir)
    end_date = end or date.today().isoformat()

    if pilot:
        symbols = list(PILOT_SYMBOLS)
    else:
        universe_path = root / "data" / "weekly_options_stocks.csv"
        symbols = load_stock_universe(universe_path)

    typer.echo(f"Fetching {'vol history' if vol_only else 'chain'} for {len(symbols)} symbols ({start} → {end_date})...")
    ok, fail = 0, 0
    for i, sym in enumerate(symbols, 1):
        try:
            if vol_only:
                vol_df = fetch_dolthub_vol_history(sym, start, end_date)
                if not vol_df.empty:
                    vol_dir = root / prefs.paths.data_dir / "cache" / "vol"
                    vol_dir.mkdir(parents=True, exist_ok=True)
                    vol_df.to_parquet(vol_dir / f"{sym}.parquet", index=False)
                    typer.echo(f"  [{i}/{len(symbols)}] {sym}: {len(vol_df)} vol rows ✓")
                    ok += 1
                else:
                    typer.echo(f"  [{i}/{len(symbols)}] {sym}: no data")
                    fail += 1
            else:
                chain_df = fetch_dolthub_chain(sym, start, end_date)
                if not chain_df.empty:
                    bars = store.read_bars(sym)
                    if not bars.empty:
                        close_map = dict(zip(bars["date"].astype(str).str[:10], pd.to_numeric(bars["close"], errors="coerce")))
                        chain_df["underlying_last"] = chain_df["asof"].map(close_map)
                    else:
                        chain_df["underlying_last"] = 0.0
                    chain_df["dte"] = (
                        pd.to_datetime(chain_df["expiry"], errors="coerce") -
                        pd.to_datetime(chain_df["asof"], errors="coerce")
                    ).dt.days
                    store.write_chain(sym, chain_df)
                    typer.echo(f"  [{i}/{len(symbols)}] {sym}: {len(chain_df)} chain rows ✓")
                    ok += 1
                else:
                    typer.echo(f"  [{i}/{len(symbols)}] {sym}: no data")
                    fail += 1
        except Exception as exc:
            typer.echo(f"  [{i}/{len(symbols)}] {sym}: ERROR {exc}")
            fail += 1
        if i < len(symbols):
            time.sleep(delay)
    typer.echo(f"\nDone: {ok} ok, {fail} failed/no data, out of {len(symbols)} symbols.")


@app.command("vol")
def vol_symbol(
    symbol: str,
    start: str | None = typer.Option(None, help="Start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD"),
) -> None:
    """Build daily volatility series for *symbol* from stored chain or vol cache."""
    root = _repo_root()
    sym = _validate_symbol(symbol)
    prefs = _load_preferences(root)
    store = DuckDBStore(root / prefs.paths.data_dir)

    bars = store.read_bars(sym, start=start, end=end)
    if bars.empty:
        typer.echo(f"No stock bars for {sym}. Run `lambdaclass fetch {sym}` first.")
        raise typer.Exit(1)
    earnings = store.read_earnings(sym)

    # Try vol history cache first (fast path)
    vol_cache_path = root / prefs.paths.data_dir / "cache" / "vol" / f"{sym}.parquet"
    if vol_cache_path.is_file():
        vol_history = pd.read_parquet(vol_cache_path)
        series = build_vol_series_from_history(vol_history, earnings)
        if not series.empty:
            # Merge close from bars
            close_map = dict(zip(bars["date"].astype(str).str[:10], pd.to_numeric(bars["close"], errors="coerce")))
            series["close"] = series["date"].map(close_map).fillna(0.0)
            typer.echo(f"Vol series (from vol_history) {sym}: {len(series)} rows")
        else:
            typer.echo(f"vol_history empty for {sym}, falling back to chain.")
            series = _vol_from_chain(store, sym, bars, earnings)
    else:
        series = _vol_from_chain(store, sym, bars, earnings)

    if series.empty:
        typer.echo(f"No vol series for {sym}. Fetch chain or vol history first.")
        raise typer.Exit(1)

    vol_dir = root / prefs.paths.data_dir / "cache" / "vol"
    vol_dir.mkdir(parents=True, exist_ok=True)
    out_path = vol_dir / f"{sym}.parquet"
    series.to_parquet(out_path, index=False)
    typer.echo(f"Vol series {sym}: {len(series)} rows → {out_path}")
    if "iv30" in series.columns:
        iv = pd.to_numeric(series["iv30"], errors="coerce").dropna()
        if not iv.empty:
            typer.echo(f"  IV30 range: {iv.min():.4f} – {iv.max():.4f} (last: {iv.iloc[-1]:.4f})")


def _vol_from_chain(store: DuckDBStore, sym: str, bars: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    """Build vol series from stored option chain."""
    chain = store.read_chain(sym)
    if chain is None or chain.empty:
        return pd.DataFrame()
    return build_vol_series(bars, chain, earnings)


@app.command("iv-study")
def iv_study(
    symbols: list[str] | None = typer.Option(None, "--symbol", help="Specific symbols (repeatable); default: pilot list"),
    start: str | None = typer.Option(None, help="Start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD"),
    pilot: bool = typer.Option(True, "--pilot/--no-pilot", help="Use 25-symbol pilot list (default)"),
) -> None:
    """Run the earnings IV ramp/crush study across symbols."""
    from lambdaclass.earnings_cycle_runner import run_earnings_iv_study

    root = _repo_root()
    prefs = _load_preferences(root)
    store = DuckDBStore(root / prefs.paths.data_dir)
    vol_cache_dir = root / prefs.paths.data_dir / "cache" / "vol"

    if symbols:
        sym_list = [s.upper() for s in symbols]
    elif pilot:
        sym_list = list(PILOT_SYMBOLS)
    else:
        universe_path = root / "data" / "weekly_options_stocks.csv"
        sym_list = load_stock_universe(universe_path)

    typer.echo(f"Running IV ramp/crush study for {len(sym_list)} symbols...")
    result = run_earnings_iv_study(
        store=store,
        symbols=sym_list,
        start=start,
        end=end,
        vol_cache_dir=vol_cache_dir,
    )

    summary = result["summary"]
    typer.echo(f"\n{'='*50}")
    typer.echo(f"IV Ramp/Crush Study Results")
    typer.echo(f"{'='*50}")
    typer.echo(f"Symbols with data:  {summary.get('symbols_with_data', 0)}")
    typer.echo(f"Total events:       {summary.get('total_events', 0)}")
    typer.echo(f"Median ramp pct:    {summary.get('median_ramp_pct', 0)*100:.2f}%")
    typer.echo(f"Median crush pct:   {summary.get('median_crush_pct', 0)*100:.2f}%")
    typer.echo(f"Realized < implied: {summary.get('share_realized_below_implied', 0)*100:.1f}% of the time")
    typer.echo(f"Median realized/impl: {summary.get('median_realized_over_implied', 0):.2f}x")

    events_df = result["event_table"]
    if not events_df.empty:
        out_dir = root / prefs.paths.data_dir / "earnings" / "events"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "iv_ramp_crush_study.parquet"
        events_df.to_parquet(out_path, index=False)
        typer.echo(f"\nEvent table: {out_path} ({len(events_df)} events)")

        cycle_df = result["cycle"]
        if not cycle_df.empty:
            cycle_path = out_dir / "iv_cycle_aggregate.parquet"
            cycle_df.to_parquet(cycle_path, index=False)
            typer.echo(f"Cycle aggregate: {cycle_path}")

    per_symbol = result.get("per_symbol", {})
    if per_symbol:
        typer.echo(f"\nPer-symbol event counts:")
        for sym, count in sorted(per_symbol.items(), key=lambda x: -x[1])[:10]:
            typer.echo(f"  {sym}: {count} events")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
