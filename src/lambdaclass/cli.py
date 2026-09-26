from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import typer

from lambdaclass.config import DEFAULT_PREFERENCES, Preferences, compute_config_hash
from lambdaclass.data_adapters.optionsdx_normalize import NormalizeOptions, run_normalize
from lambdaclass.data_adapters.yfinance_adapter import YFinanceAdapter
from lambdaclass.earnings.calendar import normalize_earnings_frame
from lambdaclass.reporting import dashboard as reporting_dashboard
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
    return root / "config" / "preferences.toml"


def _load_preferences(root: Path) -> Preferences:
    path = _preferences_path(root)
    if not path.exists():
        return DEFAULT_PREFERENCES
    return Preferences.load(path)


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
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fetch_fn()
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            last_error = exc
            if attempt == retries:
                break
            time.sleep(delay_seconds * attempt)
    if last_error is not None:
        raise typer.BadParameter(f"Data fetch failed after {retries} attempts: {last_error}") from last_error
    raise typer.BadParameter("Data fetch failed for an unknown reason.")


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
    else:
        adapter = _get_adapter(prefs.defaults.data_adapter)
        raw = _fetch_with_retry(lambda: adapter.get_earnings_dates(symbol))
        frame = normalize_earnings_frame(raw, symbol=symbol, source="yfinance")
    out_path = store.write_earnings(symbol, frame)
    markers_path = root / "state" / "earnings_fetch_markers.json"
    markers = load_json(markers_path)
    markers[symbol] = {
        "rows": int(len(frame)),
        "path": str(out_path),
        "source": "csv" if csv else "yfinance",
        "updated_at": datetime.now(tz=UTC).isoformat(),
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
