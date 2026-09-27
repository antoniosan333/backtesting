from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import typer

from lambdaclass.backtest.engine import run_backtest, write_run_outputs
from lambdaclass.config import (
    DEFAULT_PREFERENCES,
    Preferences,
    build_snapshot_payload,
    compute_config_hash,
    snapshot_preferences,
)
from lambdaclass.data_adapters.optionsdx_chain_loader import load_normalized_optionsdx_chain
from lambdaclass.data_adapters.optionsdx_normalize import NormalizeOptions, run_normalize
from lambdaclass.data_adapters.yfinance_adapter import YFinanceAdapter
from lambdaclass.data_adapters.dolthub_chain import fetch_dolthub_chain
from lambdaclass.data_adapters.edgar_earnings import fetch_earnings_frame
from lambdaclass.earnings.calendar import normalize_earnings_frame
from lambdaclass.reporting.earnings_metrics import (
    average_iv_from_option_trades,
    compute_earnings_events,
    summarize_earnings_events,
)
from lambdaclass.reporting.metrics import compute_metrics
from lambdaclass.reporting import dashboard as reporting_dashboard
from lambdaclass.reporting.tearsheet import write_tearsheet
from lambdaclass.state import load_json, save_json
from lambdaclass.storage.duckdb_store import DuckDBStore
from lambdaclass.storage.optionsdx_reader import read_atm_slice
from lambdaclass.volatility.earnings_cycle import align_events, cycle_summary, event_table, normalize
from lambdaclass.volatility.series import build_vol_series, merge_vol_cache
from lambdaclass.volatility.universe import PILOT_SYMBOLS, load_stock_universe
from lambdaclass.strategies.base import Strategy
from lambdaclass.strategies.scaffolder import ensure_month_dir, scaffold_strategy

app = typer.Typer(help="LambdaClass stock + options backtesting CLI")
STRATEGY_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


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
            f"Warning: no options chain rows loaded from '{source}'. "
            "Any option orders will be rejected.",
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
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{_git_short_sha(root)}-{config_hash}"


def _validate_strategy_name(strategy_name: str) -> str:
    if not STRATEGY_NAME_PATTERN.match(strategy_name):
        raise typer.BadParameter(
            "Strategy name must start with a letter and contain only letters, numbers, and underscores."
        )
    return strategy_name


def _load_strategy(path: Path) -> Strategy:
    module_name = f"strategy_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise typer.BadParameter(f"Could not load strategy module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategy_cls = getattr(module, "StrategyImpl", None)
    if strategy_cls is None:
        raise typer.BadParameter("Strategy file must define StrategyImpl class")
    strategy = strategy_cls()
    if not isinstance(strategy, Strategy):
        raise typer.BadParameter("StrategyImpl must inherit lambdaclass.strategies.base.Strategy")
    return strategy


def _find_strategy_file(strategies_dir: Path, strategy_name: str) -> Path:
    matches = sorted(strategies_dir.glob(f"*/{strategy_name}.py"))
    if not matches:
        raise typer.BadParameter(f"Strategy `{strategy_name}` not found under {strategies_dir}")
    resolved = matches[-1].resolve()
    root = strategies_dir.resolve()
    if root not in resolved.parents:
        raise typer.BadParameter("Strategy path escapes strategies directory.")
    return resolved


def _fetch_with_retry(fetch_fn: Callable[[], object], retries: int = 3, delay_seconds: float = 1.0) -> object:
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
        data_dir / "cache",
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
    end: str = typer.Option(date.today().isoformat(), help="End date in YYYY-MM-DD"),
) -> None:
    root = _repo_root()
    prefs = _load_preferences(root)
    store = DuckDBStore(root / prefs.paths.data_dir)
    adapter = _get_adapter(prefs.defaults.data_adapter)
    start_dt = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)
    bars = _fetch_with_retry(lambda: adapter.get_stock_bars(symbol, start_dt, end_dt))
    chain = _fetch_with_retry(lambda: adapter.get_option_chain(symbol, end_dt))
    bars_path = store.write_bars(symbol, bars)
    chain_path = store.write_chain(symbol, chain)
    markers_path = root / "state" / "fetch_markers.json"
    markers = load_json(markers_path)
    markers[symbol.upper()] = {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "bars_path": str(bars_path),
        "options_path": str(chain_path),
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
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
    _ = force  # dedupe keep-last always overwrites matching dates
    root = _repo_root()
    prefs = _load_preferences(root)
    store = DuckDBStore(root / prefs.paths.data_dir)
    symbol = symbol.upper()
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
        help="Exit non-zero if any option order could not be priced from the chain",
    ),
) -> None:
    root = _repo_root()
    prefs = _load_preferences(root)
    strategies_dir = root / prefs.paths.strategies_dir
    validated_strategy = _validate_strategy_name(strategy_name)
    strategy_path = _find_strategy_file(strategies_dir, validated_strategy)
    strategy = _load_strategy(strategy_path)
    store = DuckDBStore(root / prefs.paths.data_dir)
    symbol = symbol.upper()
    bars = store.read_bars(symbol, start=start, end=end)
    if bars.empty:
        raise typer.BadParameter("No stock bars found. Run `lambdaclass fetch <SYMBOL>` first.")
    bars = bars.sort_values("date").reset_index(drop=True)
    chain_src = (options_source or prefs.defaults.options_chain_source).strip().lower()
    if chain_src not in ("yfinance", "optionsdx"):
        raise typer.BadParameter("options_source must be yfinance or optionsdx")
    if chain_src == "optionsdx":
        ox_root = Path(prefs.optionsdx.output_dir)
        if not ox_root.is_absolute():
            ox_root = (root / ox_root).resolve()
        options_chain = load_normalized_optionsdx_chain(ox_root, symbol, bars["date"])
    else:
        options_chain = store.read_chain(symbol)
    _warn_chain_coverage(options_chain, bars["date"], chain_src)
    # Earnings calendar (full history so days_to/since work at window edges)
    earnings = store.read_earnings(symbol)
    vol_path = root / prefs.paths.data_dir / "cache" / "vol" / f"{symbol}.parquet"
    vol_series = pd.read_parquet(vol_path) if vol_path.is_file() else None
    now = datetime.now()
    month = ensure_month_dir(root / prefs.paths.strategies_dir, now=now).name
    cli_overrides = {"start": start, "end": end}
    snapshot_payload = build_snapshot_payload(prefs, strategy.params, cli_overrides)
    cfg_hash = compute_config_hash(snapshot_payload)
    run_id = _run_id(cfg_hash, root)
    run_dir = root / prefs.paths.runs_dir / month / strategy.name / run_id
    run_result = run_backtest(
        strategy, bars, options_chain, prefs, earnings=earnings, vol_series=vol_series
    )
    trades_path, equity_path = write_run_outputs(run_result, run_dir)
    metrics = compute_metrics(run_result.equity_curve)
    events = compute_earnings_events(
        option_trades=run_result.option_trades,
        earnings=earnings,
        bars=bars,
        iv_by_date=average_iv_from_option_trades(run_result.option_trades),
        expected_moves=run_result.expected_moves,
    )
    if not events.empty:
        events.to_parquet(run_dir / "events.parquet", index=False)
        metrics.update(summarize_earnings_events(events))
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    snapshot_path = run_dir / "config.snapshot.toml"
    snapshot_preferences(
        preferences=prefs,
        strategy_params=strategy.params,
        cli_overrides=cli_overrides,
        output_path=snapshot_path,
    )
    log_path = run_dir / "run.log"
    log_path.write_text(
        (
            f"strategy={strategy.name}\n"
            f"symbol={symbol}\n"
            f"rows={len(bars)}\n"
            f"trades={len(run_result.trades)}\n"
            f"option_trades={len(run_result.option_trades)}\n"
            f"rejected_orders={len(run_result.rejected_orders)}\n"
            f"earnings_events={len(events)}\n"
            f"config_hash={cfg_hash}\n"
        ),
        encoding="utf-8",
    )
    if prefs.reporting.save_html:
        write_tearsheet(run_result.equity_curve, run_dir / "report.html", theme=prefs.reporting.plot_theme)
    typer.echo(f"Run complete: {run_dir}")
    typer.echo(f"Metrics: {metrics_path}")
    typer.echo(f"Trades: {trades_path}")
    typer.echo(f"Equity: {equity_path}")
    if not events.empty:
        typer.echo(f"Events: {run_dir / 'events.parquet'}")
    rejected = run_result.rejected_orders
    if not rejected.empty:
        by_reason = ", ".join(f"{reason}={count}" for reason, count in rejected["reason"].value_counts().items())
        typer.secho(
            f"Warning: {len(rejected)} option order(s) rejected ({by_reason}). "
            f"See {run_dir / 'rejected_orders.csv'}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        if fail_on_rejected_orders:
            raise typer.Exit(code=1)


def _symbols_for_vol(root: Path, symbol: str | None, pilot: bool, universe: bool) -> list[str]:
    selected = int(symbol is not None) + int(pilot) + int(universe)
    if selected != 1:
        raise typer.BadParameter("Pass a symbol, --pilot, or --universe")
    if symbol is not None:
        return [symbol.upper()]
    if pilot:
        return list(PILOT_SYMBOLS)
    return load_stock_universe(root / "data" / "weekly_options_stocks.csv")


def _build_one_vol_series(
    *,
    root: Path,
    prefs: Preferences,
    store: DuckDBStore,
    symbol: str,
    start: str,
    end: str,
    rebuild: bool,
) -> pd.DataFrame:
    bars = store.read_bars(symbol, start=start, end=end)
    if bars.empty:
        raise RuntimeError("no stock bars; run fetch first")
    ox_root = Path(prefs.optionsdx.output_dir)
    if not ox_root.is_absolute():
        ox_root = root / ox_root
    chain = read_atm_slice(
        ox_root,
        symbol,
        start,
        end,
        max_strike_distance_pct=prefs.volatility.max_strike_distance_pct,
    )
    if chain.empty:
        chain = store.read_chain(symbol)
        if not chain.empty and "asof" in chain.columns:
            chain = chain[(chain["asof"].astype(str) >= start) & (chain["asof"].astype(str) <= end)]
    if chain.empty:
        chain = fetch_dolthub_chain(symbol, start, end)
        if chain.empty:
            raise RuntimeError("no option chain from DoltHub")
        store.write_chain(symbol, chain)
    earnings = store.read_earnings(symbol)
    updated = build_vol_series(
        bars,
        chain,
        earnings,
        windows=prefs.volatility.hv_windows,
        targets=prefs.volatility.iv_targets_dte,
        lookback_days=prefs.volatility.lookback_days,
    )
    cache_path = root / prefs.paths.data_dir / "cache" / "vol" / f"{symbol}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild or not cache_path.is_file():
        series = updated
    else:
        series = merge_vol_cache(pd.read_parquet(cache_path), updated)
    series.to_parquet(cache_path, index=False)
    return series


@app.command("vol")
def vol_command(
    symbol: str | None = typer.Argument(None, help="Ticker, or omit with --pilot / --universe"),
    start: str | None = typer.Option(None, help="YYYY-MM-DD start; default is [volatility].history_start"),
    end: str | None = typer.Option(None, help="YYYY-MM-DD end; default is today"),
    rebuild: bool = typer.Option(False, help="Replace the cached series instead of refreshing the tail"),
    csv: str | None = typer.Option(None, help="Write the series CSV for a single symbol"),
    pilot: bool = typer.Option(False, help="Build the 25-name earnings pilot"),
    universe: bool = typer.Option(False, help="Build every name in weekly_options_stocks.csv"),
) -> None:
    """Build the historical-versus-implied volatility series and print the earnings cycle."""
    root = _repo_root()
    prefs = _load_preferences(root)
    store = DuckDBStore(root / prefs.paths.data_dir)
    symbols = _symbols_for_vol(root, symbol, pilot, universe)
    if csv and len(symbols) != 1:
        raise typer.BadParameter("--csv applies to a single symbol")
    start_day = start or prefs.volatility.history_start
    end_day = end or datetime.now(tz=timezone.utc).date().isoformat()
    markers_path = root / "state" / "vol_series_markers.json"
    markers = load_json(markers_path)
    skipped: list[dict[str, str]] = []
    for ticker in symbols:
        try:
            series = _build_one_vol_series(
                root=root,
                prefs=prefs,
                store=store,
                symbol=ticker,
                start=start_day,
                end=end_day,
                rebuild=rebuild,
            )
        except Exception as exc:
            skipped.append({"symbol": ticker, "reason": str(exc)})
            typer.secho(f"skip {ticker}: {exc}", fg=typer.colors.YELLOW, err=True)
            continue
        latest = series.dropna(subset=["iv30"]).tail(1)
        if latest.empty:
            latest = series.tail(1)
        row = latest.iloc[0]
        typer.echo(
            f"{ticker} {row['date']} iv30={row['iv30']} hv20={row['hv20']} "
            f"spread={row['iv_hv_spread']} ratio={row['iv_hv_ratio']} "
            f"percentile={row['iv_pctile_252']} days_to_earnings={row['days_to_next_earnings']}"
        )
        earnings = store.read_earnings(ticker)
        summary = cycle_summary(
            event_table(
                normalize(
                    align_events(
                        series,
                        earnings,
                        pre_days=prefs.volatility.pre_days,
                        post_days=prefs.volatility.post_days,
                    )
                )
            )
        )
        typer.echo(
            f"  cycle ramp={summary['median_ramp_pct']:.3f} "
            f"crush={summary['median_crush_pct']:.3f} "
            f"inside_implied={summary['share_realized_below_implied']:.3f}"
        )
        markers[ticker] = {
            "last_date": str(series["date"].iloc[-1]),
            "rows": int(len(series)),
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        if csv:
            series.to_csv(csv, index=False)
            typer.echo(f"  csv={csv}")
    markers["skipped"] = skipped
    save_json(markers_path, markers)


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
