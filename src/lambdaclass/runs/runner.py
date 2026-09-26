"""Load backtest inputs and execute one run into a run directory.

Shared by ``lambdaclass run`` and ``lambdaclass sweep``; performs no console output.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lambdaclass.backtest.engine import RunResult, prepare_chain_by_date, run_backtest, write_run_outputs
from lambdaclass.config import (
    Preferences,
    build_snapshot_payload,
    compute_config_hash,
    snapshot_preferences,
)
from lambdaclass.data_adapters.optionsdx_chain_loader import load_normalized_optionsdx_chain
from lambdaclass.reporting.earnings_metrics import (
    average_iv_from_option_trades,
    compute_earnings_events,
    summarize_earnings_events,
)
from lambdaclass.reporting.metrics import compute_metrics
from lambdaclass.reporting.tearsheet import write_tearsheet
from lambdaclass.storage.duckdb_store import DuckDBStore
from lambdaclass.strategies.base import Strategy

OPTIONS_SOURCES = ("yfinance", "optionsdx")


@dataclass(frozen=True)
class RunInputs:
    symbol: str
    options_source: str
    bars: pd.DataFrame
    options_chain: pd.DataFrame
    chain_by_date: dict[str, pd.DataFrame]
    earnings: pd.DataFrame
    dividends_backfilled: bool = False


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    config_hash: str
    metrics: dict[str, Any]
    result: RunResult
    events: pd.DataFrame


def load_run_inputs(
    store: DuckDBStore,
    *,
    symbol: str,
    start: str | None,
    end: str | None,
    options_source: str,
    optionsdx_root: Path,
) -> RunInputs:
    """Read bars, chain, and earnings for one symbol; raise ``ValueError`` when unusable."""
    source = options_source.strip().lower()
    if source not in OPTIONS_SOURCES:
        raise ValueError("options_source must be yfinance or optionsdx")
    bars = store.read_bars(symbol, start=start, end=end)
    if bars.empty:
        raise ValueError("No stock bars found. Run `lambdaclass fetch <SYMBOL>` first.")
    backfilled = bool(bars.attrs.get("dividends_backfilled"))
    bars = bars.sort_values("date").reset_index(drop=True)
    if source == "optionsdx":
        options_chain = load_normalized_optionsdx_chain(optionsdx_root, symbol, bars["date"])
    else:
        options_chain = store.read_chain(symbol)
    # Full earnings history so days_to/since work at window edges.
    earnings = store.read_earnings(symbol)
    return RunInputs(
        symbol=symbol,
        options_source=source,
        bars=bars,
        options_chain=options_chain,
        chain_by_date=prepare_chain_by_date(options_chain),
        earnings=earnings,
        dividends_backfilled=backfilled,
    )


def execute_run(
    strategy: Strategy,
    inputs: RunInputs,
    preferences: Preferences,
    *,
    cli_overrides: Mapping[str, Any],
    run_dir_for: Callable[[str], Path],
    write_html: bool | None = None,
    log_extra: Mapping[str, Any] | None = None,
) -> RunSummary:
    """Backtest ``strategy`` and write the standard run artifacts.

    ``run_dir_for`` maps the config hash to the run directory. ``write_html``
    defaults to ``[reporting].save_html``.
    """
    overrides = dict(cli_overrides)
    snapshot_payload = build_snapshot_payload(preferences, strategy.params, overrides)
    config_hash = compute_config_hash(snapshot_payload)
    run_dir = run_dir_for(config_hash)
    result = run_backtest(
        strategy,
        inputs.bars,
        inputs.options_chain,
        preferences,
        earnings=inputs.earnings,
        chain_by_date=inputs.chain_by_date,
    )
    write_run_outputs(result, run_dir)
    metrics: dict[str, Any] = compute_metrics(
        result.equity_curve,
        risk_free_rate=preferences.defaults.risk_free_rate,
    )
    events = compute_earnings_events(
        option_trades=result.option_trades,
        earnings=inputs.earnings,
        bars=inputs.bars,
        iv_by_date=average_iv_from_option_trades(result.option_trades),
    )
    if not events.empty:
        events.to_parquet(run_dir / "events.parquet", index=False)
        metrics.update(summarize_earnings_events(events))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    snapshot_preferences(
        preferences=preferences,
        strategy_params=strategy.params,
        cli_overrides=overrides,
        output_path=run_dir / "config.snapshot.toml",
    )
    log_lines = {
        "strategy": strategy.name,
        "symbol": inputs.symbol,
        "rows": len(inputs.bars),
        "trades": len(result.trades),
        "option_trades": len(result.option_trades),
        "rejected_orders": len(result.rejected_orders),
        "earnings_events": len(events),
        "config_hash": config_hash,
        **(log_extra or {}),
    }
    (run_dir / "run.log").write_text(
        "".join(f"{key}={value}\n" for key, value in log_lines.items()),
        encoding="utf-8",
    )
    save_html = preferences.reporting.save_html if write_html is None else write_html
    if save_html:
        write_tearsheet(result.equity_curve, run_dir / "report.html", theme=preferences.reporting.plot_theme)
    return RunSummary(
        run_dir=run_dir,
        config_hash=config_hash,
        metrics=metrics,
        result=result,
        events=events,
    )
