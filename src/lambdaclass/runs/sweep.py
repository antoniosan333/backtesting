"""Parameter sweeps: expand a grid of strategy params and run every combination.

Each combination is written as an ordinary run directory; the sweep adds a
summary under ``<strategy runs dir>/_sweeps/<sweep_id>/``.
"""

from __future__ import annotations

import itertools
import json
import math
import multiprocessing
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from lambdaclass.config import Preferences
from lambdaclass.runs.layout import PARAM_PREFIX, SWEEP_MANIFEST_FILE, SWEEP_RESULTS_FILE
from lambdaclass.runs.runner import RunInputs, execute_run, load_run_inputs
from lambdaclass.storage.duckdb_store import DuckDBStore
from lambdaclass.strategies.base import Strategy
from lambdaclass.strategies.loader import load_strategy_class
from lambdaclass.strategies.params import ParamError, apply_params, coerce_param, parse_assignments

MAX_RANGE_VALUES = 10_000


class SweepError(ValueError):
    """Invalid grid or sweep configuration."""


@dataclass(frozen=True)
class SweepSpec:
    """Everything a worker process needs to rebuild the run context."""

    strategy_path: Path
    data_dir: Path
    optionsdx_root: Path
    symbol: str
    start: str | None
    end: str | None
    options_source: str
    preferences_payload: dict[str, Any]
    strategy_runs_dir: Path
    run_prefix: str
    sweep_id: str
    fixed_params: dict[str, Any] = field(default_factory=dict)
    write_html: bool = False


@dataclass(frozen=True)
class SweepContext:
    spec: SweepSpec
    strategy_cls: type[Strategy]
    inputs: RunInputs
    preferences: Preferences


def _range_values(text: str) -> list[str] | None:
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        start, stop, step = (Decimal(part.strip()) for part in parts)
    except InvalidOperation:
        return None
    if step <= 0:
        raise SweepError(f"range {text!r} needs a positive step")
    if stop < start:
        raise SweepError(f"range {text!r} has stop below start")
    count = int((stop - start) / step) + 1
    if count > MAX_RANGE_VALUES:
        raise SweepError(f"range {text!r} expands to {count} values (max {MAX_RANGE_VALUES})")
    return [format((start + step * index).normalize(), "f") for index in range(count)]


def parse_grid(items: Iterable[str]) -> dict[str, list[str]]:
    """Parse ``key=a,b,c`` lists and ``key=start:stop:step`` inclusive ranges."""
    grid: dict[str, list[str]] = {}
    for key, text in parse_assignments(items, flag="--grid").items():
        values = _range_values(text)
        if values is None:
            values = [value.strip() for value in text.split(",")]
        if not values or any(value == "" for value in values):
            raise SweepError(f"--grid {key} has an empty value")
        grid[key] = values
    if not grid:
        raise SweepError("sweep needs at least one --grid key=values")
    return grid


def expand_grid(
    grid: Mapping[str, Sequence[Any]],
    defaults: Mapping[str, Any],
    *,
    max_combos: int,
) -> list[dict[str, Any]]:
    """Type-check each grid value against the strategy defaults and return the product."""
    unknown = sorted(set(grid) - set(defaults))
    if unknown:
        available = ", ".join(sorted(defaults)) or "(none)"
        raise ParamError(f"Unknown param(s) {', '.join(unknown)}; available: {available}")
    axes: dict[str, list[Any]] = {}
    for name, raw_values in grid.items():
        coerced: list[Any] = []
        for raw in raw_values:
            value = coerce_param(name, raw, defaults[name])
            if value not in coerced:
                coerced.append(value)
        axes[name] = coerced
    total = math.prod(len(values) for values in axes.values())
    if total > max_combos:
        raise SweepError(
            f"grid has {total} combinations; raise --max-combos (currently {max_combos}) to run it"
        )
    names = list(axes)
    return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*axes.values())]


def build_context(spec: SweepSpec, inputs: RunInputs | None = None) -> SweepContext:
    preferences = Preferences.model_validate(spec.preferences_payload)
    if inputs is None:
        inputs = load_run_inputs(
            DuckDBStore(spec.data_dir),
            symbol=spec.symbol,
            start=spec.start,
            end=spec.end,
            options_source=spec.options_source,
            optionsdx_root=spec.optionsdx_root,
        )
    return SweepContext(
        spec=spec,
        strategy_cls=load_strategy_class(spec.strategy_path),
        inputs=inputs,
        preferences=preferences,
    )


def run_combo(context: SweepContext, index: int, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Run one combination; failures become an ``error`` row instead of raising."""
    spec = context.spec
    row: dict[str, Any] = {"combo": index}
    row.update({f"{PARAM_PREFIX}{name}": value for name, value in overrides.items()})
    try:
        strategy = context.strategy_cls()
        apply_params(strategy, {**spec.fixed_params, **overrides})
        summary = execute_run(
            strategy,
            context.inputs,
            context.preferences,
            cli_overrides={
                "start": spec.start,
                "end": spec.end,
                "symbol": context.inputs.symbol,
                "options_source": context.inputs.options_source,
            },
            run_dir_for=lambda config_hash: spec.strategy_runs_dir / f"{spec.run_prefix}-{config_hash}",
            write_html=spec.write_html,
            log_extra={"sweep_id": spec.sweep_id},
        )
    except Exception as exc:
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "run_dir": ""})
        return row
    row.update(summary.metrics)
    row.update({"status": "ok", "error": "", "run_dir": str(summary.run_dir)})
    return row


_WORKER_CONTEXT: SweepContext | None = None


def _init_worker(spec: SweepSpec) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = build_context(spec)


def _worker_run(index: int, overrides: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("sweep worker was not initialized")
    return run_combo(_WORKER_CONTEXT, index, overrides)


def run_sweep(
    context: SweepContext,
    combos: Sequence[Mapping[str, Any]],
    *,
    jobs: int = 1,
    fail_fast: bool = False,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Run ``combos`` in order (``jobs == 1``) or across spawned worker processes.

    Workers rebuild inputs once each from ``context.spec`` rather than receiving
    the chain per task. ``fail_fast`` stops scheduling after the first error row.
    """
    rows: list[dict[str, Any]] = []

    def record(row: dict[str, Any]) -> bool:
        rows.append(row)
        if on_result is not None:
            on_result(row)
        return fail_fast and row["status"] == "error"

    if jobs <= 1:
        for index, overrides in enumerate(combos):
            if record(run_combo(context, index, overrides)):
                break
    else:
        executor = ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(context.spec,),
        )
        try:
            pending: set[Future[dict[str, Any]]] = {
                executor.submit(_worker_run, index, dict(overrides)) for index, overrides in enumerate(combos)
            }
            stop = False
            while pending and not stop:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    if record(future.result()):
                        stop = True
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    frame = pd.DataFrame(rows)
    return frame.sort_values("combo").reset_index(drop=True) if not frame.empty else frame


def rank_results(results: pd.DataFrame, metric: str, *, minimize: bool = False) -> pd.DataFrame:
    """Successful rows ordered best-first by ``metric``."""
    if results.empty or metric not in results.columns:
        return results.iloc[0:0]
    ok = results[results["status"] == "ok"]
    return ok.sort_values(metric, ascending=minimize, kind="stable").reset_index(drop=True)


def write_sweep_summary(sweep_dir: Path, results: pd.DataFrame, manifest: Mapping[str, Any]) -> Path:
    """Write ``sweep.parquet`` (one row per combination) and ``sweep.json``."""
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(sweep_dir / SWEEP_RESULTS_FILE, index=False)
    payload = dict(manifest)
    payload["completed"] = int((results["status"] == "ok").sum()) if not results.empty else 0
    payload["failed"] = int((results["status"] == "error").sum()) if not results.empty else 0
    (sweep_dir / SWEEP_MANIFEST_FILE).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return sweep_dir
