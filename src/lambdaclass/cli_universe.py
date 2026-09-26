"""``lambdaclass universe``: weekly-options universe, earnings history, and reaction events."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TypeVar

import pandas as pd
import typer

from lambdaclass.config import Preferences, load_project_preferences
from lambdaclass.data_adapters.cboe_weeklys import download_cboe_weeklys, parse_cboe_weeklys
from lambdaclass.data_adapters.nasdaq_earnings import NasdaqEarningsAdapter
from lambdaclass.data_adapters.yfinance_adapter import YFinanceAdapter
from lambdaclass.earnings.events import apply_event_timing, build_earnings_events, summarize_events_by_symbol
from lambdaclass.earnings.history import merge_calendar_day, plan_calendar_days, symbol_earnings_frames
from lambdaclass.retry import with_retry
from lambdaclass.state import load_json, save_json
from lambdaclass.storage.duckdb_store import DuckDBStore

UNIVERSE_NAME = "weekly_options"
SUMMARY_NAME = f"{UNIVERSE_NAME}_summary"
CATEGORIES = ("equity", "etp", "all")
RETRY_DELAY_SECONDS = 2.0
COVERAGE_SLACK_DAYS = 5
Result = TypeVar("Result")

universe_app = typer.Typer(help="Weekly-options universe, earnings history, and earnings reaction events.")


def _repo_root() -> Path:
    return Path.cwd()


def _today() -> date:
    return date.today()


def _bars_adapter() -> YFinanceAdapter:
    return YFinanceAdapter()


def _earnings_adapter() -> NasdaqEarningsAdapter:
    return NasdaqEarningsAdapter()


def _download_weeklys() -> str:
    return download_cboe_weeklys()


def _project() -> tuple[Path, Preferences, DuckDBStore]:
    root = _repo_root()
    prefs = load_project_preferences(root)
    return root, prefs, DuckDBStore(root / prefs.paths.data_dir)


def _parse_date(value: str, option: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"Expected YYYY-MM-DD, got {value!r}", param_hint=option) from exc


def _check_category(category: str) -> str | None:
    if category not in CATEGORIES:
        raise typer.BadParameter(f"Choose one of {', '.join(CATEGORIES)}", param_hint="--category")
    return None if category == "all" else category


def _universe_symbols(store: DuckDBStore, category: str) -> list[str]:
    universe = store.read_universe(UNIVERSE_NAME, _check_category(category))
    if universe.empty:
        raise typer.BadParameter(
            "No weekly-options universe stored yet; run `lambdaclass universe fetch-weeklys`."
        )
    return universe["symbol"].astype(str).tolist()


def _retry(fn: Callable[[], Result]) -> Result:
    return with_retry(fn, retries=3, delay_seconds=RETRY_DELAY_SECONDS)


def _stamp() -> str:
    return datetime.now(tz=UTC).isoformat()


@universe_app.command("fetch-weeklys")
def fetch_weeklys(
    csv: str | None = typer.Option(None, help="Parse a saved Cboe CSV instead of downloading it"),
) -> None:
    """Download the Cboe "Available Weeklys" list and store it with a dated snapshot."""
    root, _, store = _project()
    if csv:
        path = Path(csv)
        if not path.is_file():
            raise typer.BadParameter(f"CSV not found: {path}", param_hint="--csv")
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    else:
        text = _retry(_download_weeklys)
    universe = parse_cboe_weeklys(text)
    if universe.empty:
        raise typer.BadParameter("No equity or ETP rows found; the Cboe file format may have changed.")
    as_of = _today()
    out_path = store.write_universe(UNIVERSE_NAME, universe, as_of=as_of)
    counts = universe["category"].value_counts().to_dict()
    markers_path = root / "state" / "universe_markers.json"
    markers = load_json(markers_path)
    markers[UNIVERSE_NAME] = {"as_of": as_of.isoformat(), "counts": counts, "updated_at": _stamp()}
    save_json(markers_path, markers)
    detail = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    typer.echo(f"Weekly-options universe: {len(universe)} symbols ({detail}) → {out_path}")


@universe_app.command("show")
def show_universe(
    category: str = typer.Option("all", help="equity, etp, or all"),
    limit: int = typer.Option(20, help="Rows to print (0 = all)"),
) -> None:
    """Print the stored weekly-options universe."""
    _, _, store = _project()
    universe = store.read_universe(UNIVERSE_NAME, _check_category(category))
    if universe.empty:
        typer.echo("No weekly-options universe stored yet; run `lambdaclass universe fetch-weeklys`.")
        return
    typer.echo(f"{len(universe)} symbols (list date {universe['list_date'].iloc[0]})")
    shown = universe if limit <= 0 else universe.head(limit)
    typer.echo(shown[["symbol", "category", "name"]].to_string(index=False))


def _covers(stored: tuple[str, str] | None, start: date, end: date) -> bool:
    if stored is None:
        return False
    slack = timedelta(days=COVERAGE_SLACK_DAYS)
    first, last = date.fromisoformat(stored[0]), date.fromisoformat(stored[1])
    return first <= start + slack and last >= end - slack


@universe_app.command("fetch-bars")
def fetch_bars(
    start: str = typer.Option(..., help="Start date in YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End date in YYYY-MM-DD (default: today)"),
    category: str = typer.Option("equity", help="equity, etp, or all"),
    skip_existing: bool = typer.Option(True, help="Skip symbols whose stored bars already cover the window"),
    pause: float = typer.Option(0.0, min=0.0, help="Seconds to wait between symbols"),
) -> None:
    """Fetch daily bars (yfinance) for every universe symbol; failures are reported, not fatal."""
    root, _, store = _project()
    start_dt = _parse_date(start, "--start")
    end_dt = _parse_date(end, "--end") if end else _today()
    symbols = _universe_symbols(store, category)
    adapter = _bars_adapter()
    fetched = skipped = 0
    failures: dict[str, str] = {}
    markers_path = root / "state" / "fetch_markers.json"
    markers = load_json(markers_path)
    for position, symbol in enumerate(symbols, start=1):
        if skip_existing and _covers(store.bar_date_range(symbol), start_dt, end_dt):
            skipped += 1
            continue
        try:
            bars = _retry(partial(adapter.get_stock_bars, symbol, start_dt, end_dt))
        except Exception as exc:
            failures[symbol] = str(exc)
            continue
        if bars.empty:
            failures[symbol] = "no bars returned"
            continue
        bars_path = store.write_bars(symbol, bars)
        markers[symbol] = {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "bars_path": str(bars_path),
            "options_path": None,
            "updated_at": _stamp(),
        }
        fetched += 1
        if position % 50 == 0:
            typer.echo(f"  {position}/{len(symbols)} symbols processed")
        if pause:
            time.sleep(pause)
    save_json(markers_path, markers)
    typer.echo(f"Bars: fetched={fetched} skipped={skipped} failed={len(failures)} of {len(symbols)} symbols")
    for symbol, reason in sorted(failures.items()):
        typer.secho(f"  {symbol}: {reason}", fg=typer.colors.YELLOW, err=True)


@universe_app.command("fetch-earnings")
def fetch_earnings(
    start: str = typer.Option(..., help="First calendar day in YYYY-MM-DD"),
    end: str | None = typer.Option(
        None, help="Last calendar day (default: today + 28 days, for upcoming reports)"
    ),
    category: str = typer.Option(
        "all", help="Universe symbols to write per-symbol files for: equity, etp, or all"
    ),
    refresh_days: int = typer.Option(7, min=0, help="Refetch cached days this recent (and all future days)"),
    include_weekends: bool = typer.Option(False, help="Also query Saturdays and Sundays"),
    pause: float = typer.Option(0.2, min=0.0, help="Seconds to wait between calendar requests"),
) -> None:
    """Fetch the Nasdaq earnings calendar day by day, then write per-symbol earnings files.

    Every day is cached under ``data/earnings/calendar/`` (all companies, not only
    the universe), so reruns only request missing, recent, and upcoming days.
    """
    root, _, store = _project()
    today = _today()
    start_dt = _parse_date(start, "--start")
    end_dt = _parse_date(end, "--end") if end else today + timedelta(days=28)
    symbols = _universe_symbols(store, category)
    try:
        days = plan_calendar_days(
            start_dt,
            end_dt,
            cached=store.cached_earnings_days(),
            today=today,
            refresh_days=refresh_days,
            include_weekends=include_weekends,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    adapter = _earnings_adapter()
    failures: dict[str, str] = {}
    typer.echo(f"Requesting {len(days)} calendar days from Nasdaq ({start_dt} to {end_dt})")
    for position, day in enumerate(days, start=1):
        try:
            fresh = _retry(partial(adapter.get_earnings_day, day))
        except Exception as exc:
            failures[day.isoformat()] = str(exc)
            continue
        store.write_earnings_day(day, merge_calendar_day(store.read_earnings_day(day), fresh))
        if position % 50 == 0:
            typer.echo(f"  {position}/{len(days)} days fetched")
        if pause and position < len(days):
            time.sleep(pause)

    calendar = store.read_earnings_calendar(
        start=start_dt.isoformat(), end=end_dt.isoformat(), symbols=symbols
    )
    frames = symbol_earnings_frames(calendar, symbols)
    for symbol, frame in frames.items():
        store.write_earnings(symbol, frame)
    markers_path = root / "state" / "earnings_fetch_markers.json"
    markers = load_json(markers_path)
    markers[UNIVERSE_NAME] = {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "days_requested": len(days),
        "days_failed": sorted(failures),
        "symbols_with_events": len(frames),
        "source": "nasdaq",
        "updated_at": _stamp(),
    }
    save_json(markers_path, markers)
    typer.echo(
        f"Earnings: {len(calendar)} universe reports across {len(frames)} of {len(symbols)} symbols; "
        f"{len(days) - len(failures)}/{len(days)} days fetched"
    )
    for day_text, reason in sorted(failures.items()):
        typer.secho(f"  {day_text}: {reason}", fg=typer.colors.YELLOW, err=True)


def _timing_agreement(events: pd.DataFrame) -> str:
    checked = events[events["vendor_timing"].notna() & events["inferred_timing"].notna()]
    if checked.empty:
        return "no vendor timings to check inference against"
    rate = float((checked["vendor_timing"] == checked["inferred_timing"]).mean())
    return f"gap inference matches vendor timing on {rate:.0%} of {len(checked)} events"


@universe_app.command("build-events")
def build_events(
    category: str = typer.Option("equity", help="equity, etp, or all"),
    start: str | None = typer.Option(None, help="Only events on or after YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="Only events on or before YYYY-MM-DD"),
    min_events: int = typer.Option(4, min=1, help="Minimum events for a symbol to appear in the summary"),
    top: int = typer.Option(15, min=0, help="Summary rows to print"),
) -> None:
    """Join cached earnings dates with stored bars into reaction events and per-symbol statistics."""
    _, _, store = _project()
    if start:
        _parse_date(start, "--start")
    if end:
        _parse_date(end, "--end")
    symbols = _universe_symbols(store, category)
    calendar = store.read_earnings_calendar(start=start, end=end, symbols=symbols)
    if calendar.empty:
        raise typer.BadParameter(
            "No cached earnings for the universe; run `lambdaclass universe fetch-earnings`."
        )
    reporting_symbols = sorted(set(calendar["symbol"]))
    bars_by_symbol = store.read_bars_many(reporting_symbols)
    events = build_earnings_events(calendar, bars_by_symbol)
    if events.empty:
        raise typer.BadParameter(
            "No events had bars around them; run `lambdaclass universe fetch-bars` first."
        )
    events_path = store.write_earnings_events(UNIVERSE_NAME, events)
    summary = summarize_events_by_symbol(events, min_events=min_events)
    summary_path = store.write_earnings_events(SUMMARY_NAME, summary)
    earnings = store.read_earnings_many(sorted(set(events["symbol"])))
    updated = apply_event_timing(earnings, events)
    if not updated.empty:
        before = earnings.reindex(columns=["timing", "timing_source"]).fillna("")
        changed = updated[["timing", "timing_source"]].fillna("").ne(before).any(axis=1)
        for symbol in updated.loc[changed, "symbol"].unique():
            store.write_earnings(str(symbol), updated[updated["symbol"] == symbol])
    no_bars = len(reporting_symbols) - len(bars_by_symbol)
    typer.echo(
        f"Events: {len(events)} across {events['symbol'].nunique()} symbols → {events_path}\n"
        f"Summary: {len(summary)} symbols with >= {min_events} events → {summary_path}\n"
        f"Timing: {events['timing_source'].value_counts().to_dict()}; {_timing_agreement(events)}"
    )
    if no_bars:
        typer.secho(
            f"  {no_bars} symbols have earnings but no stored bars.", fg=typer.colors.YELLOW, err=True
        )
    if top and not summary.empty:
        shown = summary.head(top)[
            ["symbol", "events", "median_abs_move", "mean_abs_move", "up_rate", "mean_drift_5d", "beat_rate"]
        ]
        typer.echo(shown.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
