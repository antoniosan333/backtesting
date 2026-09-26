"""Weekly-options universe storage, earnings history planning, reaction events, and the CLI."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from lambdaclass import cli, cli_universe
from lambdaclass.data_adapters.nasdaq_earnings import empty_earnings_day, parse_nasdaq_earnings_rows
from lambdaclass.data_adapters.yfinance_adapter import YAHOO_EARNINGS_COLUMNS, parse_yahoo_earnings_dates
from lambdaclass.earnings.events import (
    BarSeries,
    apply_event_timing,
    build_earnings_events,
    infer_timing,
    summarize_events_by_symbol,
)
from lambdaclass.earnings.history import (
    combine_earnings_sources,
    merge_calendar_day,
    plan_calendar_days,
    symbol_earnings_frames,
)
from lambdaclass.retry import with_retry
from lambdaclass.storage.duckdb_store import DuckDBStore

FIXTURES = Path(__file__).parent / "fixtures"
UNIVERSE_CSV = (
    "List of Available Weekly Options\n"
    '"SPXW (MON)","09/28/26"\n\n'
    "Available Weeklys - Exchange Traded Products (ETFs and ETNs)\n"
    '"SPY","SPDR S&P 500"\n\n'
    "Available Weeklys - Equity\n"
    '"CRM","SALESFORCE"\n"NVDA","NVIDIA"\n"PBR.A","PETROBRAS A"\n'
)


def _nasdaq_day(day: str) -> pd.DataFrame:
    payload = json.loads((FIXTURES / "nasdaq" / "earnings_days.json").read_text(encoding="utf-8"))
    rows = payload.get(day, {"data": {"rows": None}})["data"]["rows"] or []
    return parse_nasdaq_earnings_rows(rows, date.fromisoformat(day))


def _yahoo_raw(
    stamps: list[str],
    *,
    estimates: list[float] | None = None,
    actuals: list[float] | None = None,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """A frame shaped like ``yfinance.Ticker.get_earnings_dates`` output."""
    index = pd.DatetimeIndex(pd.to_datetime(stamps), name="Earnings Date").tz_localize(tz)
    nan = [float("nan")] * len(stamps)
    return pd.DataFrame(
        {"EPS Estimate": estimates or nan, "Reported EPS": actuals or nan, "Surprise(%)": nan}, index=index
    )


def _synthetic_bars(start: str, end: str, *, gaps: dict[str, float]) -> pd.DataFrame:
    """Flat-ish daily bars; ``gaps`` maps a date to an overnight jump applied at that open."""
    dates = pd.bdate_range(start, end)
    closes: list[float] = []
    opens: list[float] = []
    price = 100.0
    for position, stamp in enumerate(dates):
        day = stamp.date().isoformat()
        wiggle = 0.004 if position % 2 else -0.004
        opened = price * (1.0 + gaps.get(day, 0.0))
        price = opened * (1.0 + wiggle)
        opens.append(opened)
        closes.append(price)
    return pd.DataFrame(
        {
            "date": [stamp.date().isoformat() for stamp in dates],
            "open": opens,
            "high": np.maximum(opens, closes),
            "low": np.minimum(opens, closes),
            "close": closes,
            "volume": [1_000.0] * len(dates),
            "dividends": [0.0] * len(dates),
        }
    )


# --- storage -------------------------------------------------------------------------------


def test_universe_roundtrip_keeps_dated_snapshots(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path)
    first = pd.DataFrame(
        {"symbol": ["AAPL", "SPY"], "name": ["Apple", "SPDR"], "category": ["equity", "etp"]}
    )
    store.write_universe("weekly_options", first, as_of=date(2026, 9, 1))
    store.write_universe("weekly_options", first.head(1), as_of=date(2026, 9, 26))

    current = store.read_universe("weekly_options")
    assert current["symbol"].tolist() == ["AAPL"]
    assert current["list_date"].tolist() == ["2026-09-26"]
    assert store.read_universe("weekly_options", "etp").empty
    snapshots = sorted((tmp_path / "universe" / "snapshots" / "weekly_options").glob("*.parquet"))
    assert [path.stem for path in snapshots] == ["2026-09-01", "2026-09-26"]


@pytest.mark.parametrize("name", ["../escape", "Upper", "", "a/b", "x" * 65])
def test_dataset_names_reject_path_tricks(tmp_path: Path, name: str) -> None:
    store = DuckDBStore(tmp_path)
    with pytest.raises(ValueError, match="Invalid dataset name"):
        store.read_universe(name)
    with pytest.raises(ValueError, match="Invalid dataset name"):
        store.write_earnings_events(name, pd.DataFrame())


def test_earnings_calendar_cache_and_filtered_read(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path)
    store.write_earnings_day(date(2025, 2, 26), _nasdaq_day("2025-02-26"))
    store.write_earnings_day(date(2026, 10, 15), _nasdaq_day("2026-10-15"))
    store.write_earnings_day(date(2026, 9, 26), empty_earnings_day())

    assert store.cached_earnings_days() == {date(2025, 2, 26), date(2026, 10, 15), date(2026, 9, 26)}
    assert store.read_earnings_day(date(2026, 9, 26)) is not None
    assert store.read_earnings_day(date(2020, 1, 1)) is None
    everything = store.read_earnings_calendar()
    assert len(everything) == 6
    subset = store.read_earnings_calendar(start="2026-01-01", symbols=["schw", "NVDA"])
    assert subset["symbol"].tolist() == ["SCHW"]
    assert store.read_earnings_calendar(symbols=[]).empty


def test_bar_date_range(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path)
    assert store.bar_date_range("AAPL") is None
    store.write_bars("AAPL", _synthetic_bars("2025-01-02", "2025-01-10", gaps={}))
    assert store.bar_date_range("AAPL") == ("2025-01-02", "2025-01-10")


# --- history planning ----------------------------------------------------------------------


def test_plan_calendar_days_skips_weekends_and_cached_history() -> None:
    days = plan_calendar_days(
        date(2026, 9, 14),
        date(2026, 9, 30),
        cached={date(2026, 9, 14), date(2026, 9, 15), date(2026, 9, 22), date(2026, 9, 28)},
        today=date(2026, 9, 26),
        refresh_days=5,
    )
    assert date(2026, 9, 14) not in days and date(2026, 9, 15) not in days
    assert date(2026, 9, 16) in days
    assert date(2026, 9, 22) in days  # within refresh window
    assert date(2026, 9, 28) in days  # future days are always refreshed
    assert all(day.weekday() < 5 for day in days)


def test_plan_calendar_days_validates_inputs() -> None:
    with pytest.raises(ValueError, match="before start"):
        plan_calendar_days(date(2026, 1, 2), date(2026, 1, 1), cached=(), today=date(2026, 1, 1))
    weekend = plan_calendar_days(
        date(2026, 9, 26), date(2026, 9, 27), cached=(), today=date(2026, 9, 1), include_weekends=True
    )
    assert weekend == [date(2026, 9, 26), date(2026, 9, 27)]


def test_merge_calendar_day_keeps_known_timing_and_ignores_empty_refetch() -> None:
    upcoming = _nasdaq_day("2026-10-15")
    refetched = upcoming.assign(timing="unknown", eps_actual=[1.7, 0.7])

    merged = merge_calendar_day(upcoming, refetched)
    assert merged["timing"].tolist() == ["BMO", "AMC"]
    assert merged["eps_actual"].tolist() == [1.7, 0.7]
    assert merge_calendar_day(upcoming, empty_earnings_day()) is upcoming
    assert merge_calendar_day(None, refetched) is refetched


def test_symbol_earnings_frames_use_canonical_schema() -> None:
    calendar = pd.concat([_nasdaq_day("2025-02-26"), _nasdaq_day("2026-10-15")], ignore_index=True)
    frames = symbol_earnings_frames(calendar, ["nvda", "SCHW", "MISSING"])

    assert sorted(frames) == ["NVDA", "SCHW"]
    nvda = frames["NVDA"]
    assert list(nvda.columns[:5]) == ["symbol", "earnings_date", "timing", "source", "fetched_at"]
    assert nvda["timing_source"].tolist() == ["none"]
    assert frames["SCHW"]["timing_source"].tolist() == ["vendor"]


def test_parse_yahoo_earnings_dates_maps_announcement_times() -> None:
    raw = _yahoo_raw(
        ["2025-01-15 06:00", "2025-01-16 12:30", "2025-01-29 16:05", "2025-04-22 20:00", "2025-04-22 20:00"],
        actuals=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    frame = parse_yahoo_earnings_dates(raw)

    assert list(frame.columns) == YAHOO_EARNINGS_COLUMNS
    assert frame["earnings_date"].tolist() == ["2025-01-15", "2025-01-16", "2025-01-29", "2025-04-23"]
    # 20:00 New York in April is midnight UTC: Yahoo's "date known, time unknown" marker.
    assert frame["timing"].tolist() == ["BMO", "unknown", "AMC", "unknown"]
    assert frame["announce_time"].tolist() == ["06:00", "12:30", "16:05", None]
    assert frame["eps_actual"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert parse_yahoo_earnings_dates(None).empty
    assert list(parse_yahoo_earnings_dates(None).columns) == YAHOO_EARNINGS_COLUMNS


def test_combine_earnings_sources_prefers_yahoo_timing_and_nasdaq_eps() -> None:
    nasdaq = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "CCC"],
            "earnings_date": ["2025-01-29", "2025-04-23", "2025-02-05", "2025-03-03"],
            "timing": ["unknown", "unknown", "BMO", "unknown"],
            "eps_actual": [1.0, np.nan, 2.0, 3.0],
            "eps_estimate": [0.9, 1.1, 1.9, 2.9],
            "surprise_pct": [11.0, np.nan, 5.0, 3.0],
        }
    )
    yahoo = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA", "BBB"],
            "earnings_date": ["2024-10-23", "2025-01-29", "2025-04-24", "2025-02-05"],
            "timing": ["AMC", "AMC", "BMO", "unknown"],
            "announce_time": ["16:05", "16:05", "07:00", None],
            "eps_actual": [0.8, 0.99, 1.2, 2.1],
            "eps_estimate": [0.7, 0.9, 1.0, 1.8],
            "surprise_pct": [14.0, 10.0, 20.0, 16.0],
        }
    )
    combined = combine_earnings_sources(nasdaq, yahoo).set_index(["symbol", "earnings_date"])

    assert list(combined.index) == [
        ("AAA", "2024-10-23"),
        ("AAA", "2025-01-29"),
        ("AAA", "2025-04-24"),
        ("BBB", "2025-02-05"),
        ("CCC", "2025-03-03"),
    ]
    assert combined.loc[("AAA", "2024-10-23"), "source"] == "yahoo"
    paired = combined.loc[("AAA", "2025-01-29")]
    assert (paired["timing"], paired["timing_vendor"], paired["source"]) == ("AMC", "yahoo", "nasdaq+yahoo")
    assert paired["eps_actual"] == 1.0  # Nasdaq EPS wins
    moved = combined.loc[("AAA", "2025-04-24")]  # Yahoo knows the time, so its date wins
    assert moved["timing"] == "BMO" and moved["eps_actual"] == 1.2  # filled from Yahoo
    bbb = combined.loc[("BBB", "2025-02-05")]
    assert (bbb["timing"], bbb["timing_vendor"]) == ("BMO", "nasdaq")
    assert combined.loc[("CCC", "2025-03-03"), "source"] == "nasdaq"
    assert combine_earnings_sources(nasdaq.head(0), yahoo.head(0)).empty
    assert len(combine_earnings_sources(nasdaq, yahoo.head(0))) == 4


def test_write_earnings_replace_sources_keeps_other_sources(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path)
    store.write_earnings("AAA", pd.DataFrame({"earnings_date": ["2020-01-01"], "source": ["csv"]}))
    store.write_earnings("AAA", pd.DataFrame({"earnings_date": ["2025-01-28"], "source": ["nasdaq"]}))
    store.write_earnings(
        "AAA",
        pd.DataFrame({"earnings_date": ["2025-01-29"], "source": ["nasdaq+yahoo"]}),
        replace_sources={"nasdaq"},
    )
    assert store.read_earnings("AAA")["earnings_date"].tolist() == ["2020-01-01", "2025-01-29"]


# --- events --------------------------------------------------------------------------------


def test_infer_timing_from_the_larger_gap() -> None:
    bars = BarSeries.from_frame(
        _synthetic_bars("2025-02-03", "2025-02-28", gaps={"2025-02-12": 0.08, "2025-02-21": -0.06})
    )
    bmo_index = int(np.searchsorted(bars.dates, "2025-02-12"))
    amc_index = int(np.searchsorted(bars.dates, "2025-02-20"))

    assert infer_timing(bars, bmo_index)[0] == "BMO"
    timing, confidence = infer_timing(bars, amc_index)
    assert timing == "AMC" and confidence > 0.9
    assert infer_timing(bars, 0) == (None, pytest.approx(float("nan"), nan_ok=True))


def test_build_earnings_events_measures_the_reaction_session() -> None:
    bars = _synthetic_bars("2024-12-02", "2025-04-30", gaps={"2025-02-26": 0.10, "2025-02-27": -0.08})
    calendar = pd.DataFrame(
        {
            "symbol": ["CRM", "NVDA", "SCHW", "NVDA"],
            "earnings_date": ["2025-02-26", "2025-02-26", "2025-02-26", "2026-10-15"],
            "timing": ["unknown", "unknown", "BMO", "AMC"],
            "eps_actual": [2.22, 0.85, np.nan, np.nan],
            "eps_estimate": [1.95, 0.90, 1.0, 0.7],
            "surprise_pct": [13.85, -5.0, np.nan, np.nan],
        }
    )
    events = build_earnings_events(calendar, {"CRM": bars, "NVDA": bars, "SCHW": bars}).set_index("symbol")

    assert sorted(events.index) == ["CRM", "NVDA", "SCHW"]  # the 2026 NVDA report has no bars yet
    crm = events.loc["CRM"]
    assert crm["timing"] == "BMO" and crm["timing_source"] == "inferred_gap"
    assert crm["reaction_date"] == "2025-02-26" and crm["pre_date"] == "2025-02-25"
    assert crm["gap_return"] == pytest.approx(0.10, abs=1e-9)
    assert bool(crm["beat"]) is True
    assert crm["abs_reaction"] == pytest.approx(abs(crm["reaction_return"]))
    assert abs(crm["move_sigma"]) > 10
    assert crm["volume_ratio"] == pytest.approx(1.0)
    schw = events.loc["SCHW"]
    assert schw["timing_source"] == "vendor" and schw["vendor_timing"] == "BMO"
    assert pd.isna(schw["beat"])
    drift = bars.set_index("date")["close"]
    assert crm["drift_5d"] == pytest.approx(
        drift.iloc[drift.index.get_loc("2025-02-26") + 5] / crm["reaction_close"] - 1
    )


def test_symbol_vote_overrides_low_confidence_gap_guesses() -> None:
    # Three clear AMC reactions, then a quiet report whose larger gap happens to be on the report day.
    gaps = {
        "2024-10-24": 0.08,
        "2025-01-30": -0.07,
        "2025-04-24": 0.06,
        "2025-07-23": 0.012,
        "2025-07-24": 0.008,
    }
    bars = _synthetic_bars("2024-06-03", "2025-09-30", gaps=gaps)
    calendar = pd.DataFrame(
        {
            "symbol": ["AAA"] * 5,
            "earnings_date": ["2024-10-23", "2025-01-29", "2025-04-23", "2025-07-23", "2026-10-22"],
            "timing": ["unknown"] * 4 + ["AMC"],
        }
    )
    events = build_earnings_events(calendar, {"AAA": bars}).set_index("earnings_date")

    quiet = events.loc["2025-07-23"]
    assert quiet["inferred_timing"] == "BMO" and quiet["timing_confidence"] < 0.9
    assert quiet["timing"] == "AMC" and quiet["timing_source"] == "inferred_symbol"
    assert quiet["reaction_date"] == "2025-07-24"
    assert set(events["symbol_timing"]) == {"AMC"}
    assert events.loc["2024-10-23", "timing_source"] == "inferred_gap"


def test_symbol_vote_needs_a_clear_majority() -> None:
    bars = _synthetic_bars("2024-06-03", "2025-06-30", gaps={"2024-10-23": 0.06, "2025-01-30": -0.06})
    calendar = pd.DataFrame(
        {"symbol": ["AAA"] * 2, "earnings_date": ["2024-10-23", "2025-01-29"], "timing": ["unknown"] * 2}
    )
    events = build_earnings_events(calendar, {"AAA": bars})

    assert events["symbol_timing"].isna().all()
    assert events["timing"].tolist() == ["BMO", "AMC"]
    assert set(events["timing_source"]) == {"inferred_gap"}


def test_weekend_report_reacts_on_next_session() -> None:
    bars = _synthetic_bars("2024-12-02", "2025-04-30", gaps={"2025-03-03": 0.05})
    calendar = pd.DataFrame({"symbol": ["BRK-B"], "earnings_date": ["2025-03-01"], "timing": ["unknown"]})
    event = build_earnings_events(calendar, {"BRK-B": bars}).iloc[0]

    assert event["timing_source"] == "non_trading_day"
    assert event["reaction_date"] == "2025-03-03" and event["pre_date"] == "2025-02-28"
    assert event["gap_return"] == pytest.approx(0.05, abs=1e-9)
    assert math.isnan(event["eps_actual"])


def test_summary_and_timing_backfill() -> None:
    bars = _synthetic_bars(
        "2024-06-03", "2025-06-30", gaps={"2024-10-24": 0.06, "2025-01-30": -0.04, "2025-04-24": 0.02}
    )
    calendar = pd.DataFrame(
        {
            "symbol": ["AAA"] * 3,
            "earnings_date": ["2024-10-23", "2025-01-29", "2025-04-23"],
            "timing": ["unknown"] * 3,
            "eps_actual": [1.0, 1.0, 1.0],
            "eps_estimate": [0.9, 1.1, 0.9],
        }
    )
    events = build_earnings_events(calendar, {"AAA": bars})
    assert events["timing"].tolist() == ["AMC", "AMC", "AMC"]

    summary = summarize_events_by_symbol(events)
    row = summary.iloc[0]
    assert row["events"] == 3
    assert row["up_rate"] == pytest.approx(2 / 3)
    assert row["beat_rate"] == pytest.approx(2 / 3)
    assert row["bmo_share"] == 0.0
    assert summarize_events_by_symbol(events, min_events=4).empty

    earnings = symbol_earnings_frames(calendar.assign(source="nasdaq", fetched_at="x"), ["AAA"])["AAA"]
    filled = apply_event_timing(earnings, events)
    assert filled["timing"].tolist() == ["AMC", "AMC", "AMC"]
    assert filled["timing_source"].tolist() == ["inferred_gap"] * 3


def test_apply_event_timing_refreshes_inferences_but_keeps_vendor_rows() -> None:
    earnings = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "earnings_date": ["2025-01-29", "2025-04-23", "2025-07-23"],
            "timing": ["BMO", "AMC", "BMO"],
            "timing_source": ["inferred_gap", "vendor", None],
        }
    )
    events = pd.DataFrame(
        {
            "symbol": ["AAA"] * 3,
            "earnings_date": ["2025-01-29", "2025-04-23", "2025-07-23"],
            "timing": ["AMC", "BMO", "AMC"],
            "timing_source": ["inferred_symbol", "inferred_gap", "inferred_gap"],
        }
    )
    updated = apply_event_timing(earnings, events)

    assert updated["timing"].tolist() == ["AMC", "AMC", "BMO"]
    assert updated["timing_source"].tolist() == ["inferred_symbol", "vendor", "vendor"]


def test_batch_readers_use_one_query(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path)
    store.write_bars("AAA", _synthetic_bars("2025-01-02", "2025-01-08", gaps={}))
    store.write_bars("BBB", _synthetic_bars("2025-01-06", "2025-01-08", gaps={}).drop(columns="dividends"))
    store.write_earnings("AAA", pd.DataFrame({"earnings_date": ["2025-01-29"], "timing": ["AMC"]}))

    bars = store.read_bars_many(["AAA", "BBB", "MISSING"])
    assert sorted(bars) == ["AAA", "BBB"]
    assert bars["BBB"]["date"].tolist() == ["2025-01-06", "2025-01-07", "2025-01-08"]
    assert bars["BBB"]["dividends"].tolist() == [0.0, 0.0, 0.0]
    assert store.read_bars_many(["MISSING"]) == {}
    assert store.read_earnings_many(["AAA", "BBB"])["symbol"].tolist() == ["AAA"]


def test_with_retry_retries_then_raises() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise OSError("temporary")
        return "ok"

    assert with_retry(flaky, retries=3, delay_seconds=1.5, sleep=sleeps.append) == "ok"
    assert sleeps == [1.5, 3.0]
    with pytest.raises(OSError, match="always"):
        with_retry(lambda: (_ for _ in ()).throw(OSError("always")), retries=2, sleep=sleeps.append)


# --- CLI -----------------------------------------------------------------------------------


class _FakeNasdaq:
    def __init__(self) -> None:
        self.requested: list[date] = []

    def get_earnings_day(self, day: date) -> pd.DataFrame:
        self.requested.append(day)
        return _nasdaq_day(day.isoformat())


class _FakeBars:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def get_stock_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        self.requested.append(symbol)
        if symbol == "PBR-A":
            raise OSError("vendor down")
        gaps = {"CRM": {"2025-02-26": 0.09}, "NVDA": {"2025-02-27": -0.07}}.get(symbol, {})
        return _synthetic_bars(start.isoformat(), end.isoformat(), gaps=gaps)

    def get_earnings_dates(self, symbol: str) -> pd.DataFrame:
        self.requested.append(f"earnings:{symbol}")
        if symbol == "NVDA":
            return parse_yahoo_earnings_dates(
                _yahoo_raw(
                    ["2024-11-20 16:20", "2025-02-26 16:20"], estimates=[0.75, 0.84], actuals=[0.81, 0.89]
                )
            )
        return parse_yahoo_earnings_dates(None)


@pytest.fixture()
def universe_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_universe, "_today", lambda: date(2025, 3, 3))
    monkeypatch.setattr(cli_universe, "_download_weeklys", lambda: UNIVERSE_CSV)
    monkeypatch.setattr(cli_universe, "RETRY_DELAY_SECONDS", 0.0)
    return tmp_path


def test_universe_cli_end_to_end(universe_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nasdaq, bars = _FakeNasdaq(), _FakeBars()
    monkeypatch.setattr(cli_universe, "_earnings_adapter", lambda: nasdaq)
    monkeypatch.setattr(cli_universe, "_bars_adapter", lambda: bars)
    runner = CliRunner()

    result = runner.invoke(cli.app, ["universe", "fetch-weeklys"])
    assert result.exit_code == 0, result.output
    assert "4 symbols (equity=3, etp=1)" in result.output
    shown = runner.invoke(cli.app, ["universe", "show", "--category", "equity"])
    assert "PBR-A" in shown.output and "SPY" not in shown.output

    result = runner.invoke(
        cli.app,
        ["universe", "fetch-earnings", "--start", "2025-02-24", "--end", "2025-02-28", "--pause", "0"],
    )
    assert result.exit_code == 0, result.output
    assert len(nasdaq.requested) == 5
    store = DuckDBStore(universe_project / "data")
    assert "SNOW" in set(store.read_earnings_day(date(2025, 2, 26))["symbol"])

    nasdaq.requested.clear()
    rerun = runner.invoke(
        cli.app,
        ["universe", "fetch-earnings", "--start", "2025-02-17", "--end", "2025-02-28", "--refresh-days", "3"],
    )
    assert rerun.exit_code == 0, rerun.output
    # Mon 17th..Fri 21st are new; the 28th is within 3 days of "today" (2025-03-03) and is refetched.
    assert [day.isoformat() for day in nasdaq.requested] == [
        "2025-02-17",
        "2025-02-18",
        "2025-02-19",
        "2025-02-20",
        "2025-02-21",
        "2025-02-28",
    ]

    result = runner.invoke(cli.app, ["universe", "fetch-bars", "--start", "2024-12-02"])
    assert result.exit_code == 0, result.output
    assert "fetched=2 skipped=0 failed=1" in result.output
    bars.requested.clear()
    result = runner.invoke(cli.app, ["universe", "fetch-bars", "--start", "2024-12-02"])
    assert "fetched=0 skipped=2 failed=1" in result.output
    assert bars.requested == ["PBR-A"] * 3  # retried, never skipped

    bars.requested.clear()
    result = runner.invoke(cli.app, ["universe", "fetch-yahoo-earnings"])
    assert result.exit_code == 0, result.output
    assert "fetched=3 (no earnings: 2) skipped=0 failed=0" in result.output
    assert bars.requested == ["earnings:CRM", "earnings:NVDA", "earnings:PBR-A"]
    result = runner.invoke(cli.app, ["universe", "fetch-yahoo-earnings"])
    assert "fetched=0 (no earnings: 0) skipped=3" in result.output

    store.write_earnings(
        "CRM", pd.DataFrame({"earnings_date": ["2019-06-04"], "timing": ["AMC"], "source": ["csv"]})
    )
    result = runner.invoke(cli.app, ["universe", "build-events", "--min-events", "1"])
    assert result.exit_code == 0, result.output
    events = store.read_earnings_events("weekly_options").set_index("symbol")
    assert events.loc["CRM", "timing"] == "BMO"
    assert events.loc["CRM", "timing_source"] == "inferred_gap"
    nvda = events.loc["NVDA"]
    assert (nvda["timing"], nvda["timing_source"], nvda["timing_vendor"]) == ("AMC", "vendor", "yahoo")
    assert nvda["announce_time"] == "16:20" and nvda["eps_actual"] == pytest.approx(0.85)
    assert "PBR-A" not in events.index
    assert set(store.read_earnings_events("weekly_options_summary")["symbol"]) == {"CRM", "NVDA"}
    nvda_file = store.read_earnings("NVDA")
    assert nvda_file["earnings_date"].tolist() == ["2024-11-20", "2025-02-26"]
    assert nvda_file["source"].tolist() == ["yahoo", "nasdaq+yahoo"]
    assert nvda_file["timing_source"].tolist() == ["vendor", "vendor"]
    crm_file = store.read_earnings("CRM")
    assert crm_file["source"].tolist() == ["csv", "nasdaq"]  # user-imported rows survive
    assert crm_file["timing_source"].tolist()[-1] == "inferred_gap"
    assert store.read_earnings("PBR-A")["eps_actual"].tolist() == [0.49]
    assert store.read_earnings("SNOW").empty  # not in the universe, but cached in the day file
    assert "Per-symbol earnings files updated: 3" in result.output
    assert "1 symbols have earnings but no stored bars" in result.output


def test_universe_cli_requires_a_universe(universe_project: Path) -> None:
    result = CliRunner().invoke(cli.app, ["universe", "fetch-earnings", "--start", "2025-02-24"])
    assert result.exit_code != 0
    assert "fetch-weeklys" in result.output


def test_universe_cli_rejects_bad_inputs(universe_project: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli.app, ["universe", "fetch-weeklys"])
    assert runner.invoke(cli.app, ["universe", "show", "--category", "bonds"]).exit_code != 0
    assert runner.invoke(cli.app, ["universe", "fetch-bars", "--start", "03/01/2025"]).exit_code != 0
    bad_range = runner.invoke(
        cli.app, ["universe", "fetch-earnings", "--start", "2025-03-01", "--end", "2025-02-01"]
    )
    assert bad_range.exit_code != 0


def test_fetch_weeklys_refuses_to_store_an_unrecognized_file(
    universe_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_universe, "_download_weeklys", lambda: "<html>maintenance</html>")
    result = CliRunner().invoke(cli.app, ["universe", "fetch-weeklys"])
    assert result.exit_code != 0
    assert not (universe_project / "data" / "universe" / "weekly_options.parquet").exists()


def test_today_default_end_covers_upcoming_reports(
    universe_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nasdaq = _FakeNasdaq()
    monkeypatch.setattr(cli_universe, "_earnings_adapter", lambda: nasdaq)
    runner = CliRunner()
    runner.invoke(cli.app, ["universe", "fetch-weeklys"])
    result = runner.invoke(cli.app, ["universe", "fetch-earnings", "--start", "2025-03-03", "--pause", "0"])
    assert result.exit_code == 0, result.output
    assert max(nasdaq.requested) >= date(2025, 3, 3) + timedelta(days=26)
