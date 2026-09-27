"""Earnings calendar math and normalization."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from lambdaclass.earnings import calendar as cal
from lambdaclass.storage.duckdb_store import DuckDBStore


def _earn(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "earnings_date": d,
                "timing": t,
                "source": "test",
                "fetched_at": "2026-01-01T00:00:00Z",
            }
            for d, t in rows
        ]
    )


def test_normalize_timing_aliases() -> None:
    assert cal.normalize_timing("BMO") == "BMO"
    assert cal.normalize_timing("After Market Close") == "AMC"
    assert cal.normalize_timing("") == "unknown"
    assert cal.normalize_timing("TAS") == "AMC"


def test_days_to_next_earnings() -> None:
    e = _earn([("2026-02-10", "AMC"), ("2026-05-10", "BMO")])
    assert cal.days_to_next_earnings("2026-02-01", e) == 9
    assert cal.days_to_next_earnings("2026-02-10", e) == 0
    assert cal.days_to_next_earnings("2026-06-01", e) is None


def test_days_since_bmo_same_day() -> None:
    e = _earn([("2026-02-10", "BMO")])
    assert cal.days_since_last_earnings("2026-02-10", e) == 0
    assert cal.days_since_last_earnings("2026-02-11", e) == 1


def test_days_since_amc_same_day_is_none() -> None:
    e = _earn([("2026-02-10", "AMC")])
    assert cal.days_since_last_earnings("2026-02-10", e) is None
    assert cal.days_since_last_earnings("2026-02-11", e) == 1


def test_context_fields_empty() -> None:
    ctx = cal.context_fields("2026-01-01", cal.empty_earnings_frame())
    assert ctx["days_to_next_earnings"] is None
    assert ctx["next_earnings_date"] is None


def test_indexed_calendar_matches_public_helpers_across_dates_and_timings() -> None:
    earnings = _earn(
        [
            ("2026-05-10", "AMC"),
            ("2026-02-10", "AMC"),
            ("2026-02-10", "BMO"),
            ("2026-08-15", "unknown"),
        ]
    )
    calendar = cal.EarningsCalendar(earnings)

    day = cal.parse_date("2026-02-08")
    through = cal.parse_date("2026-08-17")
    while day <= through:
        iso = day.isoformat()
        assert calendar.next_earnings_row(iso) == cal.next_earnings_row(iso, earnings)
        assert calendar.previous_earnings_row(iso) == cal.previous_earnings_row(iso, earnings)
        assert calendar.days_to_next_earnings(iso) == cal.days_to_next_earnings(iso, earnings)
        assert calendar.days_since_last_earnings(iso) == cal.days_since_last_earnings(iso, earnings)
        assert calendar.context_fields(iso) == cal.context_fields(iso, earnings)
        day += timedelta(days=1)


def test_indexed_calendar_is_independent_of_later_frame_mutation() -> None:
    earnings = _earn([("2026-02-10", "AMC")])
    calendar = cal.EarningsCalendar(earnings)
    earnings.loc[0, "earnings_date"] = "2030-01-01"

    assert calendar.next_earnings_row("2026-02-01") == {
        "earnings_date": "2026-02-10",
        "timing": "AMC",
    }


def test_normalize_csv_fixture() -> None:
    path = Path(__file__).parent / "fixtures" / "earnings" / "aapl_sample.csv"
    raw = pd.read_csv(path)
    frame = cal.normalize_earnings_frame(raw, symbol="aapl", source="csv")
    assert list(frame.columns) == list(cal.EARNINGS_COLUMNS)
    assert frame["symbol"].iloc[0] == "AAPL"
    assert "AMC" in set(frame["timing"])
    assert "BMO" in set(frame["timing"])


def test_store_write_read_dedupe(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path)
    a = cal.normalize_earnings_frame(
        pd.DataFrame({"earnings_date": ["2024-01-25"], "timing": ["AMC"]}),
        symbol="ZZZ",
        source="csv",
    )
    b = cal.normalize_earnings_frame(
        pd.DataFrame({"earnings_date": ["2024-01-25"], "timing": ["BMO"]}),
        symbol="ZZZ",
        source="csv",
    )
    store.write_earnings("ZZZ", a)
    store.write_earnings("ZZZ", b)
    out = store.read_earnings("ZZZ")
    assert len(out) == 1
    assert out.iloc[0]["timing"] == "BMO"
