from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import pytest

from lambdaclass.data_adapters.cboe_weeklys import parse_cboe_weeklys, to_vendor_neutral_symbol
from lambdaclass.data_adapters.nasdaq_earnings import (
    EARNINGS_DAY_COLUMNS,
    parse_money,
    parse_nasdaq_earnings_rows,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_cboe_weeklys_keeps_only_etp_and_equity_sections() -> None:
    frame = parse_cboe_weeklys((FIXTURES / "cboe" / "weeklys_sample.csv").read_text(encoding="utf-8"))

    assert list(frame.columns) == ["symbol", "name", "category"]
    assert frame[frame["category"] == "etp"]["symbol"].tolist() == ["AGQ", "SPY"]
    assert frame[frame["category"] == "equity"]["symbol"].tolist() == [
        "AA",
        "AAL",
        "AAOI",
        "AAP",
        "AAPL",
        "JD",
        "NFLX",
    ]
    assert not frame["symbol"].str.contains(r"\(").any()
    assert frame.loc[frame["symbol"] == "AAPL", "name"].item() == "APPLE INC COM"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("brk.b", "BRK-B"), ("BRK/B", "BRK-B"), (" aapl ", "AAPL"), ("PBR.A", "PBR-A")],
)
def test_to_vendor_neutral_symbol(raw: str, expected: str) -> None:
    assert to_vendor_neutral_symbol(raw) == expected


def test_parse_cboe_weeklys_drops_unsafe_symbols() -> None:
    text = (
        'Available Weeklys - Equity\n"../../etc","EVIL"\n"A B","SPACE"\n"^VIX","INDEX"\n"BRK.B","BERKSHIRE"\n'
    )

    assert parse_cboe_weeklys(text)["symbol"].tolist() == ["BRK-B"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$1.23", 1.23),
        ("($0.09)", -0.09),
        ("-50", -50.0),
        ("$4,902,477,100,000", 4_902_477_100_000.0),
        ("13", 13.0),
    ],
)
def test_parse_money(raw: str, expected: float) -> None:
    assert parse_money(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["N/A", "", None, "--", "$"])
def test_parse_money_missing(raw: object) -> None:
    assert math.isnan(parse_money(raw))


def _fixture_rows(day: str) -> list[dict[str, object]]:
    payload = json.loads((FIXTURES / "nasdaq" / "earnings_days.json").read_text(encoding="utf-8"))
    return payload[day]["data"]["rows"] or []


def test_parse_historical_nasdaq_day() -> None:
    frame = parse_nasdaq_earnings_rows(_fixture_rows("2025-02-26"), date(2025, 2, 26))

    assert list(frame.columns) == EARNINGS_DAY_COLUMNS
    assert frame["symbol"].tolist() == ["NVDA", "CRM", "SNOW", "PBR-A"]
    nvda = frame.set_index("symbol").loc["NVDA"]
    assert nvda["earnings_date"] == "2025-02-26"
    assert nvda["eps_actual"] == pytest.approx(0.85)
    assert nvda["eps_estimate"] == pytest.approx(0.79)
    assert nvda["surprise_pct"] == pytest.approx(7.59)
    assert nvda["timing"] == "unknown"
    snow = frame.set_index("symbol").loc["SNOW"]
    assert snow["eps_actual"] == pytest.approx(-0.90)
    assert snow["eps_estimate"] == pytest.approx(-0.60)
    pbr = frame.set_index("symbol").loc["PBR-A"]
    assert math.isnan(pbr["eps_estimate"]) and math.isnan(pbr["surprise_pct"])


def test_parse_upcoming_nasdaq_day_maps_timing() -> None:
    frame = parse_nasdaq_earnings_rows(_fixture_rows("2026-10-15"), date(2026, 10, 15))

    assert frame["timing"].tolist() == ["BMO", "AMC"]
    assert frame["eps_actual"].isna().all()
    assert frame["eps_estimate"].notna().all()


def test_parse_empty_nasdaq_day_keeps_schema() -> None:
    frame = parse_nasdaq_earnings_rows(_fixture_rows("2026-09-26"), date(2026, 9, 26))

    assert frame.empty
    assert list(frame.columns) == EARNINGS_DAY_COLUMNS
