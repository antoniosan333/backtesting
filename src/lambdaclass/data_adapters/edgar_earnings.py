"""Earnings dates from SEC EDGAR 8-K Item 2.02 filings."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

from lambdaclass.earnings.calendar import normalize_earnings_frame

CONTACT_ENV = "LAMBDACLASS__edgar__contact_email"
TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EASTERN = ZoneInfo("America/New_York")


def contact_email() -> str:
    """Read the SEC User-Agent contact from the environment."""
    email = os.environ.get(CONTACT_ENV, "").strip()
    if "@" not in email:
        raise RuntimeError(
            f"Set {CONTACT_ENV} to a contact email before fetching EDGAR earnings."
        )
    return email


def user_agent(email: str) -> str:
    return f"LambdaClass research {email}"


def timing_from_acceptance(timestamp: str) -> str:
    """Map an EDGAR acceptance timestamp to BMO, AMC, or unknown."""
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(EASTERN)
    minutes = parsed.hour * 60 + parsed.minute
    if minutes < 9 * 60 + 30:
        return "BMO"
    if minutes >= 16 * 60:
        return "AMC"
    return "unknown"


def earnings_rows_from_submission(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Pull Item 2.02 filings out of one submissions JSON document."""
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    items = recent.get("items", [])
    filing_dates = recent.get("filingDate", [])
    accepted = recent.get("acceptanceDateTime", [])
    rows: list[dict[str, str]] = []
    for index, form in enumerate(forms):
        if str(form).upper() != "8-K":
            continue
        item_text = str(items[index]) if index < len(items) else ""
        if "2.02" not in item_text:
            continue
        acceptance = str(accepted[index]) if index < len(accepted) else ""
        filed = str(filing_dates[index])[:10] if index < len(filing_dates) else ""
        event_day = acceptance[:10] or filed
        if not event_day:
            continue
        rows.append(
            {
                "earnings_date": event_day,
                "timing": timing_from_acceptance(acceptance) if acceptance else "unknown",
            }
        )
    return rows


def fetch_earnings_frame(
    symbol: str,
    *,
    http_get: Callable[[str, str], dict[str, Any]] | None = None,
    email: str | None = None,
) -> pd.DataFrame:
    """Download the earnings calendar for ``symbol`` and return the canonical frame."""
    contact = email or contact_email()
    getter = http_get or _default_get
    tickers = getter(TICKER_URL, contact)
    cik = _cik_for_ticker(tickers, symbol)
    if cik is None:
        return normalize_earnings_frame(pd.DataFrame(), symbol=symbol, source="edgar")
    submission = getter(SUBMISSIONS_URL.format(cik=cik), contact)
    rows = earnings_rows_from_submission(submission)
    for extra in submission.get("filings", {}).get("files", []):
        name = str(extra.get("name", ""))
        if not name:
            continue
        older = getter(f"https://data.sec.gov/submissions/{name}", contact)
        rows.extend(earnings_rows_from_submission({"filings": {"recent": older}}))
    return normalize_earnings_frame(pd.DataFrame(rows), symbol=symbol, source="edgar")


def _cik_for_ticker(payload: dict[str, Any], symbol: str) -> str | None:
    wanted = symbol.upper()
    for entry in payload.values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("ticker", "")).upper() == wanted:
            return str(entry.get("cik_str", "")).zfill(10)
    return None


def _default_get(url: str, email: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": user_agent(email), "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))
