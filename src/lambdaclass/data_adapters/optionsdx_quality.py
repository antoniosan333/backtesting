from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd


def _date_from_unix(quote_unix: int | None) -> str | None:
    if quote_unix is None:
        return None
    return datetime.fromtimestamp(quote_unix, tz=UTC).strftime("%Y-%m-%d")


def apply_quality_rules(row: dict[str, Any]) -> dict[str, Any]:
    flags: list[str] = []
    out = dict(row)

    quote_date = (out.get("quote_date") or "").strip()
    quote_unix = out.get("quote_unixtime")
    unix_date = _date_from_unix(quote_unix if isinstance(quote_unix, int) else None)
    if quote_date and unix_date and quote_date != unix_date:
        flags.append("DATE_MISMATCH")

    bid = out.get("bid")
    ask = out.get("ask")
    bid_f = float(bid) if bid is not None else None
    ask_f = float(ask) if ask is not None else None

    if bid_f is not None and bid_f < 0:
        out["bid"] = None
        flags.append("INVALID_PRICE")
        bid_f = None
    if ask_f is not None and ask_f < 0:
        out["ask"] = None
        flags.append("INVALID_PRICE")
        ask_f = None

    crossed = False
    if bid_f is not None and ask_f is not None and ask_f > 0 and bid_f > ask_f:
        out["bid"] = None
        out["ask"] = None
        flags.append("CROSSED_MARKET")
        crossed = True
        bid_f = ask_f = None

    iv = out.get("iv")
    iv_f = float(iv) if iv is not None else None
    if iv_f is not None and iv_f < 0:
        out["iv"] = None
        flags.append("NEGATIVE_IV")
        iv_f = None

    mid: float | None = None
    spread: float | None = None
    if bid_f is not None and ask_f is not None and bid_f >= 0 and ask_f >= 0 and bid_f <= ask_f:
        mid = (bid_f + ask_f) / 2.0
        spread = ask_f - bid_f
    elif bid_f is not None and ask_f is None:
        mid = bid_f
    elif ask_f is not None and bid_f is None:
        mid = ask_f

    if mid is None and not crossed and bid_f is not None and ask_f is not None:
        flags.append("NO_TRUSTED_MID")

    out["mid"] = mid
    out["spread"] = spread
    out["quality_flags"] = "|".join(sorted(set(flags))) if flags else ""
    out["is_crossed_market"] = crossed
    out["is_valid_iv"] = iv_f is not None and iv_f >= 0
    out["is_valid_price"] = bid_f is not None or ask_f is not None
    out["is_kept"] = True
    return out


def rows_to_cleaned_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    cleaned = [apply_quality_rules(r) for r in rows]
    if not cleaned:
        return pd.DataFrame()
    return pd.DataFrame(cleaned)


@dataclass
class FileQualityReport:
    source_path: str
    schema: str
    raw_lines: int
    output_rows: int
    parse_errors: int
    blank_iv: int = 0
    negative_iv_nulled: int = 0
    crossed_market: int = 0
    invalid_price: int = 0
    date_mismatch: int = 0
    flags: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "schema": self.schema,
            "raw_lines": self.raw_lines,
            "output_rows": self.output_rows,
            "parse_errors": self.parse_errors,
            "blank_iv": self.blank_iv,
            "negative_iv_nulled": self.negative_iv_nulled,
            "crossed_market": self.crossed_market,
            "invalid_price": self.invalid_price,
            "date_mismatch": self.date_mismatch,
            "flags": self.flags,
        }


def summarize_frame(
    df: pd.DataFrame, schema: str, raw_lines: int, parse_error_count: int, source: str
) -> FileQualityReport:
    rep = FileQualityReport(
        source_path=source,
        schema=schema,
        raw_lines=raw_lines,
        output_rows=len(df),
        parse_errors=parse_error_count,
    )
    if df.empty:
        return rep
    if "iv" in df.columns:
        rep.blank_iv = int(df["iv"].isna().sum())
    if "quality_flags" in df.columns:
        for _, val in df["quality_flags"].items():
            if not val:
                continue
            for part in str(val).split("|"):
                if not part:
                    continue
                rep.flags[part] = rep.flags.get(part, 0) + 1
    if "is_crossed_market" in df.columns:
        rep.crossed_market = int(df["is_crossed_market"].astype(bool).sum())
    rep.negative_iv_nulled = rep.flags.get("NEGATIVE_IV", 0)
    rep.invalid_price = rep.flags.get("INVALID_PRICE", 0)
    rep.date_mismatch = rep.flags.get("DATE_MISMATCH", 0)
    return rep
