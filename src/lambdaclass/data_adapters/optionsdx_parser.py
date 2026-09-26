from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any


def strip_header_name(raw: str) -> str:
    return raw.strip().strip("[]").strip()


def detect_schema_column_count(header_line: str) -> int:
    reader = csv.reader(StringIO(header_line.strip()))
    row = next(reader, None)
    if not row:
        return 0
    return len(row)


def parse_size_field(raw: str | None) -> tuple[int | None, int | None]:
    if raw is None:
        return None, None
    s = str(raw).strip()
    if not s:
        return None, None
    m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", s, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    t = str(raw).strip()
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _parse_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    t = str(raw).strip()
    if t == "":
        return None
    try:
        return int(float(t))
    except ValueError:
        return None


def _parse_quote_datetime(quote_readtime: str, quote_unix: int | None) -> datetime | None:
    t = (quote_readtime or "").strip()
    if not t:
        if quote_unix is not None:
            return datetime.fromtimestamp(quote_unix, tz=UTC)
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(t, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    if quote_unix is not None:
        return datetime.fromtimestamp(quote_unix, tz=UTC)
    return None


def _infer_symbol_from_path(path: Path) -> str:
    m = re.match(r"^([A-Za-z]+)_eod_", path.parent.name)
    if m:
        return m.group(1).upper()
    m2 = re.match(r"^([A-Za-z]+)_eod_", path.stem)
    if m2:
        return m2.group(1).upper()
    return path.parent.name.upper()


def _partition_year_month_from_stem(stem: str) -> tuple[int | None, int | None]:
    m = re.search(r"_eod_(\d{4})(\d{2})$", stem)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _csv_row(line: str) -> list[str]:
    reader = csv.reader(StringIO(line.rstrip("\n\r")))
    return next(reader, [])


@dataclass
class ParseResult:
    schema: str
    rows: list[dict[str, Any]]
    raw_line_count: int
    parse_errors: list[str]


def parse_optionsdx_file(path: Path) -> ParseResult:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not text:
        return ParseResult(schema="empty", rows=[], raw_line_count=0, parse_errors=[])
    header = text[0]
    ncols = detect_schema_column_count(header)
    if ncols == 33:
        return _parse_file_33(path, text[1:])
    if ncols == 28:
        return _parse_file_28(path, text[1:])
    return ParseResult(
        schema=f"unknown_{ncols}",
        rows=[],
        raw_line_count=len(text) - 1,
        parse_errors=[f"Unsupported column count: {ncols}"],
    )


def _base_row(path: Path, parts: list[str], source_file: str) -> dict[str, Any]:
    quote_unix = _parse_int(parts[0]) if len(parts) > 0 else None
    quote_read = parts[1] if len(parts) > 1 else ""
    quote_date = parts[2].strip() if len(parts) > 2 else ""
    return {
        "source_file": source_file,
        "schema_version": "optionsdx",
        "quote_unixtime": quote_unix,
        "quote_datetime": _parse_quote_datetime(quote_read, quote_unix),
        "quote_date": quote_date,
        "symbol": _infer_symbol_from_path(path),
    }


def _parse_file_33(path: Path, body_lines: list[str]) -> ParseResult:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    source = str(path.as_posix())
    stem = path.stem
    for idx, line in enumerate(body_lines, start=2):
        if not line.strip():
            continue
        parts = _csv_row(line)
        if len(parts) != 33:
            errors.append(f"{path.name}:{idx}: expected 33 columns, got {len(parts)}")
            continue
        base = _base_row(path, parts, source)
        try:
            strike = _parse_float(parts[19])
            expire_date = parts[5].strip()
            expire_unix = _parse_int(parts[6])
            dte = _parse_float(parts[7])
            strike_dist = _parse_float(parts[31])
            strike_dist_pct = _parse_float(parts[32])
            ul = _parse_float(parts[4])

            c_bid_sz, c_ask_sz = parse_size_field(parts[16])
            p_bid_sz, p_ask_sz = parse_size_field(parts[22])

            call = {
                **base,
                "side": "call",
                "underlying_last": ul,
                "expire_date": expire_date,
                "expire_unixtime": expire_unix,
                "dte": dte,
                "strike": strike,
                "contract_symbol": None,
                "bid": _parse_float(parts[17]),
                "ask": _parse_float(parts[18]),
                "last": _parse_float(parts[15]),
                "volume": _parse_float(parts[14]),
                "open_interest": None,
                "bid_size": c_bid_sz,
                "ask_size": c_ask_sz,
                "delta": _parse_float(parts[8]),
                "gamma": _parse_float(parts[9]),
                "vega": _parse_float(parts[10]),
                "theta": _parse_float(parts[11]),
                "rho": _parse_float(parts[12]),
                "iv": _parse_float(parts[13]),
                "strike_distance": strike_dist,
                "strike_distance_pct": strike_dist_pct,
                "partition_year": _partition_year_month_from_stem(stem)[0],
                "partition_month": _partition_year_month_from_stem(stem)[1],
            }
            put = {
                **base,
                "side": "put",
                "underlying_last": ul,
                "expire_date": expire_date,
                "expire_unixtime": expire_unix,
                "dte": dte,
                "strike": strike,
                "contract_symbol": None,
                "bid": _parse_float(parts[20]),
                "ask": _parse_float(parts[21]),
                "last": _parse_float(parts[23]),
                "volume": _parse_float(parts[30]),
                "open_interest": None,
                "bid_size": p_bid_sz,
                "ask_size": p_ask_sz,
                "delta": _parse_float(parts[24]),
                "gamma": _parse_float(parts[25]),
                "vega": _parse_float(parts[26]),
                "theta": _parse_float(parts[27]),
                "rho": _parse_float(parts[28]),
                "iv": _parse_float(parts[29]),
                "strike_distance": strike_dist,
                "strike_distance_pct": strike_dist_pct,
                "partition_year": _partition_year_month_from_stem(stem)[0],
                "partition_month": _partition_year_month_from_stem(stem)[1],
            }
            rows.extend([call, put])
        except Exception as exc:  # pragma: no cover
            errors.append(f"{path.name}:{idx}: {exc}")
    return ParseResult(schema="33", rows=rows, raw_line_count=len(body_lines), parse_errors=errors)


def _parse_file_28(path: Path, body_lines: list[str]) -> ParseResult:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    source = str(path.as_posix())
    stem = path.stem
    for idx, line in enumerate(body_lines, start=2):
        if not line.strip():
            continue
        parts = _csv_row(line)
        if len(parts) != 28:
            errors.append(f"{path.name}:{idx}: expected 28 columns, got {len(parts)}")
            continue
        base = _base_row(path, parts, source)
        right = (parts[13] or "").strip().lower()
        side = "call" if right.startswith("c") else "put" if right.startswith("p") else right
        row = {
            **base,
            "side": side,
            "underlying_last": _parse_float(parts[8]),
            "expire_date": parts[9].strip(),
            "expire_unixtime": _parse_int(parts[10]),
            "dte": _parse_float(parts[12]),
            "strike": _parse_float(parts[14]),
            "contract_symbol": parts[4].strip() if len(parts) > 4 else None,
            "bid": _parse_float(parts[16]),
            "ask": _parse_float(parts[17]),
            "last": None,
            "volume": _parse_float(parts[20]),
            "open_interest": _parse_float(parts[19]),
            "bid_size": _parse_int(parts[15]),
            "ask_size": _parse_int(parts[18]),
            "delta": _parse_float(parts[21]),
            "gamma": _parse_float(parts[22]),
            "vega": _parse_float(parts[23]),
            "theta": _parse_float(parts[24]),
            "rho": _parse_float(parts[25]),
            "iv": _parse_float(parts[26]),
            "strike_distance": None,
            "strike_distance_pct": _parse_float(parts[27]),
            "partition_year": _partition_year_month_from_stem(stem)[0],
            "partition_month": _partition_year_month_from_stem(stem)[1],
        }
        rows.append(row)
    return ParseResult(schema="28", rows=rows, raw_line_count=len(body_lines), parse_errors=errors)


def iter_optionsdx_files(input_root: Path) -> Iterator[Path]:
    yield from sorted(input_root.rglob("*.txt"))
