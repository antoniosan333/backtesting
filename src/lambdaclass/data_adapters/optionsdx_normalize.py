from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lambdaclass.data_adapters.optionsdx_parser import (
    _partition_year_month_from_stem,
    iter_optionsdx_files,
    parse_optionsdx_file,
)
from lambdaclass.data_adapters.optionsdx_quality import rows_to_cleaned_dataframe, summarize_frame
from lambdaclass.state import load_json, save_json


@dataclass
class NormalizeOptions:
    input_root: Path
    output_root: Path
    reports_dir: Path
    state_path: Path
    dry_run: bool = False
    fail_on_errors: bool = False
    max_negative_iv_rate: float | None = None
    max_crossed_market_rate: float | None = None


def _partition_from_df(df: pd.DataFrame, stem: str) -> tuple[str, int, int]:
    sym = str(df["symbol"].iloc[0]) if not df.empty and "symbol" in df.columns else "UNKNOWN"
    y, m = _partition_year_month_from_stem(stem)
    if (y is None or m is None) and not df.empty and "quote_date" in df.columns:
        qd = str(df["quote_date"].iloc[0])[:10]
        parts = qd.split("-")
        if len(parts) >= 2:
            try:
                y = int(parts[0])
                m = int(parts[1])
            except ValueError:
                pass
    if y is None:
        y = 0
    if m is None:
        m = 0
    return sym, int(y), int(m)


def write_normalized_parquet(df: pd.DataFrame, output_root: Path, stem: str) -> Path:
    sym, year, month = _partition_from_df(df, stem)
    out_dir = output_root / sym / f"{year:04d}" / f"{month:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.parquet"
    df.to_parquet(path, index=False)
    return path


def run_normalize(opts: NormalizeOptions) -> dict[str, Any]:
    opts.input_root = opts.input_root.resolve()
    opts.output_root = opts.output_root.resolve()
    opts.reports_dir = opts.reports_dir.resolve()
    opts.state_path = opts.state_path.resolve()
    if not opts.input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {opts.input_root}")
    files_dir = opts.reports_dir / "files"
    runs_dir = opts.reports_dir / "runs"
    if not opts.dry_run:
        files_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        opts.output_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    gate_failures: list[str] = []
    totals: dict[str, Any] = {
        "files": 0,
        "raw_lines": 0,
        "output_rows": 0,
        "parse_errors": 0,
        "schema_33_files": 0,
        "schema_28_files": 0,
        "unknown_schema_files": 0,
    }

    for path in iter_optionsdx_files(opts.input_root):
        totals["files"] += 1
        parsed = parse_optionsdx_file(path)
        err_count = len(parsed.parse_errors)
        totals["parse_errors"] += err_count
        totals["raw_lines"] += parsed.raw_line_count
        if parsed.schema == "33":
            totals["schema_33_files"] += 1
        elif parsed.schema == "28":
            totals["schema_28_files"] += 1
        else:
            totals["unknown_schema_files"] += 1

        df = rows_to_cleaned_dataframe(parsed.rows)
        rep = summarize_frame(df, parsed.schema, parsed.raw_line_count, err_count, str(path.resolve()))
        file_dict = rep.to_dict()
        file_dict["parse_error_messages"] = parsed.parse_errors[:50]

        if not opts.dry_run and not df.empty:
            out_path = write_normalized_parquet(df, opts.output_root, path.stem)
            file_dict["parquet_path"] = str(out_path)
        elif not opts.dry_run:
            file_dict["parquet_path"] = None
        else:
            file_dict["parquet_path"] = None

        totals["output_rows"] += len(df)

        if err_count and opts.fail_on_errors:
            gate_failures.append(f"{path.name}: {err_count} parse errors")

        n = max(len(df), 1)
        neg_rate = rep.negative_iv_nulled / n
        cross_rate = rep.crossed_market / n
        if opts.max_negative_iv_rate is not None and neg_rate > opts.max_negative_iv_rate:
            gate_failures.append(
                f"{path.name}: negative_iv_rate {neg_rate:.4f} > {opts.max_negative_iv_rate}"
            )
        if opts.max_crossed_market_rate is not None and cross_rate > opts.max_crossed_market_rate:
            gate_failures.append(
                f"{path.name}: crossed_market_rate {cross_rate:.4f} > {opts.max_crossed_market_rate}"
            )

        if not opts.dry_run:
            (files_dir / f"{path.stem}.json").write_text(
                json.dumps(file_dict, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )

    run_summary: dict[str, Any] = {
        "run_id": run_id,
        "options": {
            "input_root": str(opts.input_root),
            "output_root": str(opts.output_root),
            "reports_dir": str(opts.reports_dir),
            "dry_run": opts.dry_run,
            "fail_on_errors": opts.fail_on_errors,
            "max_negative_iv_rate": opts.max_negative_iv_rate,
            "max_crossed_market_rate": opts.max_crossed_market_rate,
        },
        "totals": totals,
        "gate_failures": gate_failures,
        "files_processed": totals["files"],
    }
    if not opts.dry_run:
        (runs_dir / f"{run_id}.json").write_text(
            json.dumps(run_summary, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        prev = load_json(opts.state_path)
        prev["last_run"] = run_summary
        save_json(opts.state_path, prev)

    return run_summary
