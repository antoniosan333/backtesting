"""Run the earnings IV ramp/crush study across multiple symbols.

Loads (or builds) a daily vol series per symbol, aligns earnings events on a
relative-day timeline, and aggregates the median IV path and per-event stats.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from lambdaclass.earnings.calendar import context_fields
from lambdaclass.volatility.earnings_cycle import (
    aggregate_cycle,
    align_events,
    cycle_summary,
    event_table,
    normalize,
)
from lambdaclass.volatility.series import build_vol_series


def run_earnings_iv_study(
    *,
    store: Any,
    symbols: Sequence[str],
    start: str | None = None,
    end: str | None = None,
    vol_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the IV ramp/crush study across *symbols*.

    For each symbol:
      1. Load bars + earnings from ``store``.
      2. Build vol series (from chain if available, else from vol_cache).
      3. Align earnings events on a relative-day timeline.

    Across all symbols:
      - Concatenate aligned events.
      - Normalize IV to a pre-event base.
      - Aggregate the median cycle path.
      - Build a per-event table with ramp_pct, crush_pct.

    Returns a dict with ``summary``, ``event_table``, ``cycle``, and ``aligned``.
    """
    all_aligned: list[pd.DataFrame] = []
    per_symbol_count: dict[str, int] = {}

    for symbol in symbols:
        sym = symbol.upper()
        bars = store.read_bars(sym, start=start, end=end)
        if bars is None or bars.empty:
            continue
        earnings = store.read_earnings(sym)
        if earnings is None or earnings.empty:
            continue

        # Try chain-based vol series first
        chain = store.read_chain(sym)
        vol_df: pd.DataFrame
        if chain is not None and not chain.empty:
            vol_df = build_vol_series(bars, chain, earnings)
        elif vol_cache_dir is not None:
            vol_path = vol_cache_dir / f"{sym}.parquet"
            if vol_path.is_file():
                vol_df = pd.read_parquet(vol_path)
                # Merge earnings context if missing
                if "days_to_next_earnings" not in vol_df.columns:
                    vol_df["date"] = vol_df["date"].astype(str).str[:10]
                    ctx_rows = [context_fields(d, earnings) for d in vol_df["date"]]
                    ctx_df = pd.DataFrame(ctx_rows)
                    for col in ctx_df.columns:
                        vol_df[col] = ctx_df[col].values
            else:
                continue
        else:
            continue

        aligned = align_events(vol_df, earnings, pre_days=30, post_days=10, symbol=sym)
        if aligned is None or aligned.empty:
            per_symbol_count[sym] = 0
            continue
        all_aligned.append(aligned)
        per_symbol_count[sym] = aligned["earnings_date"].nunique() if "earnings_date" in aligned.columns else 0

    if not all_aligned:
        return {
            "summary": {
                "median_ramp_pct": 0.0,
                "median_crush_pct": 0.0,
                "share_realized_below_implied": 0.0,
                "median_realized_over_implied": 0.0,
                "total_events": 0,
                "symbols_with_data": 0,
            },
            "event_table": pd.DataFrame(),
            "cycle": pd.DataFrame(),
            "aligned": pd.DataFrame(),
            "per_symbol": per_symbol_count,
        }

    combined = pd.concat(all_aligned, ignore_index=True)
    normalized = normalize(combined)
    events = event_table(normalized)
    cycle = aggregate_cycle(normalized)
    summary = cycle_summary(events)
    summary["total_events"] = int(len(events))
    summary["symbols_with_data"] = len(per_symbol_count)

    return {
        "summary": summary,
        "event_table": events,
        "cycle": cycle,
        "aligned": normalized,
        "per_symbol": per_symbol_count,
    }