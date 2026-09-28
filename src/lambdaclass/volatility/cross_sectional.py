"""Cross-sectional ranking of symbols by earnings IV characteristics."""

from __future__ import annotations

import pandas as pd


def rank_symbols(
    event_table: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank symbols by historical earnings IV behavior.

    Input: event_table from earnings_cycle.event_table, must include a
    ``symbol`` column (add it before calling if not present).

    Returns one row per symbol with median metrics and a composite score.
    Default weights favor long-vol strategies (high realized_over_implied,
    high crush, high ramp). Adjust weights for short-vol screening.
    """
    if event_table is None or event_table.empty:
        return pd.DataFrame()

    w = weights or {
        "realized_over_implied": 0.5,
        "crush_pct": 0.3,
        "ramp_pct": 0.2,
    }

    grouped = event_table.groupby("symbol")
    rows: list[dict] = []
    for symbol, group in grouped:
        row = {"symbol": symbol}
        row["n_events"] = len(group)
        # Short display names map source columns to output median columns.
        metric_names = {
            "crush_pct": "crush",
            "ramp_pct": "ramp",
            "realized_over_implied": "realized_over_implied",
        }
        for metric, out_name in metric_names.items():
            vals = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            row[f"median_{out_name}"] = float(vals.median()) if not vals.empty else float("nan")
        # Composite score (higher = better for long vol)
        score = 0.0
        for metric, weight in w.items():
            out_name = metric_names.get(metric, metric)
            val = row.get(f"median_{out_name}")
            if val is not None and val == val:  # not NaN
                score += weight * val
        row["score"] = score
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return result