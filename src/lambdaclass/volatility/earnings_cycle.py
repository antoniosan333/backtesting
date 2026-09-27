"""Earnings event study for the implied-volatility ramp and crush."""

from __future__ import annotations

import pandas as pd

from lambdaclass.earnings.calendar import normalize_timing, parse_date


def align_events(
    series: pd.DataFrame,
    earnings: pd.DataFrame,
    *,
    pre_days: int = 30,
    post_days: int = 10,
) -> pd.DataFrame:
    """Stack each earnings event on trading days around its reaction session.

    ``rel_day`` 0 is the last close before the reaction. ``rel_day`` 1 is the reaction close.
    """
    if series is None or series.empty or earnings is None or earnings.empty:
        return pd.DataFrame()
    bars = series.copy()
    bars["date"] = bars["date"].astype(str).str[:10]
    bars = bars.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    days = bars["date"].tolist()
    rows: list[dict[str, object]] = []
    for _, event in earnings.iterrows():
        event_day = parse_date(event["earnings_date"]).isoformat()
        timing = normalize_timing(event.get("timing", "unknown"))
        reaction = _reaction_index(days, event_day, timing)
        if reaction is None or reaction == 0:
            continue
        peak = reaction - 1
        for offset in range(-pre_days, post_days + 1):
            index = peak + offset
            if index < 0 or index >= len(bars):
                continue
            row = bars.iloc[index].to_dict()
            row["earnings_date"] = event_day
            row["timing"] = timing
            row["rel_day"] = offset
            rows.append(row)
    return pd.DataFrame(rows)


def normalize(aligned: pd.DataFrame, *, base: str = "rel_day=-30") -> pd.DataFrame:
    """Express IV as a ratio to a pre-event base so events can be averaged."""
    if aligned is None or aligned.empty:
        return pd.DataFrame() if aligned is None else aligned.copy()
    out = aligned.copy()
    bases: dict[str, float] = {}
    for event_day, group in out.groupby("earnings_date"):
        if base == "ambient":
            anchor = group[group["rel_day"] == 0]
            column = "hv20_ex_earnings"
        else:
            anchor = group[group["rel_day"] == -30]
            column = "iv_front"
        value = pd.to_numeric(anchor[column], errors="coerce") if column in anchor else pd.Series(dtype=float)
        bases[str(event_day)] = float(value.iloc[0]) if not value.empty and pd.notna(value.iloc[0]) and float(value.iloc[0]) > 0 else float("nan")
    base_value = out["earnings_date"].map(bases)
    for column in ("iv30", "iv_front"):
        if column in out.columns:
            out[f"{column}_norm"] = pd.to_numeric(out[column], errors="coerce") / base_value
    return out


def aggregate_cycle(aligned: pd.DataFrame) -> pd.DataFrame:
    """Median path and interquartile band for each relative day."""
    columns = ["rel_day", "metric", "median", "q25", "q75", "mean", "n"]
    if aligned is None or aligned.empty:
        return pd.DataFrame(columns=columns)
    metrics = [
        column
        for column in ("iv30", "iv_front", "iv30_norm", "iv_front_norm")
        if column in aligned.columns
    ]
    rows: list[dict[str, object]] = []
    for metric in metrics:
        for rel_day, group in aligned.groupby("rel_day"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "rel_day": int(rel_day),
                    "metric": metric,
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "mean": float(values.mean()),
                    "n": int(len(values)),
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(["metric", "rel_day"]).reset_index(drop=True)


def event_table(aligned: pd.DataFrame) -> pd.DataFrame:
    """One row per earnings event: ramp, crush, and realized versus implied move."""
    columns = [
        "earnings_date",
        "timing",
        "iv_front_pre",
        "iv_front_post",
        "crush_pct",
        "iv30_pre",
        "iv30_post",
        "implied_event_move",
        "expected_move_straddle",
        "realized_move",
        "realized_over_implied",
        "ramp_pct",
    ]
    if aligned is None or aligned.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for event_day, group in aligned.groupby("earnings_date"):
        by_day = group.set_index("rel_day")

        def _value(rel_day: int, column: str) -> float | None:
            if rel_day not in by_day.index or column not in by_day.columns:
                return None
            value = pd.to_numeric(by_day.loc[rel_day, column], errors="coerce")
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            return None if pd.isna(value) else float(value)

        pre = _value(0, "iv_front")
        post = _value(1, "iv_front")
        base = _value(-20, "iv_front")
        close_pre = _value(0, "close")
        close_post = _value(1, "close")
        implied = _value(0, "implied_event_move")
        realized = None
        if close_pre not in (None, 0.0) and close_post is not None:
            realized = abs(close_post - close_pre) / close_pre
        implied_pct = implied / close_pre if implied not in (None, 0.0) and close_pre not in (None, 0.0) else None
        rows.append(
            {
                "earnings_date": event_day,
                "timing": group["timing"].iloc[0],
                "iv_front_pre": pre,
                "iv_front_post": post,
                "crush_pct": None if pre in (None, 0.0) or post is None else 1.0 - post / pre,
                "iv30_pre": _value(0, "iv30"),
                "iv30_post": _value(1, "iv30"),
                "implied_event_move": implied,
                "expected_move_straddle": _value(0, "front_straddle"),
                "realized_move": realized,
                "realized_over_implied": None if implied_pct in (None, 0.0) or realized is None else realized / implied_pct,
                "ramp_pct": None if base in (None, 0.0) or pre is None else pre / base - 1.0,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def cycle_summary(table: pd.DataFrame) -> dict[str, float]:
    """Median ramp, crush, and how often the realized move stayed inside the implied move."""
    if table is None or table.empty:
        return {
            "median_ramp_pct": 0.0,
            "median_crush_pct": 0.0,
            "share_realized_below_implied": 0.0,
            "median_realized_over_implied": 0.0,
        }
    ratio = pd.to_numeric(table["realized_over_implied"], errors="coerce").dropna()
    return {
        "median_ramp_pct": float(pd.to_numeric(table["ramp_pct"], errors="coerce").median()),
        "median_crush_pct": float(pd.to_numeric(table["crush_pct"], errors="coerce").median()),
        "share_realized_below_implied": float((ratio < 1.0).mean()) if len(ratio) else 0.0,
        "median_realized_over_implied": float(ratio.median()) if len(ratio) else 0.0,
    }


def _reaction_index(days: list[str], event_day: str, timing: str) -> int | None:
    if timing == "BMO":
        return next((index for index, day in enumerate(days) if day >= event_day), None)
    return next((index for index, day in enumerate(days) if day > event_day), None)
