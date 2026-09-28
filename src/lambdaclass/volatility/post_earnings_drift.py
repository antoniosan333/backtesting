"""Post-earnings volatility drift: is post-event realized vol underpriced?"""

from __future__ import annotations

import pandas as pd

from lambdaclass.earnings.calendar import normalize_timing, parse_date


def analyze_drift(
    vol_series: pd.DataFrame,
    earnings: pd.DataFrame,
    *,
    pre_window: int = 5,
    post_window: int = 21,
) -> pd.DataFrame:
    """For each earnings event, compare pre-event IV30 to post-event realized vol.

    Columns:
        earnings_date, timing,
        pre_iv30 (median IV30 in the N sessions before reaction),
        post_rv21 (forward 21-day realized vol at the pre-event close),
        vrp (pre_iv30 - post_rv21, positive = IV overpriced),
        vrp_ratio (pre_iv30 / post_rv21, >1 = IV overpriced),
        pre_close, post_close_5d, price_drift_pct
    """
    if vol_series is None or vol_series.empty or earnings is None or earnings.empty:
        return pd.DataFrame()

    bars = vol_series.copy()
    bars["date"] = bars["date"].astype(str).str[:10]
    bars = bars.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    days = bars["date"].tolist()

    rows: list[dict] = []
    for _, event in earnings.iterrows():
        event_day = parse_date(event["earnings_date"]).isoformat()
        timing = normalize_timing(event.get("timing", "unknown"))

        # Reaction index: first bar on/after (BMO) or after (AMC) the event
        if timing == "BMO":
            reaction_idx = next((i for i, d in enumerate(days) if d >= event_day), None)
        else:
            reaction_idx = next((i for i, d in enumerate(days) if d > event_day), None)

        if reaction_idx is None or reaction_idx == 0:
            continue

        pre_start = max(0, reaction_idx - pre_window)
        pre_slice = bars.iloc[pre_start:reaction_idx]

        pre_iv30 = pd.to_numeric(pre_slice.get("iv30"), errors="coerce").dropna()
        pre_iv30_val = float(pre_iv30.median()) if not pre_iv30.empty else float("nan")

        # rv_fwd21 at the last pre-event bar
        rv_col = "rv_fwd21" if "rv_fwd21" in bars.columns else None
        post_rv = float("nan")
        if rv_col:
            rv_val = pd.to_numeric(bars.iloc[reaction_idx - 1].get(rv_col), errors="coerce")
            if pd.notna(rv_val):
                post_rv = float(rv_val)

        pre_close = float(bars.iloc[reaction_idx - 1].get("close", 0))
        post_idx = min(reaction_idx + 5, len(bars) - 1)
        post_close = float(bars.iloc[post_idx].get("close", 0))
        price_drift = (post_close - pre_close) / pre_close if pre_close > 0 else float("nan")

        vrp = pre_iv30_val - post_rv if pre_iv30_val == pre_iv30_val and post_rv == post_rv else float("nan")
        vrp_ratio = pre_iv30_val / post_rv if pre_iv30_val == pre_iv30_val and post_rv not in (0, float("nan")) else float("nan")

        rows.append({
            "earnings_date": event_day,
            "timing": timing,
            "pre_iv30": pre_iv30_val,
            "post_rv21": post_rv,
            "vrp": vrp,
            "vrp_ratio": vrp_ratio,
            "pre_close": pre_close,
            "post_close_5d": post_close,
            "price_drift_pct": price_drift,
        })

    return pd.DataFrame(rows)


def summarize_drift(drift: pd.DataFrame) -> dict[str, float]:
    """Aggregate stats: median VRP, share of events where IV > RV."""
    if drift is None or drift.empty:
        return {
            "median_vrp": 0.0,
            "median_vrp_ratio": 0.0,
            "share_iv_overpriced": 0.0,
            "median_price_drift": 0.0,
            "n_events": 0,
        }
    vrp = pd.to_numeric(drift["vrp"], errors="coerce").dropna()
    ratio = pd.to_numeric(drift["vrp_ratio"], errors="coerce").dropna()
    drift_pct = pd.to_numeric(drift["price_drift_pct"], errors="coerce").dropna()
    return {
        "median_vrp": float(vrp.median()) if not vrp.empty else 0.0,
        "median_vrp_ratio": float(ratio.median()) if not ratio.empty else 0.0,
        "share_iv_overpriced": float((vrp > 0).mean()) if not vrp.empty else 0.0,
        "median_price_drift": float(drift_pct.median()) if not drift_pct.empty else 0.0,
        "n_events": len(drift),
    }