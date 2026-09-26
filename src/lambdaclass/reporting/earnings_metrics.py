"""Per-event earnings metrics from option trades + calendar + equity closes."""

from __future__ import annotations

from typing import Any

import pandas as pd

from lambdaclass.earnings.calendar import normalize_timing, parse_date


def _spot_on(bars: pd.DataFrame, d: str) -> float:
    if bars is None or bars.empty:
        return 0.0
    df = bars.copy()
    df["date"] = df["date"].astype(str).str[:10]
    sub = df[df["date"] == str(d)[:10]]
    if sub.empty:
        sub = df[df["date"] <= str(d)[:10]]
    if sub.empty:
        return 0.0
    return float(sub.iloc[-1]["close"])


def _group_structures(option_trades: pd.DataFrame) -> list[dict[str, Any]]:
    """Group option_trades into open→close/expire structures by first open date cluster."""
    if option_trades is None or option_trades.empty:
        return []
    ot = option_trades.copy().reset_index(drop=True)
    structures: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for _, row in ot.iterrows():
        action = str(row.get("action", "open")).lower()
        if action == "open":
            if current is None:
                current = {
                    "entry_date": str(row["date"])[:10],
                    "exit_date": None,
                    "opens": [],
                    "closes": [],
                    "premium_sum": 0.0,
                    "iv_entry_vals": [],
                    "iv_exit_vals": [],
                }
            current["opens"].append(row)
            current["premium_sum"] += float(row.get("premium") or 0.0)
            # mid as rough IV proxy when iv not stored — skip; use mid only for premium
        elif action in ("close", "expire") and current is not None:
            current["closes"].append(row)
            current["exit_date"] = str(row["date"])[:10]
            # closing premium is opposite sign cash flow already in premium field
            current["premium_sum"] += float(row.get("premium") or 0.0)
            # If all opens have been closed (by count of close qty), finalize
            open_qty = sum(abs(int(r.get("quantity") or 0)) for r in current["opens"])
            close_qty = sum(abs(int(r.get("quantity") or 0)) for r in current["closes"])
            if close_qty >= open_qty and current["exit_date"]:
                structures.append(current)
                current = None
    if current is not None and current["exit_date"]:
        structures.append(current)
    return structures


def _nearest_earnings(entry_date: str, earnings: pd.DataFrame) -> dict[str, Any] | None:
    if earnings is None or earnings.empty:
        return None
    ed = parse_date(entry_date)
    best = None
    best_delta = None
    for _, row in earnings.iterrows():
        e_date = parse_date(row["earnings_date"])
        delta = abs((e_date - ed).days)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best = {
                "earnings_date": e_date.isoformat(),
                "timing": normalize_timing(row.get("timing", "unknown")),
            }
    # Only attach if within 14 calendar days of entry
    if best is None or best_delta is None or best_delta > 14:
        return None
    return best


def compute_earnings_events(
    *,
    option_trades: pd.DataFrame,
    earnings: pd.DataFrame,
    bars: pd.DataFrame,
    iv_by_date: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build one row per earnings-adjacent option structure.

    ``iv_by_date`` optional map date -> average IV used for entry/exit; if missing,
    IV columns are 0 and crush is 0.
    """
    iv_by_date = iv_by_date or {}
    rows: list[dict[str, Any]] = []
    for struct in _group_structures(option_trades):
        entry = struct["entry_date"]
        exit_d = struct["exit_date"] or entry
        earn = _nearest_earnings(entry, earnings)
        if earn is None:
            continue
        spot_entry = _spot_on(bars, entry)
        spot_exit = _spot_on(bars, exit_d)
        realized = abs(spot_exit - spot_entry) / spot_entry if spot_entry > 0 else 0.0
        # Implied move from debit/credit of structure at entry: |net premium| / (100 * spot)
        # For a 1-lot straddle, premium_sum at open is qty*100*mid sum; use absolute open premium / spot
        open_prem = sum(float(r.get("premium") or 0.0) for r in struct["opens"])
        # Long pays positive premium sum; implied move ≈ abs(open_prem) / (100 * spot) when 1 lot
        # Better: abs(open_prem) / spot for dollar premium already ×100
        implied = abs(open_prem) / spot_entry if spot_entry > 0 else 0.0
        # For multi-leg, open_prem is already dollar notional; divide by spot for pct of underlying
        # 1-lot ATM straddle ~ premium_dollars / spot  e.g. 800/100 = 8.0 meaning 8% — correct as pct points
        implied_move_pct = implied  # already fraction if we use open_prem/spot? 800/100=8 → need /100
        # open_prem is dollars (qty*100*mid). Fraction of spot: open_prem / (spot * 100) for 1 share equiv
        # Standard: implied move % ≈ straddle_price / spot. straddle_price = open_prem/100 for 1 lot.
        lots_proxy = max(sum(abs(int(r.get("quantity") or 0)) for r in struct["opens"]) / 2.0, 1.0)
        straddle_px = abs(open_prem) / (100.0 * lots_proxy)
        implied_move_pct = straddle_px / spot_entry if spot_entry > 0 else 0.0

        iv_entry = float(iv_by_date.get(entry, 0.0))
        iv_exit = float(iv_by_date.get(exit_d, 0.0))
        # Event P&L: -sum of all premiums (opens pay, closes receive) — cash convention in engine
        # premium on open for long is positive (cash decreased by premium); close premium negative
        # Net cash change from structure ≈ -sum(premium) for longs... Actually cash -= premium on fill,
        # so event_pnl = -sum(all premiums in opens+closes) works when close premium is qty*100*mid
        # with closing qty opposite sign.
        event_pnl = -float(struct["premium_sum"])

        rows.append(
            {
                "earnings_date": earn["earnings_date"],
                "timing": earn["timing"],
                "entry_date": entry,
                "exit_date": exit_d,
                "spot_entry": spot_entry,
                "spot_exit": spot_exit,
                "implied_move_pct": implied_move_pct,
                "realized_move_pct": realized,
                "iv_entry": iv_entry,
                "iv_exit": iv_exit,
                "iv_crush": iv_entry - iv_exit,
                "event_pnl": event_pnl,
                "beat_implied": bool(realized > implied_move_pct),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "earnings_date",
                "timing",
                "entry_date",
                "exit_date",
                "spot_entry",
                "spot_exit",
                "implied_move_pct",
                "realized_move_pct",
                "iv_entry",
                "iv_exit",
                "iv_crush",
                "event_pnl",
                "beat_implied",
            ]
        )
    return pd.DataFrame(rows)


def summarize_earnings_events(events: pd.DataFrame) -> dict[str, float]:
    if events is None or events.empty:
        return {
            "earnings_event_count": 0.0,
            "earnings_avg_iv_crush": 0.0,
            "earnings_win_rate": 0.0,
            "earnings_total_pnl": 0.0,
        }
    wins = (events["event_pnl"] > 0).sum()
    n = len(events)
    return {
        "earnings_event_count": float(n),
        "earnings_avg_iv_crush": float(events["iv_crush"].mean()),
        "earnings_win_rate": float(wins / n) if n else 0.0,
        "earnings_total_pnl": float(events["event_pnl"].sum()),
    }


def average_iv_from_option_trades(option_trades: pd.DataFrame) -> dict[str, float]:
    """Mean IV of fills by date when ``iv`` column is present."""
    if option_trades is None or option_trades.empty or "iv" not in option_trades.columns:
        return {}
    ot = option_trades.copy()
    ot["date"] = ot["date"].astype(str).str[:10]
    out: dict[str, float] = {}
    for d, grp in ot.groupby("date"):
        vals = pd.to_numeric(grp["iv"], errors="coerce").dropna()
        if len(vals):
            out[str(d)] = float(vals.mean())
    return out
