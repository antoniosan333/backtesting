"""Per-event earnings metrics from option trades + calendar + equity closes."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from typing import Any

import pandas as pd

from lambdaclass.earnings.calendar import EarningsCalendar


class _SpotLookup:
    def __init__(self, bars: pd.DataFrame | None) -> None:
        self._closes: list[str] = []
        self._exact_positions: dict[str, int] = {}
        self._dates: list[str] = []
        self._prefix_positions: list[int] = []
        if bars is None or bars.empty:
            return

        normalized_dates = bars["date"].astype(str).str[:10].tolist()
        self._closes = bars["close"].astype(str).tolist()
        positions_by_date: dict[str, int] = {}
        for position, normalized_date in enumerate(normalized_dates):
            positions_by_date[normalized_date] = position
            self._exact_positions[normalized_date] = position

        best_position = -1
        for normalized_date in sorted(positions_by_date):
            best_position = max(best_position, positions_by_date[normalized_date])
            self._dates.append(normalized_date)
            self._prefix_positions.append(best_position)

    def spot_on(self, value: str) -> float:
        requested_date = str(value)[:10]
        exact_position = self._exact_positions.get(requested_date)
        if exact_position is not None:
            return float(self._closes[exact_position])
        date_position = bisect_right(self._dates, requested_date) - 1
        if date_position < 0:
            return 0.0
        return float(self._closes[self._prefix_positions[date_position]])


def _spot_on(bars: pd.DataFrame, d: str) -> float:
    return _SpotLookup(bars).spot_on(d)


def _group_structures(option_trades: pd.DataFrame) -> list[dict[str, Any]]:
    """Group sequential opens with closes until their aggregate quantities match.

    This cannot separate concurrent or interleaved structures; callers should treat
    those fills as one structure unless trades carry a future structure identifier.
    """
    if option_trades is None or option_trades.empty:
        return []
    structures: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in option_trades.to_dict("records"):
        action = str(row.get("action", "open")).lower()
        quantity = abs(int(row.get("quantity") or 0))
        if action == "open":
            if current is None:
                current = {
                    "entry_date": str(row["date"])[:10],
                    "exit_date": None,
                    "opens": [],
                    "closes": [],
                    "premium_sum": 0.0,
                    "open_qty": 0,
                    "close_qty": 0,
                }
            current["opens"].append(row)
            current["open_qty"] += quantity
            current["premium_sum"] += float(row.get("premium") or 0.0)
        elif action in ("close", "expire") and current is not None:
            current["closes"].append(row)
            current["close_qty"] += quantity
            current["exit_date"] = str(row["date"])[:10]
            current["premium_sum"] += float(row.get("premium") or 0.0)
            if current["close_qty"] >= current["open_qty"] and current["exit_date"]:
                structures.append(current)
                current = None
    if current is not None and current["exit_date"]:
        structures.append(current)
    return structures


def _nearest_earnings(entry_date: str, earnings: pd.DataFrame | EarningsCalendar) -> dict[str, Any] | None:
    calendar = earnings if isinstance(earnings, EarningsCalendar) else EarningsCalendar(earnings)
    return calendar.nearest_earnings_row(entry_date, max_days=14)


def _long_straddle_implied_move(opens: list[Mapping[str, Any]], spot: float) -> float:
    if len(opens) != 2 or spot <= 0:
        return float("nan")
    sides = {str(leg.get("side", "")).lower() for leg in opens}
    strikes = {float(leg.get("strike") or 0.0) for leg in opens}
    expiries = {str(leg.get("expiry", "")) for leg in opens}
    quantities = [int(leg.get("quantity") or 0) for leg in opens]
    if (
        sides != {"call", "put"}
        or len(strikes) != 1
        or len(expiries) != 1
        or quantities[0] <= 0
        or quantities[0] != quantities[1]
    ):
        return float("nan")
    premium = sum(float(leg.get("premium") or 0.0) for leg in opens)
    straddle_price = premium / (100.0 * quantities[0])
    return straddle_price / spot


def _expected_move_for_event(
    expected_moves: pd.DataFrame | None,
    entry_date: str,
    earnings_date: str,
) -> pd.Series | None:
    if expected_moves is None or expected_moves.empty:
        return None
    required = {"asof", "expiry", "straddle", "iv_1sd"}
    if not required.issubset(expected_moves.columns):
        return None
    moves = expected_moves.copy()
    moves["asof"] = moves["asof"].astype(str).str[:10]
    moves["expiry"] = moves["expiry"].astype(str).str[:10]
    matches = moves[
        (moves["asof"] == str(entry_date)[:10])
        & (moves["expiry"] >= str(earnings_date)[:10])
    ].sort_values("expiry")
    return None if matches.empty else matches.iloc[0]


def compute_earnings_events(
    *,
    option_trades: pd.DataFrame,
    earnings: pd.DataFrame,
    bars: pd.DataFrame,
    iv_by_date: dict[str, float] | None = None,
    expected_moves: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one row per earnings-adjacent option structure.

    ``iv_by_date`` optional map date -> average IV used for entry/exit; if missing,
    IV columns are 0 and crush is 0.
    """
    iv_by_date = iv_by_date or {}
    structures = _group_structures(option_trades)
    matched_structures: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if structures:
        calendar = EarningsCalendar(earnings)
        for struct in structures:
            earnings_row = _nearest_earnings(struct["entry_date"], calendar)
            if earnings_row is not None:
                matched_structures.append((struct, earnings_row))
    spots = _SpotLookup(bars) if matched_structures else None
    rows: list[dict[str, Any]] = []
    for struct, earn in matched_structures:
        entry = struct["entry_date"]
        exit_d = struct["exit_date"] or entry
        assert spots is not None
        spot_entry = spots.spot_on(entry)
        spot_exit = spots.spot_on(exit_d)
        realized = abs(spot_exit - spot_entry) / spot_entry if spot_entry > 0 else 0.0
        implied_move_pct = _long_straddle_implied_move(struct["opens"], spot_entry)
        implied_move_1sd_pct = implied_move_pct * 1.25 if implied_move_pct and implied_move_pct == implied_move_pct else None

        iv_entry = float(iv_by_date.get(entry, 0.0))
        iv_exit = float(iv_by_date.get(exit_d, 0.0))
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
                "implied_move_1sd_pct": implied_move_1sd_pct,
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
                "implied_move_1sd_pct",
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
