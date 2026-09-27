from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from lambdaclass.config import Preferences
from lambdaclass.earnings.calendar import context_fields
from lambdaclass.options.expected_move import expected_moves as calculate_expected_moves
from lambdaclass.options.pricing import (
    black_scholes_price,
    intrinsic_value,
    numeric_value,
    parse_option_date,
    safe_option_mid,
    years_between,
)
from lambdaclass.strategies.base import (
    OpenOptionView,
    OptionLeg,
    Strategy,
    StrategyContext,
    StrategyDecision,
)


@dataclass
class OpenOption:
    side: str
    strike: float
    expiry: date
    quantity: int
    avg_entry_mid: float
    last_iv: float


@dataclass
class RunResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    final_cash: float
    final_position: int
    option_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    rejected_orders: pd.DataFrame = field(default_factory=pd.DataFrame)
    expected_moves: pd.DataFrame = field(default_factory=pd.DataFrame)


def _option_commission(contracts: int, prefs: Preferences) -> float:
    return abs(contracts) * prefs.defaults.commission_per_contract


def _stock_commission(shares: int, prefs: Preferences) -> float:
    return prefs.defaults.stock_commission_per_order + (
        abs(shares) * prefs.defaults.stock_commission_per_share
    )


def _select_expected_move_horizons(frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for horizon in horizons:
        candidates = frame[frame["dte"] >= float(horizon)]
        row = candidates.sort_values("dte").iloc[0] if not candidates.empty else frame.sort_values("dte").iloc[-1]
        selected.append(row)
    return pd.DataFrame(selected).drop_duplicates(subset=["expiry"]).reset_index(drop=True)


def _chain_row_for_contract(chain: pd.DataFrame | None, contract_symbol: str) -> pd.Series | None:
    if chain is None or chain.empty:
        return None
    if chain.index.name == "contract_symbol":
        try:
            row = chain.loc[contract_symbol]
        except KeyError:
            return None
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row
    matches = chain[chain["contract_symbol"] == contract_symbol]
    if matches.empty:
        return None
    return matches.iloc[0]


def prepare_chain_by_date(options_chain: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a chain into per-``asof`` frames indexed by ``contract_symbol``.

    Building this is the dominant cost of a run on large chains; callers that run
    many backtests over the same chain should build it once and pass it to
    ``run_backtest(chain_by_date=...)``. Strategies must not mutate these frames.
    """
    if options_chain.empty:
        return {}
    indexed = options_chain.set_index("contract_symbol", drop=False)
    return {str(key): frame for key, frame in indexed.groupby("asof", sort=False)}


@dataclass
class FillQuote:
    mid: float = 0.0
    iv: float = 1e-6
    reject_reason: str | None = None


def _fill_quote(chain: pd.DataFrame | None, leg: OptionLeg) -> FillQuote:
    """Price a leg from the day's chain, or explain why it cannot be filled.

    A contract absent from the chain has no observable price, so the order is
    rejected rather than filled at zero — a zero fill would hand the strategy a
    free position that is then marked to BSM.
    """
    if chain is None or chain.empty:
        return FillQuote(reject_reason="no_chain_for_date")
    row = _chain_row_for_contract(chain, leg.contract_symbol)
    if row is None:
        return FillQuote(reject_reason="contract_not_in_chain")
    mid = float(safe_option_mid(row))
    if not mid > 0.0:
        return FillQuote(reject_reason="non_positive_mid")
    iv = numeric_value(row.get("implied_volatility"))
    return FillQuote(mid=mid, iv=max(iv, 1e-6))


def _option_mark(
    pos: OpenOption,
    *,
    chain: pd.DataFrame | None,
    contract_symbol: str,
    spot: float,
    bar_date: date,
    r: float,
) -> float:
    row = _chain_row_for_contract(chain, contract_symbol)
    if row is not None:
        mid = float(safe_option_mid(row))
        iv = numeric_value(row.get("implied_volatility"))
        if iv > 0:
            pos.last_iv = max(iv, 1e-6)
        if mid > 0:
            return mid
    t = years_between(bar_date, pos.expiry)
    if t <= 0.0:
        return intrinsic_value(pos.side, pos.strike, spot)
    return black_scholes_price(pos.side, spot, pos.strike, t, r, max(pos.last_iv, 1e-6))


def _apply_option_fill(
    open_options: dict[str, OpenOption],
    leg: OptionLeg,
    *,
    mid: float,
    iv: float,
    cash: float,
    prefs: Preferences,
    date_key: str,
    option_trades: list[dict[str, Any]],
) -> float:
    """Net ``leg`` into the ledger; return updated cash."""
    qty = int(leg.quantity)
    if qty == 0:
        return cash
    commission = _option_commission(qty, prefs)
    premium = qty * 100.0 * mid
    slip = abs(premium) * (prefs.defaults.slippage_bps / 10_000.0)
    # Long (qty>0): pay premium + slip + commission. Short (qty<0): receive |premium| - slip - commission.
    if qty > 0:
        cash -= premium + slip + commission
    else:
        cash -= premium  # premium negative → cash increases
        cash -= slip + commission

    existing = open_options.get(leg.contract_symbol)
    action = "open"
    if existing is None:
        open_options[leg.contract_symbol] = OpenOption(
            side=str(leg.side).lower(),
            strike=float(leg.strike),
            expiry=parse_option_date(leg.expiry),
            quantity=qty,
            avg_entry_mid=mid,
            last_iv=iv,
        )
    elif existing.quantity * qty < 0 and abs(qty) >= abs(existing.quantity):
        # Full close (and optional reverse)
        action = "close"
        remaining = qty + existing.quantity
        del open_options[leg.contract_symbol]
        if remaining != 0:
            open_options[leg.contract_symbol] = OpenOption(
                side=str(leg.side).lower(),
                strike=float(leg.strike),
                expiry=parse_option_date(leg.expiry),
                quantity=remaining,
                avg_entry_mid=mid,
                last_iv=iv,
            )
            action = "reverse"
    elif existing.quantity * qty < 0:
        # Partial close
        action = "close"
        existing.quantity += qty
        if existing.quantity == 0:
            del open_options[leg.contract_symbol]
    else:
        # Add to same-direction position — average entry mid
        total_qty = existing.quantity + qty
        if total_qty != 0:
            existing.avg_entry_mid = (existing.avg_entry_mid * abs(existing.quantity) + mid * abs(qty)) / abs(
                total_qty
            )
        existing.quantity = total_qty
        existing.last_iv = iv

    option_trades.append(
        {
            "date": date_key,
            "action": action,
            "contract_symbol": leg.contract_symbol,
            "side": leg.side,
            "strike": leg.strike,
            "expiry": leg.expiry,
            "quantity": qty,
            "mid_price": mid,
            "iv": iv,
            "premium": premium,
            "commission": commission,
            "cash_after": cash,
        }
    )
    return cash


def _rejection_row(date_key: str, leg: OptionLeg, reason: str) -> dict[str, Any]:
    return {
        "date": date_key,
        "contract_symbol": leg.contract_symbol,
        "side": leg.side,
        "strike": leg.strike,
        "expiry": leg.expiry,
        "quantity": int(leg.quantity),
        "reason": reason,
    }


def _open_option_views(
    open_options: dict[str, OpenOption],
) -> MappingProxyType[str, OpenOptionView]:
    return MappingProxyType(
        {
            symbol: OpenOptionView(
                side=position.side,
                strike=position.strike,
                expiry=position.expiry,
                quantity=position.quantity,
                avg_entry_mid=position.avg_entry_mid,
            )
            for symbol, position in open_options.items()
        }
    )


def _execute_option_structure(
    *,
    legs: list[OptionLeg],
    chain: pd.DataFrame | None,
    open_options: dict[str, OpenOption],
    cash: float,
    preferences: Preferences,
    date_key: str,
    option_trades: list[dict[str, Any]],
    rejected_orders: list[dict[str, Any]],
) -> float:
    """Fill all legs or reject the complete structure."""
    rejection_reason: str | None = None
    quotes: list[FillQuote] = []
    for leg in legs:
        if leg.reduce_only:
            existing = open_options.get(leg.contract_symbol)
            if existing is None or existing.quantity * int(leg.quantity) >= 0:
                rejection_reason = "no_open_position"
                break
        quote = _fill_quote(chain, leg)
        quotes.append(quote)
        if quote.reject_reason is not None:
            rejection_reason = quote.reject_reason
            break

    new_symbols = {
        leg.contract_symbol for leg in legs if leg.contract_symbol not in open_options and not leg.reduce_only
    }
    if (
        rejection_reason is None
        and len(open_options) + len(new_symbols) > preferences.risk.max_open_positions
    ):
        rejection_reason = "risk_max_open_positions"

    if rejection_reason is not None:
        reason = f"structure_rejected:{rejection_reason}" if len(legs) > 1 else rejection_reason
        rejected_orders.extend(_rejection_row(date_key, leg, reason) for leg in legs)
        return cash

    for leg, quote in zip(legs, quotes, strict=True):
        cash = _apply_option_fill(
            open_options,
            leg,
            mid=quote.mid,
            iv=quote.iv,
            cash=cash,
            prefs=preferences,
            date_key=date_key,
            option_trades=option_trades,
        )
    return cash


def _stock_rejection(date_key: str, action: str, quantity: int, reason: str) -> dict[str, Any]:
    return {
        "date": date_key,
        "instrument": "stock",
        "action": action,
        "quantity": quantity,
        "reason": reason,
    }


def _execute_stock_order(
    *,
    action: str,
    quantity: int,
    price: float,
    cash: float,
    position: int,
    preferences: Preferences,
    date_key: str,
    trades: list[dict[str, Any]],
    rejected_orders: list[dict[str, Any]],
) -> tuple[float, int]:
    """Fill a stock order at ``price``; return updated ``(cash, position)``.

    Without ``risk.allow_short_stock`` a sell is capped at the held quantity. With it,
    sells may take the position below zero and buys cover before going long.
    """
    if quantity <= 0 or action not in ("buy", "sell"):
        return cash, position
    if action == "sell" and not preferences.risk.allow_short_stock:
        if position <= 0:
            rejected_orders.append(_stock_rejection(date_key, action, quantity, "no_position"))
            return cash, position
        quantity = min(quantity, position)

    signed = quantity if action == "buy" else -quantity
    new_position = position + signed
    commission = _stock_commission(quantity, preferences)
    slippage = price * (preferences.defaults.slippage_bps / 10_000.0) * quantity
    new_cash = cash - signed * price - commission - slippage

    reason: str | None = None
    if abs(new_position) > abs(position):
        current_equity = cash + position * price
        if abs(new_position) * price > current_equity * preferences.risk.max_position_pct:
            reason = "risk_max_position_pct"
    if reason is None and action == "buy" and not preferences.defaults.allow_negative_cash and new_cash < 0.0:
        reason = "insufficient_cash"
    if reason is not None:
        rejected_orders.append(_stock_rejection(date_key, action, quantity, reason))
        return cash, position

    trades.append(
        {
            "date": date_key,
            "action": action,
            "quantity": quantity,
            "price": price,
            "cash_after": new_cash,
        }
    )
    return new_cash, new_position


def _cash_flow_row(
    date_key: str, action: str, quantity: int, per_share: float, cash: float
) -> dict[str, Any]:
    return {
        "date": date_key,
        "action": action,
        "quantity": quantity,
        "price": per_share,
        "cash_after": cash,
    }


def _stock_fill_price(row: pd.Series, *, at_open: bool) -> float:
    close = float(row["close"])
    if not at_open:
        return close
    open_price = numeric_value(row.get("open"))
    return open_price if open_price > 0.0 else close


def _vol_on_date(vol_series: pd.DataFrame | None) -> dict[str, dict[str, float | None]]:
    if vol_series is None or vol_series.empty or "date" not in vol_series.columns:
        return {}
    lookup: dict[str, dict[str, float | None]] = {}
    for _, row in vol_series.iterrows():
        values: dict[str, float | None] = {}
        for column in ("iv30", "hv20", "iv_rank_252", "iv_pctile_252", "iv_hv_ratio"):
            number = pd.to_numeric(row.get(column), errors="coerce")
            values[column] = None if pd.isna(number) else float(number)
        lookup[str(row["date"])[:10]] = values
    return lookup


def run_backtest(
    strategy: Strategy,
    bars: pd.DataFrame,
    options_chain: pd.DataFrame,
    preferences: Preferences,
    earnings: pd.DataFrame | None = None,
    *,
    chain_by_date: Mapping[str, pd.DataFrame] | None = None,
    vol_series: pd.DataFrame | None = None,
) -> RunResult:
    """Run ``strategy`` over ``bars``.

    ``chain_by_date`` (from ``prepare_chain_by_date``) takes precedence over
    ``options_chain`` so repeated runs can share one index.
    """
    bars_sorted = bars.sort_values("date").reset_index(drop=True)
    if chain_by_date is None:
        chain_by_date = prepare_chain_by_date(options_chain)
    earnings_df = earnings if earnings is not None else pd.DataFrame()
    vol_by_date = _vol_on_date(vol_series)
    closes_by_date = {
        str(row["date"]): float(row["close"])
        for _, row in bars_sorted.iterrows()
    }
    expected_by_date: dict[str, pd.DataFrame] = {}
    expected_frames: list[pd.DataFrame] = []
    for date_key, chain in chain_by_date.items():
        if date_key not in closes_by_date:
            continue
        frame = calculate_expected_moves(
            chain,
            closes_by_date[date_key],
            skew_factor=preferences.expected_move.skew_factor,
            max_spread_pct=preferences.expected_move.max_spread_pct,
        )
        if frame.empty:
            continue
        expected_by_date[date_key] = frame
        artifact_frame = _select_expected_move_horizons(
            frame, preferences.expected_move.horizons_dte
        )
        artifact_frame.insert(0, "asof", date_key)
        expected_frames.append(artifact_frame)
    cash = float(preferences.defaults.starting_capital)
    position = 0
    open_options: dict[str, OpenOption] = {}
    trades: list[dict[str, Any]] = []
    option_trades: list[dict[str, Any]] = []
    rejected_orders: list[dict[str, Any]] = []
    equity_records: list[dict[str, Any]] = []
    r = float(preferences.defaults.risk_free_rate)
    next_open = preferences.defaults.fill_timing == "next_open"
    borrow_rate = float(preferences.risk.short_borrow_rate)
    last_fills: tuple[dict[str, Any], ...] = ()
    last_rejections: tuple[dict[str, Any], ...] = ()
    pending: StrategyDecision | None = None
    prev_date: date | None = None
    prev_close = 0.0

    def execute(
        decision: StrategyDecision, row: pd.Series, chain: pd.DataFrame | None, date_key: str
    ) -> None:
        nonlocal cash, position
        cash, position = _execute_stock_order(
            action=decision.action,
            quantity=int(decision.quantity),
            price=_stock_fill_price(row, at_open=next_open),
            cash=cash,
            position=position,
            preferences=preferences,
            date_key=date_key,
            trades=trades,
            rejected_orders=rejected_orders,
        )
        if decision.option_legs:
            cash = _execute_option_structure(
                legs=decision.option_legs,
                chain=chain,
                open_options=open_options,
                cash=cash,
                preferences=preferences,
                date_key=date_key,
                option_trades=option_trades,
                rejected_orders=rejected_orders,
            )

    for _, row in bars_sorted.iterrows():
        fills_before_bar = len(option_trades)
        rejections_before_bar = len(rejected_orders)
        date_key = str(row["date"])
        bar_date = parse_option_date(date_key)
        chain = chain_by_date.get(date_key)
        price = float(row["close"])
        earn_ctx = context_fields(date_key, earnings_df)
        vol_ctx = vol_by_date.get(date_key, {})

        # Settle expired options before strategy decisions
        expired = [sym for sym, pos in open_options.items() if bar_date >= pos.expiry]
        for sym in expired:
            pos = open_options.pop(sym)
            settlement = pos.quantity * 100.0 * intrinsic_value(pos.side, pos.strike, price)
            cash += settlement
            option_trades.append(
                {
                    "date": date_key,
                    "action": "expire",
                    "contract_symbol": sym,
                    "side": pos.side,
                    "strike": pos.strike,
                    "expiry": pos.expiry.isoformat(),
                    "quantity": -pos.quantity,  # closing quantity
                    "mid_price": intrinsic_value(pos.side, pos.strike, price),
                    "iv": pos.last_iv,
                    "premium": -settlement,
                    "commission": 0.0,
                    "cash_after": cash,
                }
            )

        if position < 0 and borrow_rate > 0.0 and prev_date is not None:
            days = max((bar_date - prev_date).days, 0)
            fee_per_share = prev_close * borrow_rate * days / 365.0
            if fee_per_share > 0.0:
                cash += position * fee_per_share
                trades.append(_cash_flow_row(date_key, "borrow_fee", position, fee_per_share, cash))

        dividend = numeric_value(row.get("dividends"))
        if dividend and position:
            cash += position * dividend
            trades.append(_cash_flow_row(date_key, "dividend", position, dividend, cash))

        if pending is not None:
            execute(pending, row, chain, date_key)
            pending = None
            last_fills = tuple(option_trades[fills_before_bar:])
            last_rejections = tuple(rejected_orders[rejections_before_bar:])

        context = StrategyContext(
            row=row,
            cash=cash,
            position=position,
            options_chain=chain,
            expected_moves=expected_by_date.get(date_key),
            days_to_next_earnings=earn_ctx["days_to_next_earnings"],
            days_since_last_earnings=earn_ctx["days_since_last_earnings"],
            next_earnings_date=earn_ctx["next_earnings_date"],
            earnings_timing=earn_ctx["earnings_timing"],
            open_options=_open_option_views(open_options),
            last_fills=last_fills,
            last_rejections=last_rejections,
            iv30=vol_ctx.get("iv30"),
            hv20=vol_ctx.get("hv20"),
            iv_rank_252=vol_ctx.get("iv_rank_252"),
            iv_pctile_252=vol_ctx.get("iv_pctile_252"),
            iv_hv_ratio=vol_ctx.get("iv_hv_ratio"),
        )
        decision = strategy.on_bar(context)
        if next_open:
            has_order = decision.action in ("buy", "sell") and int(decision.quantity) > 0
            pending = decision if has_order or decision.option_legs else None
            last_fills = ()
            last_rejections = ()
        else:
            execute(decision, row, chain, date_key)

        options_mtm = 0.0
        for sym, pos in open_options.items():
            mark = _option_mark(
                pos,
                chain=chain,
                contract_symbol=sym,
                spot=price,
                bar_date=bar_date,
                r=r,
            )
            options_mtm += pos.quantity * 100.0 * mark

        equity = cash + (position * price) + options_mtm
        equity_records.append(
            {
                "date": date_key,
                "cash": cash,
                "position": position,
                "close": price,
                "options_mtm": options_mtm,
                "equity": equity,
            }
        )
        if not next_open:
            last_fills = tuple(option_trades[fills_before_bar:])
            last_rejections = tuple(rejected_orders[rejections_before_bar:])
        prev_date = bar_date
        prev_close = price

    if pending is not None:
        final_key = str(bars_sorted["date"].iloc[-1])
        if pending.action in ("buy", "sell") and int(pending.quantity) > 0:
            rejected_orders.append(
                _stock_rejection(final_key, pending.action, int(pending.quantity), "no_next_bar")
            )
        rejected_orders.extend(_rejection_row(final_key, leg, "no_next_bar") for leg in pending.option_legs)

    return RunResult(
        trades=pd.DataFrame(trades),
        equity_curve=pd.DataFrame(equity_records),
        final_cash=cash,
        final_position=position,
        option_trades=pd.DataFrame(option_trades) if option_trades else pd.DataFrame(),
        rejected_orders=pd.DataFrame(rejected_orders) if rejected_orders else pd.DataFrame(),
        expected_moves=pd.concat(expected_frames, ignore_index=True) if expected_frames else pd.DataFrame(),
    )


def write_run_outputs(run_result: RunResult, run_dir: Path) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trades_path = run_dir / "trades.csv"
    equity_path = run_dir / "equity.parquet"
    if run_result.trades.empty:
        pd.DataFrame(columns=["date", "action", "quantity", "price", "cash_after"]).to_csv(
            trades_path, index=False
        )
    else:
        run_result.trades.to_csv(trades_path, index=False)
    run_result.equity_curve.to_parquet(equity_path, index=False)
    if not run_result.option_trades.empty:
        run_result.option_trades.to_csv(run_dir / "option_trades.csv", index=False)
    if not run_result.rejected_orders.empty:
        run_result.rejected_orders.to_csv(run_dir / "rejected_orders.csv", index=False)
    if not run_result.expected_moves.empty:
        run_result.expected_moves.to_parquet(run_dir / "expected_moves.parquet", index=False)
    return trades_path, equity_path
