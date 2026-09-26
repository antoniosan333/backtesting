from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from lambdaclass.config import Preferences
from lambdaclass.earnings.calendar import context_fields
from lambdaclass.options.expected_move import expected_moves as calculate_expected_moves
from lambdaclass.options.pricing import (
    black_scholes_price,
    intrinsic_value,
    parse_option_date,
    safe_option_mid,
    years_between,
)
from lambdaclass.strategies.base import OptionLeg, Strategy, StrategyContext


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


def _chain_row_for_contract(chain: pd.DataFrame | None, contract_symbol: str) -> pd.Series | None:
    if chain is None or chain.empty:
        return None
    matches = chain[chain["contract_symbol"] == contract_symbol]
    if matches.empty:
        return None
    return matches.iloc[0]


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
    iv = float(pd.to_numeric(row.get("implied_volatility"), errors="coerce") or 0.0)
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
        iv = float(pd.to_numeric(row.get("implied_volatility"), errors="coerce") or 0.0)
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
        closed = existing.quantity
        remaining = qty + closed  # e.g. existing +1, leg -1 → 0; existing +1, leg -2 → -1
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
            action = "close"
        _ = closed
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
            existing.avg_entry_mid = (
                existing.avg_entry_mid * abs(existing.quantity) + mid * abs(qty)
            ) / abs(total_qty)
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


def run_backtest(
    strategy: Strategy,
    bars: pd.DataFrame,
    options_chain: pd.DataFrame,
    preferences: Preferences,
    earnings: pd.DataFrame | None = None,
) -> RunResult:
    bars_sorted = bars.sort_values("date").reset_index(drop=True)
    chain_by_date = (
        {
            str(key): frame.reset_index(drop=True)
            for key, frame in options_chain.groupby("asof")
        }
        if not options_chain.empty
        else {}
    )
    earnings_df = earnings if earnings is not None else pd.DataFrame()
    closes_by_date = {
        str(row["date"]): float(row["close"])
        for _, row in bars_sorted.iterrows()
    }
    expected_by_date: dict[str, pd.DataFrame] = {}
    expected_frames: list[pd.DataFrame] = []
    for date_key, chain in chain_by_date.items():
        if date_key not in closes_by_date:
            continue
        frame = calculate_expected_moves(chain, closes_by_date[date_key])
        if frame.empty:
            continue
        frame.insert(0, "asof", date_key)
        expected_by_date[date_key] = frame
        expected_frames.append(frame)
    cash = float(preferences.defaults.starting_capital)
    position = 0
    open_options: dict[str, OpenOption] = {}
    trades: list[dict[str, Any]] = []
    option_trades: list[dict[str, Any]] = []
    rejected_orders: list[dict[str, Any]] = []
    equity_records: list[dict[str, Any]] = []
    r = float(preferences.defaults.risk_free_rate)

    for _, row in bars_sorted.iterrows():
        date_key = str(row["date"])
        bar_date = parse_option_date(date_key)
        chain = chain_by_date.get(date_key)
        price = float(row["close"])
        earn_ctx = context_fields(date_key, earnings_df)

        # Settle expired options before strategy decisions
        expired = [
            sym for sym, pos in open_options.items() if bar_date >= pos.expiry
        ]
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
        )
        decision = strategy.on_bar(context)
        qty = int(decision.quantity)
        if decision.action == "buy" and qty > 0:
            total_cost = (price * qty) + _stock_commission(qty, preferences)
            slippage = price * (preferences.defaults.slippage_bps / 10_000.0) * qty
            cash -= total_cost + slippage
            position += qty
            trades.append(
                {
                    "date": date_key,
                    "action": "buy",
                    "quantity": qty,
                    "price": price,
                    "cash_after": cash,
                }
            )
        elif decision.action == "sell" and qty > 0 and position > 0:
            executed = min(qty, position)
            proceeds = (price * executed) - _stock_commission(executed, preferences)
            slippage = price * (preferences.defaults.slippage_bps / 10_000.0) * executed
            cash += proceeds - slippage
            position -= executed
            trades.append(
                {
                    "date": date_key,
                    "action": "sell",
                    "quantity": executed,
                    "price": price,
                    "cash_after": cash,
                }
            )

        if decision.option_legs:
            for leg in decision.option_legs:
                quote = _fill_quote(chain, leg)
                if quote.reject_reason is not None:
                    rejected_orders.append(
                        {
                            "date": date_key,
                            "contract_symbol": leg.contract_symbol,
                            "side": leg.side,
                            "strike": leg.strike,
                            "expiry": leg.expiry,
                            "quantity": int(leg.quantity),
                            "reason": quote.reject_reason,
                        }
                    )
                    continue
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
