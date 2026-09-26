"""Fill validity: unpriceable option orders are rejected, stock commission is per order."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lambdaclass.backtest.engine import run_backtest, write_run_outputs
from lambdaclass.config import DEFAULT_PREFERENCES, Preferences
from lambdaclass.strategies.base import OptionLeg, Strategy, StrategyContext, StrategyDecision


def _prefs(**kwargs: float) -> Preferences:
    prefs = DEFAULT_PREFERENCES.model_copy(deep=True)
    for key, value in kwargs.items():
        setattr(prefs.defaults, key, value)
    return prefs


def _bars(closes: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": d, "open": c, "high": c, "low": c, "close": c, "volume": 1_000_000} for d, c in closes]
    )


def _chain_row(asof: str, *, contract: str, mid: float, bid: float | None = None, ask: float | None = None):
    return {
        "contract_symbol": contract,
        "side": "call",
        "strike": 100.0,
        "last_price": mid,
        "bid": mid - 0.05 if bid is None else bid,
        "ask": mid + 0.05 if ask is None else ask,
        "implied_volatility": 0.25,
        "open_interest": 100.0,
        "volume": 10.0,
        "expiry": "2026-06-01",
        "asof": asof,
        "symbol": "ZZZ",
    }


class BuyCallOnce(Strategy):
    name = "buy_call_once"
    params: dict = {}

    def __init__(self, contract: str = "ZZZ_C100") -> None:
        self._done = False
        self._contract = contract

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        if self._done:
            return StrategyDecision(action="hold")
        self._done = True
        return StrategyDecision(
            action="hold",
            option_legs=[OptionLeg(self._contract, "call", 100.0, "2026-06-01", 1)],
        )


class BuySharesOnce(Strategy):
    name = "buy_shares_once"
    params: dict = {}

    def __init__(self, shares: int) -> None:
        self._shares = shares

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        if context.position:
            return StrategyDecision(action="hold")
        return StrategyDecision(action="buy", quantity=self._shares)


class OpenIncompleteStructure(Strategy):
    name = "open_incomplete_structure"
    params: dict = {}

    def __init__(self) -> None:
        self._bar = 0

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        self._bar += 1
        quantity = 1 if self._bar == 1 else -1 if self._bar == 2 else 0
        if quantity == 0:
            return StrategyDecision()
        return StrategyDecision(
            option_legs=[
                OptionLeg(
                    "ZZZ_C100",
                    "call",
                    100.0,
                    "2026-06-01",
                    quantity,
                    reduce_only=quantity < 0,
                ),
                OptionLeg(
                    "ZZZ_P100",
                    "put",
                    100.0,
                    "2026-06-01",
                    quantity,
                    reduce_only=quantity < 0,
                ),
            ]
        )


class ObserveLedger(Strategy):
    name = "observe_ledger"
    params: dict = {}

    def __init__(self) -> None:
        self.seen_quantities: list[int] = []

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        self.seen_quantities.append(sum(position.quantity for position in context.open_options.values()))
        if len(self.seen_quantities) == 1:
            return StrategyDecision(option_legs=[OptionLeg("ZZZ_C100", "call", 100.0, "2026-06-01", 1)])
        return StrategyDecision()


def test_order_rejected_when_no_chain_for_the_window() -> None:
    bars = _bars([("2026-01-01", 100.0), ("2026-01-02", 130.0)])
    prefs = _prefs(starting_capital=100_000.0, commission_per_contract=0.0, slippage_bps=0.0)

    result = run_backtest(BuyCallOnce(), bars, pd.DataFrame(), prefs)

    assert result.option_trades.empty
    assert len(result.rejected_orders) == 1
    assert result.rejected_orders.iloc[0]["reason"] == "no_chain_for_date"
    # No phantom position: equity never departs from starting capital.
    assert float(result.equity_curve["equity"].iloc[-1]) == pytest.approx(100_000.0)
    assert float(result.equity_curve["options_mtm"].iloc[-1]) == pytest.approx(0.0)


def test_order_rejected_when_contract_absent_from_chain() -> None:
    bars = _bars([("2026-01-01", 100.0)])
    chain = pd.DataFrame([_chain_row("2026-01-01", contract="ZZZ_C105", mid=3.0)])
    prefs = _prefs(starting_capital=100_000.0)

    result = run_backtest(BuyCallOnce("ZZZ_C100"), bars, chain, prefs)

    assert result.option_trades.empty
    assert result.rejected_orders.iloc[0]["reason"] == "contract_not_in_chain"
    assert result.final_cash == pytest.approx(100_000.0)


def test_order_rejected_when_quote_has_no_positive_mid() -> None:
    bars = _bars([("2026-01-01", 100.0)])
    chain = pd.DataFrame([_chain_row("2026-01-01", contract="ZZZ_C100", mid=0.0, bid=0.0, ask=0.0)])
    prefs = _prefs(starting_capital=100_000.0)

    result = run_backtest(BuyCallOnce(), bars, chain, prefs)

    assert result.rejected_orders.iloc[0]["reason"] == "non_positive_mid"
    assert result.final_cash == pytest.approx(100_000.0)


def test_multileg_decision_is_rejected_atomically() -> None:
    bars = _bars([("2026-01-01", 100.0), ("2026-01-02", 100.0)])
    chain = pd.DataFrame(
        [
            _chain_row("2026-01-01", contract="ZZZ_C100", mid=2.0),
            _chain_row("2026-01-02", contract="ZZZ_C100", mid=2.0),
            _chain_row("2026-01-02", contract="ZZZ_P100", mid=2.0),
        ]
    )

    result = run_backtest(OpenIncompleteStructure(), bars, chain, _prefs())

    assert result.option_trades.empty
    assert set(result.rejected_orders["reason"]) == {
        "structure_rejected:contract_not_in_chain",
        "structure_rejected:no_open_position",
    }
    assert float(result.equity_curve.iloc[-1]["options_mtm"]) == 0.0


def test_strategy_context_exposes_engine_option_ledger() -> None:
    bars = _bars([("2026-01-01", 100.0), ("2026-01-02", 100.0)])
    chain = pd.DataFrame(
        [
            _chain_row("2026-01-01", contract="ZZZ_C100", mid=2.0),
            _chain_row("2026-01-02", contract="ZZZ_C100", mid=2.0),
        ]
    )
    strategy = ObserveLedger()

    run_backtest(strategy, bars, chain, _prefs())

    assert strategy.seen_quantities == [0, 1]


def test_stock_buy_over_position_limit_is_rejected() -> None:
    bars = _bars([("2026-01-01", 100.0)])
    prefs = _prefs(starting_capital=100_000.0, slippage_bps=0.0)

    result = run_backtest(BuySharesOnce(200), bars, pd.DataFrame(), prefs)

    assert result.trades.empty
    assert result.rejected_orders.iloc[0]["reason"] == "risk_max_position_pct"


def test_rejected_orders_written_to_run_dir(tmp_path: Path) -> None:
    bars = _bars([("2026-01-01", 100.0)])
    result = run_backtest(BuyCallOnce(), bars, pd.DataFrame(), _prefs())

    write_run_outputs(result, tmp_path)

    written = pd.read_csv(tmp_path / "rejected_orders.csv")
    assert written.iloc[0]["contract_symbol"] == "ZZZ_C100"


def test_stock_commission_is_per_order_not_per_share() -> None:
    bars = _bars([("2026-01-01", 100.0)])
    prefs = _prefs(
        starting_capital=100_000.0,
        stock_commission_per_order=1.0,
        stock_commission_per_share=0.0,
        commission_per_contract=0.65,
        slippage_bps=0.0,
    )

    result = run_backtest(BuySharesOnce(100), bars, pd.DataFrame(), prefs)

    # 100 shares at 100 plus a single $1 order fee — not 100 * commission_per_contract.
    assert result.final_cash == pytest.approx(100_000.0 - 10_000.0 - 1.0)


def test_stock_commission_per_share_scales_with_size() -> None:
    bars = _bars([("2026-01-01", 100.0)])
    prefs = _prefs(
        starting_capital=100_000.0,
        stock_commission_per_order=0.5,
        stock_commission_per_share=0.005,
        slippage_bps=0.0,
    )
    prefs.risk.max_position_pct = 1.0

    result = run_backtest(BuySharesOnce(200), bars, pd.DataFrame(), prefs)

    assert result.final_cash == pytest.approx(100_000.0 - 20_000.0 - (0.5 + 200 * 0.005))


def test_stock_trades_are_free_by_default() -> None:
    bars = _bars([("2026-01-01", 100.0)])
    prefs = _prefs(starting_capital=100_000.0, slippage_bps=0.0)

    result = run_backtest(BuySharesOnce(10), bars, pd.DataFrame(), prefs)

    assert result.final_cash == pytest.approx(99_000.0)
