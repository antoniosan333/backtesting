"""Fill timing (same_close vs next_open) and short stock accounting."""

from __future__ import annotations

import pandas as pd
import pytest

from lambdaclass.backtest.engine import run_backtest
from lambdaclass.config import DEFAULT_PREFERENCES, Preferences
from lambdaclass.strategies.base import OptionLeg, Strategy, StrategyContext, StrategyDecision


class Scripted(Strategy):
    """Return a pre-set decision per bar and record every context seen."""

    name = "scripted"

    def __init__(self, decisions: list[StrategyDecision]) -> None:
        self._decisions = decisions
        self.contexts: list[StrategyContext] = []

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        index = len(self.contexts)
        self.contexts.append(context)
        if index < len(self._decisions):
            return self._decisions[index]
        return StrategyDecision(action="hold")


def _prefs(
    *,
    fill_timing: str = "same_close",
    allow_short: bool = False,
    borrow_rate: float = 0.0,
) -> Preferences:
    prefs = DEFAULT_PREFERENCES.model_copy(deep=True)
    prefs.defaults.slippage_bps = 0.0
    prefs.defaults.commission_per_contract = 0.0
    prefs.defaults.fill_timing = fill_timing  # type: ignore[assignment]
    prefs.risk.max_position_pct = 1.0
    prefs.risk.allow_short_stock = allow_short
    prefs.risk.short_borrow_rate = borrow_rate
    return prefs


def _bars(rows: list[tuple[str, float, float]], dividends: list[float] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {"date": d, "open": o, "high": max(o, c), "low": min(o, c), "close": c, "volume": 1_000}
            for d, o, c in rows
        ]
    )
    if dividends is not None:
        frame["dividends"] = dividends
    return frame


BARS = _bars(
    [
        ("2024-01-02", 99.0, 100.0),
        ("2024-01-03", 104.0, 105.0),
        ("2024-01-04", 109.0, 110.0),
    ]
)
BUY = StrategyDecision(action="buy", quantity=10)
SELL = StrategyDecision(action="sell", quantity=10)
HOLD = StrategyDecision(action="hold")


def _chain() -> pd.DataFrame:
    rows = []
    for asof, mid in (("2024-01-02", 2.0), ("2024-01-03", 3.0), ("2024-01-04", 4.0)):
        rows.append(
            {
                "contract_symbol": "ZZZ_C100",
                "side": "call",
                "strike": 100.0,
                "last_price": mid,
                "bid": mid - 0.05,
                "ask": mid + 0.05,
                "implied_volatility": 0.25,
                "expiry": "2024-06-21",
                "asof": asof,
            }
        )
    return pd.DataFrame(rows)


def test_same_close_fills_on_decision_bar_close() -> None:
    result = run_backtest(Scripted([BUY]), BARS, pd.DataFrame(), _prefs())

    trade = result.trades.iloc[0]
    assert trade["date"] == "2024-01-02"
    assert trade["price"] == pytest.approx(100.0)


def test_next_open_fills_at_following_bar_open() -> None:
    strategy = Scripted([BUY])
    result = run_backtest(strategy, BARS, pd.DataFrame(), _prefs(fill_timing="next_open"))

    trade = result.trades.iloc[0]
    assert trade["date"] == "2024-01-03"
    assert trade["price"] == pytest.approx(104.0)
    assert strategy.contexts[1].position == 10
    assert result.equity_curve["position"].tolist() == [0, 10, 10]


def test_next_open_option_leg_fills_at_next_chain_and_reports_fill() -> None:
    leg = OptionLeg("ZZZ_C100", "call", 100.0, "2024-06-21", 1)
    strategy = Scripted([StrategyDecision(action="hold", option_legs=[leg])])
    result = run_backtest(strategy, BARS, _chain(), _prefs(fill_timing="next_open"))

    fill = result.option_trades.iloc[0]
    assert fill["date"] == "2024-01-03"
    assert fill["mid_price"] == pytest.approx(3.0)
    assert [row["contract_symbol"] for row in strategy.contexts[1].last_fills] == ["ZZZ_C100"]
    assert strategy.contexts[2].last_fills == ()


def test_next_open_order_on_last_bar_is_rejected() -> None:
    leg = OptionLeg("ZZZ_C100", "call", 100.0, "2024-06-21", 1)
    strategy = Scripted([HOLD, HOLD, StrategyDecision(action="buy", quantity=1, option_legs=[leg])])
    result = run_backtest(strategy, BARS, _chain(), _prefs(fill_timing="next_open"))

    assert result.trades.empty
    assert result.option_trades.empty
    assert result.rejected_orders["reason"].tolist() == ["no_next_bar", "no_next_bar"]
    assert result.rejected_orders["date"].tolist() == ["2024-01-04", "2024-01-04"]


def test_sell_without_position_is_rejected_when_shorting_disabled() -> None:
    result = run_backtest(Scripted([SELL]), BARS, pd.DataFrame(), _prefs())

    assert result.trades.empty
    rejection = result.rejected_orders.iloc[0]
    assert rejection["reason"] == "no_position"
    assert rejection["instrument"] == "stock"


def test_sell_is_capped_at_held_quantity_when_shorting_disabled() -> None:
    oversell = StrategyDecision(action="sell", quantity=25)
    result = run_backtest(Scripted([BUY, oversell]), BARS, pd.DataFrame(), _prefs())

    assert result.trades["quantity"].tolist() == [10, 10]
    assert result.final_position == 0


def test_short_then_cover_round_trip() -> None:
    result = run_backtest(Scripted([SELL, HOLD, BUY]), BARS, pd.DataFrame(), _prefs(allow_short=True))

    assert result.equity_curve["position"].tolist() == [-10, -10, 0]
    # Short at 100, cover at 110: lose $100.
    assert result.final_cash == pytest.approx(100_000.0 - 100.0)
    assert result.equity_curve["equity"].tolist() == pytest.approx([100_000.0, 99_950.0, 99_900.0])


def test_short_pays_borrow_fee_and_dividends() -> None:
    bars = _bars(
        [("2024-01-02", 100.0, 100.0), ("2024-01-05", 100.0, 100.0)],
        dividends=[0.0, 0.5],
    )
    result = run_backtest(Scripted([SELL]), bars, pd.DataFrame(), _prefs(allow_short=True, borrow_rate=0.365))

    # 3 calendar days at 36.5%/yr on $1,000 = $3.00; dividend 10 × $0.50 = $5.00.
    rows = result.trades.set_index("action")
    assert rows.loc["borrow_fee", "quantity"] == -10
    assert rows.loc["borrow_fee", "price"] == pytest.approx(0.30)
    assert rows.loc["dividend", "price"] == pytest.approx(0.5)
    assert result.final_cash == pytest.approx(100_000.0 + 1_000.0 - 3.0 - 5.0)


def test_short_respects_max_position_pct() -> None:
    prefs = _prefs(allow_short=True)
    prefs.risk.max_position_pct = 0.005
    result = run_backtest(Scripted([SELL]), BARS, pd.DataFrame(), prefs)

    assert result.trades.empty
    assert result.rejected_orders.iloc[0]["reason"] == "risk_max_position_pct"


def test_covering_short_is_not_blocked_by_position_limit() -> None:
    prefs = _prefs(allow_short=True)
    # $1,000 short fits a $1,010 limit; by the cover bar it is $1,100 notional.
    prefs.risk.max_position_pct = 0.0101
    result = run_backtest(Scripted([SELL, HOLD, BUY]), BARS, pd.DataFrame(), prefs)

    assert result.rejected_orders.empty
    assert result.final_position == 0
