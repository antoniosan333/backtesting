"""Engine options ledger: fills, MTM, settlement, netting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lambdaclass.backtest.engine import prepare_chain_by_date, run_backtest, write_run_outputs
from lambdaclass.config import DEFAULT_PREFERENCES, Preferences
from lambdaclass.strategies.base import OptionLeg, Strategy, StrategyContext, StrategyDecision


def _prefs(**kwargs: float) -> Preferences:
    d = DEFAULT_PREFERENCES.model_copy(deep=True)
    for k, v in kwargs.items():
        setattr(d.defaults, k, v)
    return d


def _bars(closes: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": d, "open": c, "high": c, "low": c, "close": c, "volume": 1_000_000} for d, c in closes]
    )


def _chain_row(
    asof: str,
    *,
    contract: str,
    side: str,
    strike: float,
    expiry: str,
    mid: float,
    iv: float = 0.25,
) -> dict[str, object]:
    return {
        "contract_symbol": contract,
        "side": side,
        "strike": strike,
        "last_price": mid,
        "bid": mid - 0.05,
        "ask": mid + 0.05,
        "implied_volatility": iv,
        "open_interest": 100.0,
        "volume": 10.0,
        "expiry": expiry,
        "asof": asof,
        "symbol": "ZZZ",
        "underlying_last": 100.0,
        "dte": float((pd.Timestamp(expiry) - pd.Timestamp(asof)).days),
    }


def test_prepare_chain_indexes_contracts_by_date() -> None:
    chain = pd.DataFrame(
        [
            _chain_row(
                "2026-01-01",
                contract="ZZZ_C100",
                side="call",
                strike=100.0,
                expiry="2026-02-01",
                mid=2.0,
            )
        ]
    )

    prepared = prepare_chain_by_date(chain)

    assert prepared["2026-01-01"].index.name == "contract_symbol"
    assert prepared["2026-01-01"].loc["ZZZ_C100", "strike"] == 100.0


class LongCallOnce(Strategy):
    name = "long_call_once"
    params: dict = {}

    def __init__(self, contract: str, expiry: str, strike: float) -> None:
        self._done = False
        self._contract = contract
        self._expiry = expiry
        self._strike = strike

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        if self._done:
            return StrategyDecision(action="hold")
        self._done = True
        return StrategyDecision(
            action="hold",
            option_legs=[
                OptionLeg(self._contract, "call", self._strike, self._expiry, 1),
            ],
        )


class ShortPutOnce(Strategy):
    name = "short_put_once"
    params: dict = {}

    def __init__(self, contract: str, expiry: str, strike: float) -> None:
        self._done = False
        self._contract = contract
        self._expiry = expiry
        self._strike = strike

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        if self._done:
            return StrategyDecision(action="hold")
        self._done = True
        return StrategyDecision(
            action="hold",
            option_legs=[
                OptionLeg(self._contract, "put", self._strike, self._expiry, -1),
            ],
        )


class OpenThenClose(Strategy):
    name = "open_close"
    params: dict = {}

    def __init__(self, contract: str, expiry: str, strike: float) -> None:
        self._n = 0
        self._contract = contract
        self._expiry = expiry
        self._strike = strike

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        self._n += 1
        if self._n == 1:
            return StrategyDecision(
                action="hold",
                option_legs=[OptionLeg(self._contract, "call", self._strike, self._expiry, 1)],
            )
        if self._n == 3:
            return StrategyDecision(
                action="hold",
                option_legs=[OptionLeg(self._contract, "call", self._strike, self._expiry, -1)],
            )
        return StrategyDecision(action="hold")


def test_long_call_marked_then_settled_itm(tmp_path: Path) -> None:
    expiry = "2026-01-05"
    contract = "ZZZ_C100"
    bars = _bars(
        [
            ("2026-01-01", 100.0),
            ("2026-01-02", 102.0),
            ("2026-01-03", 104.0),
            ("2026-01-04", 106.0),
            ("2026-01-05", 110.0),  # settle ITM: intrinsic = 10
        ]
    )
    chain = pd.DataFrame(
        [
            _chain_row("2026-01-01", contract=contract, side="call", strike=100.0, expiry=expiry, mid=4.0),
            _chain_row("2026-01-02", contract=contract, side="call", strike=100.0, expiry=expiry, mid=5.0),
            _chain_row("2026-01-03", contract=contract, side="call", strike=100.0, expiry=expiry, mid=6.0),
            _chain_row("2026-01-04", contract=contract, side="call", strike=100.0, expiry=expiry, mid=7.0),
            # no row on expiry day — settles via intrinsic
        ]
    )
    prefs = _prefs(starting_capital=100_000.0, commission_per_contract=0.0, slippage_bps=0.0)
    result = run_backtest(LongCallOnce(contract, expiry, 100.0), bars, chain, prefs)

    assert not result.option_trades.empty
    opens = result.option_trades[result.option_trades["action"] == "open"]
    expires = result.option_trades[result.option_trades["action"] == "expire"]
    assert len(opens) == 1
    assert len(expires) == 1

    # Day 1: paid 4.0 * 100 = 400; MTM = 400 → equity ≈ 100_000
    eq1 = float(result.equity_curve.iloc[0]["equity"])
    assert eq1 == pytest.approx(100_000.0, abs=1.0)

    # Day 2 mark 5.0: cash 99_600 + mtm 500 = 100_100
    eq2 = float(result.equity_curve.iloc[1]["equity"])
    assert eq2 == pytest.approx(100_100.0, abs=1.0)

    # After settle ITM at 110: cash = 99_600 + 10*100 = 100_600; no open MTM
    assert result.final_cash == pytest.approx(100_600.0, abs=1.0)
    assert float(result.equity_curve.iloc[-1]["options_mtm"]) == pytest.approx(0.0)

    write_run_outputs(result, tmp_path)
    assert (tmp_path / "option_trades.csv").is_file()

    shared = run_backtest(
        LongCallOnce(contract, expiry, 100.0),
        bars,
        pd.DataFrame(),
        prefs,
        chain_by_date=prepare_chain_by_date(chain),
    )
    assert shared.equity_curve.equals(result.equity_curve)
    assert shared.option_trades.equals(result.option_trades)


def test_short_put_settled_otm_keeps_premium() -> None:
    expiry = "2026-01-03"
    contract = "ZZZ_P90"
    bars = _bars(
        [
            ("2026-01-01", 100.0),
            ("2026-01-02", 100.0),
            ("2026-01-03", 100.0),  # OTM vs 90 put
        ]
    )
    chain = pd.DataFrame(
        [
            _chain_row("2026-01-01", contract=contract, side="put", strike=90.0, expiry=expiry, mid=1.50),
            _chain_row("2026-01-02", contract=contract, side="put", strike=90.0, expiry=expiry, mid=1.00),
        ]
    )
    prefs = _prefs(starting_capital=50_000.0, commission_per_contract=0.0, slippage_bps=0.0)
    result = run_backtest(ShortPutOnce(contract, expiry, 90.0), bars, chain, prefs)
    # Received 150 premium; OTM settle adds 0 → final cash 50_150
    assert result.final_cash == pytest.approx(50_150.0, abs=1.0)


def test_missing_chain_uses_bsm_fallback_mark() -> None:
    expiry = "2026-02-01"
    contract = "ZZZ_C100"
    bars = _bars(
        [
            ("2026-01-01", 100.0),
            ("2026-01-02", 100.0),
            ("2026-01-03", 100.0),
        ]
    )
    chain = pd.DataFrame(
        [
            _chain_row(
                "2026-01-01",
                contract=contract,
                side="call",
                strike=100.0,
                expiry=expiry,
                mid=5.0,
                iv=0.30,
            ),
            # missing mid-window rows → BSM fallback
        ]
    )
    prefs = _prefs(starting_capital=100_000.0, commission_per_contract=0.0, slippage_bps=0.0)
    result = run_backtest(LongCallOnce(contract, expiry, 100.0), bars, chain, prefs)
    mtm_mid = float(result.equity_curve.iloc[1]["options_mtm"])
    assert mtm_mid > 0.0


def test_netting_close_realizes_pnl() -> None:
    expiry = "2026-03-01"
    contract = "ZZZ_C100"
    bars = _bars(
        [
            ("2026-01-01", 100.0),
            ("2026-01-02", 100.0),
            ("2026-01-03", 100.0),
            ("2026-01-04", 100.0),
        ]
    )
    chain = pd.DataFrame(
        [
            _chain_row("2026-01-01", contract=contract, side="call", strike=100.0, expiry=expiry, mid=4.0),
            _chain_row("2026-01-02", contract=contract, side="call", strike=100.0, expiry=expiry, mid=5.0),
            _chain_row("2026-01-03", contract=contract, side="call", strike=100.0, expiry=expiry, mid=6.0),
            _chain_row("2026-01-04", contract=contract, side="call", strike=100.0, expiry=expiry, mid=6.0),
        ]
    )
    prefs = _prefs(starting_capital=100_000.0, commission_per_contract=0.0, slippage_bps=0.0)
    result = run_backtest(OpenThenClose(contract, expiry, 100.0), bars, chain, prefs)
    closes = result.option_trades[result.option_trades["action"] == "close"]
    assert len(closes) == 1
    # Paid 400, sold for 600 → cash 100_200; no open position
    assert float(result.equity_curve.iloc[-1]["options_mtm"]) == pytest.approx(0.0)
    assert result.final_cash == pytest.approx(100_200.0, abs=1.0)


def test_commission_and_slippage_on_option_fill() -> None:
    expiry = "2026-02-01"
    contract = "ZZZ_C100"
    bars = _bars([("2026-01-01", 100.0), ("2026-01-02", 100.0)])
    chain = pd.DataFrame(
        [
            _chain_row("2026-01-01", contract=contract, side="call", strike=100.0, expiry=expiry, mid=4.0),
            _chain_row("2026-01-02", contract=contract, side="call", strike=100.0, expiry=expiry, mid=4.0),
        ]
    )
    prefs = _prefs(
        starting_capital=100_000.0,
        commission_per_contract=0.65,
        slippage_bps=100.0,  # 1% of premium
    )
    result = run_backtest(LongCallOnce(contract, expiry, 100.0), bars, chain, prefs)
    row = result.option_trades.iloc[0]
    assert float(row["commission"]) == pytest.approx(0.65)
    # premium 400 + slip 4 + commission 0.65
    assert result.final_cash < 100_000.0 - 400.0


class CaptureExpectedMoves(Strategy):
    name = "capture_expected_moves"
    params: dict = {}

    def __init__(self) -> None:
        self.seen: list[pd.DataFrame | None] = []

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        self.seen.append(context.expected_moves)
        return StrategyDecision()


def test_expected_moves_are_available_to_strategy_and_written(tmp_path: Path) -> None:
    asof = "2026-01-01"
    expiry = "2026-02-01"
    chain = pd.DataFrame(
        [
            _chain_row(
                asof,
                contract=f"ZZZ_{side}",
                side=side,
                strike=100.0,
                expiry=expiry,
                mid=4.0,
            )
            for side in ("call", "put")
        ]
    )
    strategy = CaptureExpectedMoves()

    result = run_backtest(strategy, _bars([(asof, 100.0)]), chain, _prefs())

    assert strategy.seen[0] is not None
    assert strategy.seen[0].iloc[0]["expiry"] == expiry
    assert not result.expected_moves.empty
    write_run_outputs(result, tmp_path)
    written = pd.read_parquet(tmp_path / "expected_moves.parquet")
    assert written["asof"].tolist() == [asof]


def test_expected_moves_are_none_without_a_chain() -> None:
    strategy = CaptureExpectedMoves()

    run_backtest(strategy, _bars([("2026-01-01", 100.0)]), pd.DataFrame(), _prefs())

    assert strategy.seen == [None]
