from __future__ import annotations

from datetime import date

from lambdaclass.reporting.dashboard.option_strategies import preset_iron_condor
from lambdaclass.strategies.base import OptionLeg, Strategy, StrategyContext, StrategyDecision


class StrategyImpl(Strategy):
    """Short iron condor into earnings; exit after IV crush."""

    name = "earnings_short_iron_condor"
    params = {
        "lots": 1,
        "width_inner": 10.0,
        "width_outer": 30.0,
        "enter_days_before": 1,
        "exit_days_after": 1,
        "min_dte": 7,
    }

    def __init__(self) -> None:
        self._open_legs: list[OptionLeg] = []

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        bar_date = str(context.row.get("date", ""))[:10]

        if self._open_legs:
            since = context.days_since_last_earnings
            if since is not None and since >= int(self.params["exit_days_after"]):
                close_legs = [
                    OptionLeg(
                        contract_symbol=lg.contract_symbol,
                        side=lg.side,
                        strike=lg.strike,
                        expiry=lg.expiry,
                        quantity=-lg.quantity,
                    )
                    for lg in self._open_legs
                ]
                self._open_legs = []
                return StrategyDecision(
                    action="hold",
                    option_legs=close_legs,
                    metadata={"reason": "earnings_exit", "days_since": since},
                )
            return StrategyDecision(action="hold")

        days_to = context.days_to_next_earnings
        if days_to is None or days_to != int(self.params["enter_days_before"]):
            return StrategyDecision(action="hold")
        if context.options_chain is None or context.options_chain.empty:
            return StrategyDecision(action="hold")

        spot = float(context.row.get("close", 100))
        min_dte = int(self.params["min_dte"])
        expiries = sorted({str(x) for x in context.options_chain["expiry"].dropna().unique()})
        chosen = None
        bar_d = date.fromisoformat(bar_date)
        for exp in expiries:
            try:
                dte = (date.fromisoformat(exp[:10]) - bar_d).days
            except ValueError:
                continue
            if dte >= min_dte:
                chosen = exp
                break
        if chosen is None and expiries:
            chosen = expiries[-1]
        if chosen is None:
            return StrategyDecision(action="hold")

        legs = preset_iron_condor(
            context.options_chain,
            chosen,
            spot,
            width_inner=float(self.params["width_inner"]),
            width_outer=float(self.params["width_outer"]),
            lots=int(self.params["lots"]),
        )
        option_legs = [
            OptionLeg(
                contract_symbol=leg.contract_symbol,
                side=leg.side,
                strike=leg.strike,
                expiry=leg.expiry,
                quantity=leg.quantity,
            )
            for leg in legs
        ]
        self._open_legs = list(option_legs)
        return StrategyDecision(
            action="hold",
            option_legs=option_legs,
            metadata={
                "reason": "earnings_entry",
                "days_to": days_to,
                "earnings_date": context.next_earnings_date,
                "timing": context.earnings_timing,
            },
        )
