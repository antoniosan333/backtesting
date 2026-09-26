from __future__ import annotations

from lambdaclass.reporting.dashboard.option_strategies import preset_strangle
from lambdaclass.strategies.base import OptionLeg, Strategy, StrategyContext, StrategyDecision


class StrategyImpl(Strategy):
    name = "options_strangle"
    params = {
        "lots": 1,
        "offset": 10.0,
    }

    def __init__(self) -> None:
        self._open_expiry: str | None = None

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        bar_date = str(context.row.get("date", ""))[:10]
        if self._open_expiry is not None:
            if bar_date >= self._open_expiry:
                self._open_expiry = None
            return StrategyDecision(action="hold")

        if context.options_chain is None or context.options_chain.empty:
            return StrategyDecision(action="hold")

        expiries = context.options_chain["expiry"].dropna().unique()
        if len(expiries) == 0:
            return StrategyDecision(action="hold")
        expiry = str(sorted(expiries)[0])
        spot = float(context.row.get("close", 100))
        legs = preset_strangle(
            context.options_chain,
            expiry,
            spot,
            offset=self.params["offset"],
            lots=self.params["lots"],
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
        self._open_expiry = expiry
        return StrategyDecision(
            action="hold",
            option_legs=option_legs,
            metadata={"expiry": expiry, "spot": spot, "offset": self.params["offset"]},
        )
