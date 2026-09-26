from __future__ import annotations

from lambdaclass.options import preset_straddle
from lambdaclass.strategies.base import OptionLeg, Strategy, StrategyContext, StrategyDecision


class StrategyImpl(Strategy):
    name = "options_straddle"
    params = {
        "lots": 1,
    }

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        if context.open_options:
            return StrategyDecision(action="hold")

        if context.options_chain is None or context.options_chain.empty:
            return StrategyDecision(action="hold")

        expiries = context.options_chain["expiry"].dropna().unique()
        if len(expiries) == 0:
            return StrategyDecision(action="hold")
        expiry = str(sorted(expiries)[0])
        spot = float(context.row.get("close", 100))
        legs = preset_straddle(context.options_chain, expiry, spot, lots=self.params["lots"])
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
        return StrategyDecision(
            action="hold",
            option_legs=option_legs,
            metadata={"expiry": expiry, "spot": spot},
        )
