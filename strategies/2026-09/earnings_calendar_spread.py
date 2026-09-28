from __future__ import annotations

from datetime import date

from lambdaclass.options.presets import snap_to_chain
from lambdaclass.strategies.base import OptionLeg, Strategy, StrategyContext, StrategyDecision


class StrategyImpl(Strategy):
    """Short front-week straddle, long next-week straddle into earnings.

    Captures the earnings IV premium embedded in the front expiry.
    Front expires before or through earnings; back expires after.
    """

    name = "earnings_calendar_spread"
    params = {
        "lots": 1,
        "enter_days_before": 5,
        "exit_days_after": 1,
        "min_front_dte": 3,
        "min_back_dte": 10,
    }

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        bar_date = str(context.row.get("date", ""))[:10]

        if context.open_options:
            since = context.days_since_last_earnings
            if since is not None and since >= int(self.params["exit_days_after"]):
                close_legs = [
                    OptionLeg(
                        contract_symbol=symbol,
                        side=position.side,
                        strike=position.strike,
                        expiry=position.expiry.isoformat(),
                        quantity=-position.quantity,
                        reduce_only=True,
                    )
                    for symbol, position in context.open_options.items()
                ]
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
        lots = int(self.params["lots"])
        bar_d = date.fromisoformat(bar_date)

        expiries = sorted({str(x) for x in context.options_chain["expiry"].dropna().unique()})
        # Find front expiry: closest expiry with min_front_dte
        front = None
        back = None
        for exp in expiries:
            try:
                dte = (date.fromisoformat(exp[:10]) - bar_d).days
            except ValueError:
                continue
            if front is None and dte >= int(self.params["min_front_dte"]):
                front = exp
            if dte >= int(self.params["min_back_dte"]):
                back = exp
                break
        if front is None or back is None or front == back:
            return StrategyDecision(action="hold")

        # Sell front straddle, buy back straddle
        legs = []
        for side in ("call", "put"):
            short_leg = snap_to_chain(context.options_chain, side, front, spot)
            long_leg = snap_to_chain(context.options_chain, side, back, spot)
            legs.append(OptionLeg(contract_symbol=short_leg.contract_symbol, side=short_leg.side, strike=short_leg.strike, expiry=short_leg.expiry, quantity=-lots))
            legs.append(OptionLeg(contract_symbol=long_leg.contract_symbol, side=long_leg.side, strike=long_leg.strike, expiry=long_leg.expiry, quantity=lots))

        return StrategyDecision(
            action="hold",
            option_legs=legs,
            metadata={
                "reason": "earnings_entry",
                "days_to": days_to,
                "front_expiry": front,
                "back_expiry": back,
                "earnings_date": context.next_earnings_date,
            },
        )