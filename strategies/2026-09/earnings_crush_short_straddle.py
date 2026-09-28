from __future__ import annotations

from datetime import date

from lambdaclass.options import preset_short_straddle
from lambdaclass.strategies.base import OptionLeg, Strategy, StrategyContext, StrategyDecision


class StrategyImpl(Strategy):
    """Short ATM straddle AFTER earnings to capture post-event IV crush.

    Enters when days_since_last_earnings == enter_days_after.
    Exits when days_since_last_earnings >= enter_days_after + hold_days.
    """

    name = "earnings_crush_short_straddle"
    params = {
        "lots": 1,
        "enter_days_after": 1,
        "hold_days": 5,
        "min_dte": 7,
        "min_iv_rank": None,
    }

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        bar_date = str(context.row.get("date", ""))[:10]

        if context.open_options:
            since = context.days_since_last_earnings
            exit_target = int(self.params["enter_days_after"]) + int(self.params["hold_days"])
            if since is not None and since >= exit_target:
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
                    metadata={"reason": "crush_exit", "days_since": since},
                )
            return StrategyDecision(action="hold")

        since = context.days_since_last_earnings
        if since is None or since != int(self.params["enter_days_after"]):
            return StrategyDecision(action="hold")

        # Optional IV rank filter
        min_ivr = self.params.get("min_iv_rank")
        if min_ivr is not None and context.iv_rank_252 is not None:
            if float(context.iv_rank_252) < float(min_ivr):
                return StrategyDecision(action="hold", metadata={"reason": "iv_rank_too_low"})

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

        legs = preset_short_straddle(context.options_chain, chosen, spot, lots=int(self.params["lots"]))
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
            metadata={
                "reason": "crush_entry",
                "days_since": since,
            },
        )