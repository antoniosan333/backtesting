from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class OptionLeg:
    """Single options leg for strategy decisions."""

    contract_symbol: str
    side: str  # "call" | "put"
    strike: float
    expiry: str  # YYYY-MM-DD
    quantity: int  # positive for long, negative for short


@dataclass
class StrategyContext:
    row: pd.Series
    cash: float
    position: int
    options_chain: pd.DataFrame | None = None
    expected_moves: pd.DataFrame | None = None
    days_to_next_earnings: int | None = None
    days_since_last_earnings: int | None = None
    next_earnings_date: str | None = None
    earnings_timing: str | None = None


@dataclass
class StrategyDecision:
    action: str = "hold"
    quantity: int = 0
    option_legs: list[OptionLeg] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    name: str = "base"
    params: dict[str, Any] = {}

    @abstractmethod
    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        raise NotImplementedError

    def on_chain(self, context: StrategyContext) -> StrategyDecision:
        return StrategyDecision()
