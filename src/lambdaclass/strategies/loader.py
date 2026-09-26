"""Load ``StrategyImpl`` from a strategy file."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from lambdaclass.strategies.base import Strategy


class StrategyLoadError(ValueError):
    """The file cannot be imported or does not define a valid ``StrategyImpl``."""


def load_strategy_class(path: Path) -> type[Strategy]:
    """Execute the strategy file and return its ``StrategyImpl`` class.

    Strategy files are trusted application code; this runs them.
    """
    module_name = f"strategy_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise StrategyLoadError(f"Could not load strategy module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategy_cls = getattr(module, "StrategyImpl", None)
    if strategy_cls is None:
        raise StrategyLoadError("Strategy file must define StrategyImpl class")
    if not isinstance(strategy_cls, type) or not issubclass(strategy_cls, Strategy):
        raise StrategyLoadError("StrategyImpl must inherit lambdaclass.strategies.base.Strategy")
    return strategy_cls
