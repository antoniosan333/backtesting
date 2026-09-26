"""Validated overrides for ``Strategy.params`` (``run --param`` / ``sweep --grid``)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from lambdaclass.strategies.base import Strategy

PARAM_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


class ParamError(ValueError):
    """An override that does not match the strategy's declared params."""


def parse_assignments(items: Iterable[str], *, flag: str = "--param") -> dict[str, str]:
    """Parse ``key=value`` strings, rejecting malformed or repeated keys."""
    parsed: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not PARAM_KEY_PATTERN.match(key):
            raise ParamError(f"{flag} expects key=value, got {item!r}")
        if key in parsed:
            raise ParamError(f"{flag} {key} given more than once")
        parsed[key] = value.strip()
    return parsed


def coerce_param(name: str, value: Any, default: Any) -> Any:
    """Convert ``value`` to the type of the strategy's default for ``name``."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ParamError(f"{name} expects a boolean, got {value!r}")
    if isinstance(default, int):
        if isinstance(value, bool):
            raise ParamError(f"{name} expects an integer, got {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        try:
            number = Decimal(str(value).strip())
        except InvalidOperation:
            raise ParamError(f"{name} expects an integer, got {value!r}") from None
        if not number.is_finite() or number != number.to_integral_value():
            raise ParamError(f"{name} expects an integer, got {value!r}")
        return int(number)
    if isinstance(default, float):
        if isinstance(value, bool):
            raise ParamError(f"{name} expects a number, got {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ParamError(f"{name} expects a number, got {value!r}") from None
    if isinstance(default, str):
        return str(value)
    raise ParamError(f"{name} has a {type(default).__name__} default and cannot be overridden")


def resolve_params(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``defaults`` updated with type-checked ``overrides``."""
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        available = ", ".join(sorted(defaults)) or "(none)"
        raise ParamError(f"Unknown param(s) {', '.join(unknown)}; available: {available}")
    resolved = dict(defaults)
    for name, value in overrides.items():
        resolved[name] = coerce_param(name, value, defaults[name])
    return resolved


def apply_params(strategy: Strategy, overrides: Mapping[str, Any]) -> None:
    """Set ``strategy.params`` on the instance, leaving the class default untouched."""
    strategy.params = resolve_params(strategy.params, overrides)
