from __future__ import annotations

import re

SYMBOL_PATTERN = re.compile(r"^(?:\^[A-Z0-9][A-Z0-9.-]{0,10}|[A-Z0-9][A-Z0-9.-]{0,11})$")


def validate_symbol(symbol: str) -> str:
    """Return a normalized market symbol or reject unsafe file-name input."""
    normalized = symbol.upper()
    if ".." in normalized or not SYMBOL_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Invalid symbol: use 1-12 letters, numbers, dots, or hyphens, with an optional leading caret."
        )
    return normalized
