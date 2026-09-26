"""Retry with linear backoff for flaky network adapters."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

Result = TypeVar("Result")


def with_retry(
    fn: Callable[[], Result],
    *,
    retries: int = 3,
    delay_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Result:
    """Call ``fn`` up to ``retries`` times, sleeping ``delay_seconds * attempt`` between tries.

    Re-raises the last exception when every attempt fails.
    """
    if retries < 1:
        raise ValueError("retries must be >= 1")
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == retries:
                raise
            sleep(delay_seconds * attempt)
    raise AssertionError("unreachable")
