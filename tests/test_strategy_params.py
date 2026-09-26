from __future__ import annotations

from typing import Any

import pytest

from lambdaclass.config import Preferences, build_snapshot_payload, compute_config_hash
from lambdaclass.strategies.base import Strategy, StrategyContext, StrategyDecision
from lambdaclass.strategies.params import (
    ParamError,
    apply_params,
    coerce_param,
    parse_assignments,
    resolve_params,
)


class _Tunable(Strategy):
    name = "tunable"
    params: dict[str, Any] = {"lots": 1, "width": 10.0, "enabled": True, "mode": "fast"}

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        return StrategyDecision()


def test_parse_assignments_splits_on_first_equals() -> None:
    assert parse_assignments(["lots=2", "mode=a=b", " width = 7.5 "]) == {
        "lots": "2",
        "mode": "a=b",
        "width": "7.5",
    }


@pytest.mark.parametrize("item", ["lots", "=2", "2lots=1", "bad-key=1"])
def test_parse_assignments_rejects_malformed(item: str) -> None:
    with pytest.raises(ParamError):
        parse_assignments([item])


def test_parse_assignments_rejects_repeated_key() -> None:
    with pytest.raises(ParamError, match="more than once"):
        parse_assignments(["lots=1", "lots=2"])


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        ("2", 1, 2),
        (3.0, 1, 3),
        ("7.5", 10.0, 7.5),
        ("2", 10.0, 2.0),
        (4, 10.0, 4.0),
        ("true", False, True),
        ("No", True, False),
        ("1", False, True),
        ("slow", "fast", "slow"),
    ],
)
def test_coerce_param_follows_default_type(value: Any, default: Any, expected: Any) -> None:
    result = coerce_param("p", value, default)
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    ("value", "default"),
    [("2.5", 1), (2.5, 1), ("abc", 1.0), ("maybe", True), (True, 1), (False, 1.0)],
)
def test_coerce_param_rejects_wrong_type(value: Any, default: Any) -> None:
    with pytest.raises(ParamError):
        coerce_param("p", value, default)


def test_coerce_param_rejects_non_scalar_default() -> None:
    with pytest.raises(ParamError, match="cannot be overridden"):
        coerce_param("p", "1", [1, 2])


def test_resolve_params_rejects_unknown_keys() -> None:
    with pytest.raises(ParamError, match="Unknown param\\(s\\) lotz; available: enabled, lots, mode, width"):
        resolve_params(_Tunable.params, {"lotz": "2"})


def test_apply_params_leaves_class_default_untouched() -> None:
    strategy = _Tunable()
    apply_params(strategy, {"lots": "3"})

    assert strategy.params["lots"] == 3
    assert strategy.params["width"] == 10.0
    assert _Tunable.params["lots"] == 1
    assert _Tunable().params["lots"] == 1


def test_overridden_params_change_config_hash() -> None:
    prefs = Preferences()
    base = compute_config_hash(build_snapshot_payload(prefs, resolve_params(_Tunable.params, {}), {}))
    tuned = compute_config_hash(
        build_snapshot_payload(prefs, resolve_params(_Tunable.params, {"lots": "2"}), {})
    )

    assert base != tuned
