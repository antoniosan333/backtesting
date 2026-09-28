from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value)}")


def _dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"[{key}]")
            for child_key, child_value in value.items():
                lines.append(f"{child_key} = {_format_toml_value(child_value)}")
            lines.append("")
        else:
            lines.append(f"{key} = {_format_toml_value(value)}")
    if not lines:
        return ""
    return "\n".join(lines).rstrip() + "\n"


def _coerce_env_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _set_nested_value(target: dict[str, Any], keys: list[str], value: Any) -> None:
    cursor = target
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


SENSITIVE_SUBSTRINGS = ("token", "secret", "password", "apikey", "api_key", "auth", "credential")


def _redact_if_sensitive(key: str, value: Any) -> Any:
    lowered = key.lower()
    if any(part in lowered for part in SENSITIVE_SUBSTRINGS):
        return "***REDACTED***"
    return value


class DefaultsConfig(BaseModel):
    data_adapter: str = "yfinance"
    options_chain_source: str = Field(
        default="yfinance",
        description=(
            "yfinance: chain from data/options/<SYMBOL>.parquet (fetch); "
            "optionsdx: load normalized Parquet under [optionsdx].output_dir"
        ),
    )
    starting_capital: float = 100_000
    commission_per_contract: float = Field(default=0.65, ge=0.0, description="Options only, per contract")
    stock_commission_per_order: float = Field(default=0.0, ge=0.0)
    stock_commission_per_share: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = 2.0
    risk_free_rate: float = 0.04
    timezone: str = "America/New_York"


class PathsConfig(BaseModel):
    data_dir: str = "data"
    strategies_dir: str = "strategies"
    runs_dir: str = "runs"


class ReportingConfig(BaseModel):
    plot_theme: str = "plotly_dark"
    save_html: bool = True


class RiskConfig(BaseModel):
    max_position_pct: float = Field(default=0.10, ge=0.0)
    max_open_positions: int = Field(default=5, ge=1)


class OptionsDXNormalizeConfig(BaseModel):
    """Defaults for `lambdaclass normalize-optionsdx` (paths relative to repo root unless absolute)."""

    input_dir: str = "zRawData/optionsdx"
    output_dir: str = "data/optionsdx/normalized"
    reports_dir: str = "data/optionsdx/reports"


class ExpectedMoveConfig(BaseModel):
    skew_factor: float = Field(default=0.85, gt=0.0)
    max_spread_pct: float = Field(default=0.5, gt=0.0)
    horizons_dte: list[int] = Field(default_factory=lambda: [0, 7, 30])


class VolatilityConfig(BaseModel):
    hv_windows: list[int] = Field(default_factory=lambda: [10, 20, 30, 60, 252])
    iv_targets_dte: list[int] = Field(default_factory=lambda: [7, 30, 60, 90])
    lookback_days: int = Field(default=252, ge=2)
    max_strike_distance_pct: float = Field(default=0.15, gt=0.0)
    pre_days: int = Field(default=30, ge=1)
    post_days: int = Field(default=10, ge=1)
    history_start: str = "2019-01-01"


class Preferences(BaseModel):
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    optionsdx: OptionsDXNormalizeConfig = Field(default_factory=OptionsDXNormalizeConfig)
    expected_move: ExpectedMoveConfig = Field(default_factory=ExpectedMoveConfig)
    volatility: VolatilityConfig = Field(default_factory=VolatilityConfig)

    @classmethod
    def load(cls, path: Path, env_prefix: str = "LAMBDACLASS__") -> "Preferences":
        payload: dict[str, Any] = {}
        if path.exists():
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        for key, value in os.environ.items():
            if not key.startswith(env_prefix):
                continue
            suffix = key.removeprefix(env_prefix).lower()
            nested_keys = suffix.split("__")
            _set_nested_value(payload, nested_keys, _coerce_env_value(value))
        return cls.model_validate(payload)

    def to_toml(self) -> str:
        return _dump_toml(self.model_dump())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_toml(), encoding="utf-8")


def snapshot_preferences(
    preferences: Preferences,
    strategy_params: dict[str, Any],
    cli_overrides: dict[str, Any],
    output_path: Path,
) -> str:
    snapshot = build_snapshot_payload(preferences, strategy_params, cli_overrides)
    digest = compute_config_hash(snapshot)
    redacted_strategy_params = {
        key: _redact_if_sensitive(key, value) for key, value in strategy_params.items()
    }
    redacted_cli_overrides = {
        key: _redact_if_sensitive(key, value) for key, value in cli_overrides.items()
    }
    dump = _dump_toml(
        {
            "meta": {"config_hash": digest},
            "defaults": preferences.defaults.model_dump(),
            "paths": preferences.paths.model_dump(),
            "reporting": preferences.reporting.model_dump(),
            "risk": preferences.risk.model_dump(),
            "optionsdx": preferences.optionsdx.model_dump(),
            "expected_move": preferences.expected_move.model_dump(),
            "volatility": preferences.volatility.model_dump(),
            "strategy_params": {key: str(value) for key, value in redacted_strategy_params.items()},
            "cli_overrides": {key: str(value) for key, value in redacted_cli_overrides.items()},
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dump, encoding="utf-8")
    return digest


def build_snapshot_payload(
    preferences: Preferences,
    strategy_params: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    return {
        "preferences": preferences.model_dump(),
        "strategy_params": strategy_params,
        "cli_overrides": cli_overrides,
    }


def compute_config_hash(snapshot_payload: dict[str, Any]) -> str:
    serialized = json.dumps(snapshot_payload, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]


DEFAULT_PREFERENCES = Preferences()
