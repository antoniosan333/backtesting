from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _format_toml_value(value: Any, key_path: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, int):
        return str(value)
    raise TypeError(f"Unsupported TOML value for {key_path}: {value!r} ({type(value).__name__})")


def _dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"[{key}]")
            for child_key, child_value in value.items():
                lines.append(f"{child_key} = {_format_toml_value(child_value, f'{key}.{child_key}')}")
            lines.append("")
        else:
            lines.append(f"{key} = {_format_toml_value(value, key)}")
    if not lines:
        return ""
    return "\n".join(lines).rstrip() + "\n"


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
    allow_negative_cash: bool = False


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


class Preferences(BaseModel):
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    optionsdx: OptionsDXNormalizeConfig = Field(default_factory=OptionsDXNormalizeConfig)

    @classmethod
    def load(cls, path: Path, env_prefix: str = "LAMBDACLASS__") -> Preferences:
        payload: dict[str, Any] = {}
        if path.exists():
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        for key, value in os.environ.items():
            if not key.startswith(env_prefix):
                continue
            suffix = key.removeprefix(env_prefix).lower()
            nested_keys = suffix.split("__")
            _set_nested_value(payload, nested_keys, value)
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
        key: _redact_if_sensitive(key, value) for key, value in snapshot["cli_overrides"].items()
    }
    dump = _dump_toml(
        {
            "meta": {"config_hash": digest},
            "defaults": preferences.defaults.model_dump(),
            "paths": preferences.paths.model_dump(),
            "reporting": preferences.reporting.model_dump(),
            "risk": preferences.risk.model_dump(),
            "optionsdx": preferences.optionsdx.model_dump(),
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
        "cli_overrides": {key: value for key, value in cli_overrides.items() if value is not None},
    }


def compute_config_hash(snapshot_payload: dict[str, Any]) -> str:
    serialized = json.dumps(snapshot_payload, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]


DEFAULT_PREFERENCES = Preferences()
