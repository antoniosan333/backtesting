# `strategies` — agent context

## Base API

`Strategy` (`base.py`): `name`, `params`, `on_bar(context: StrategyContext) -> StrategyDecision`.

`StrategyContext`: bar row, `cash`, `position`, optional `options_chain` for that bar date, plus earnings fields when a calendar was passed to `run_backtest`: `days_to_next_earnings`, `days_since_last_earnings`, `next_earnings_date`, `earnings_timing` (`BMO`/`AMC`/`unknown`). All earnings fields are `None` when no calendar.

Sample earnings strategies: `strategies/*/earnings_long_straddle.py`, `earnings_short_iron_condor.py` — enter on `days_to_next_earnings == N`, exit on `days_since_last_earnings >= M`.

## Scaffolding

`scaffolder.py`:

- `ensure_month_dir(strategies_dir, now)` → `strategies/<YYYY-MM>/`, ensures `_template.py` exists.
- `scaffold_strategy(strategies_dir, strategy_name, now)` → writes `<YYYY-MM>/<strategy_name>.py` from `TEMPLATE` if missing.

Strategy files must define `StrategyImpl` with `name` and `params` dict; default template is a minimal buy/hold example.

## CLI alignment

- Strategy **name** validated: `^[A-Za-z][A-Za-z0-9_]*$`.
- Lookup: `strategies_dir.glob(f"*/{strategy_name}.py")` — monthly folders; latest match wins.
- Resolved file must remain under `strategies_dir` (path confinement).

Parent: [AGENTS.md](../../../AGENTS.md).
