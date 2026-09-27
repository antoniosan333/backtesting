# `strategies` — agent context

## Base API

`Strategy` (`base.py`): `name`, `params`, `on_bar(context: StrategyContext) -> StrategyDecision`.

`StrategyContext`: bar row, `cash`, `position`, optional `options_chain`,
immutable `open_options`, prior-bar `last_fills` / `last_rejections`, plus
earnings fields. Use the engine ledger instead of shadow position state.

`OptionLeg.reduce_only=True` is required for close-only intent. A multi-leg
decision is atomic: if any leg is invalid, none fill.

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
