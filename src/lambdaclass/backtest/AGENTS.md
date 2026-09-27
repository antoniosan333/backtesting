# `backtest` — agent context

## `run_backtest`

Inputs: `Strategy`, sorted bars `DataFrame`, options chain `DataFrame`, `Preferences`, optional `earnings` DataFrame.

- Cash starts at `preferences.defaults.starting_capital`.
- Options grouped by `asof` (string key); chain for bar date passed into `StrategyContext`.
- Earnings calendar (if provided) fills `days_to_next_earnings`, `days_since_last_earnings`, `next_earnings_date`, `earnings_timing` via `lambdaclass.earnings.calendar.context_fields`.
- **Stock buy**: `price * qty + stock_commission(qty) + slippage` from `slippage_bps`, where stock commission is `stock_commission_per_order + per_share * qty` (both default `0.0`). `commission_per_contract` is options-only.
- Stock buys enforce `[risk].max_position_pct` and available cash unless `defaults.allow_negative_cash` is enabled.
- **Stock sell**: min(requested qty, position); proceeds minus commission and slippage.
- **Options ledger** (see [ADR-0005](../../../docs/decisions/0005-options-engine-accounting.md)):
  - Open positions keyed by `contract_symbol` (`OpenOption`: side, strike, expiry, qty, avg entry mid, last IV).
  - A decision's legs fill atomically at chain mid (+ slip on premium + `commission_per_contract`); one invalid leg rejects the complete structure.
  - `reduce_only=True` legs cannot create or reverse a position. `StrategyContext.open_options` is an immutable ledger view; `last_fills` / `last_rejections` expose prior-bar results.
  - New contracts enforce `[risk].max_open_positions`.
  - A leg with no priceable chain row is **rejected**, not filled: `_fill_quote` reports `no_chain_for_date` / `contract_not_in_chain` / `non_positive_mid` and the order lands in `RunResult.rejected_orders`.
  - Each bar: settle expired (`bar_date >= expiry`) at intrinsic into cash; mark open contracts to chain mid or BSM fallback (`lambdaclass.options.pricing`, `defaults.risk_free_rate`).
  - Equity: `cash + position * close + options_mtm` (`options_mtm` column on equity curve).

Returns `RunResult`: `trades`, `equity_curve`, `final_cash`, `final_position`, `option_trades`, `rejected_orders`.

## `write_run_outputs`

Writes under caller-provided `run_dir`:

- `trades.csv` (empty schema if no trades)
- `equity.parquet`
- `option_trades.csv` when there were option fills / settlements
- `rejected_orders.csv` when any option leg could not be priced

CLI additionally writes `metrics.json`, `config.snapshot.toml`, `run.log` (includes `option_trades=N` and `rejected_orders=N`), optional `report.html` — see `cli.py` `run` command. `run` warns when the chain does not cover the bar window or when orders were rejected; `--fail-on-rejected-orders` exits non-zero.

Run directory layout: `runs/<UTC-YYYY-MM>/<strategy.name>/<run_id>/`.

Parent: [AGENTS.md](../../../AGENTS.md).
