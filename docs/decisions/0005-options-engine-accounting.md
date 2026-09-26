# ADR-0005: Options accounting in the backtest engine

**Status:** Accepted

## Context

Strategies can emit `option_legs` on each bar, but equity was previously `cash + stock_position × close` only. Premium left cash on entry and never returned — no open ledger, mark-to-market, or expiry settlement — so options-strategy metrics were meaningless ([root AGENTS.md](../../AGENTS.md) continual-learning note).

## Decision

`run_backtest` maintains an open-options ledger keyed by `contract_symbol`:

- **Fill:** mid from the day's chain (`safe_option_mid`); cash changes by `qty × 100 × mid` plus `slippage_bps` on premium magnitude and `commission_per_contract × |qty|`. Opposite-sign quantity nets/closes; same-sign adds with average entry mid.
- **Reject:** a leg with no observable price is **not** filled. `_fill_quote` returns a reject reason (`no_chain_for_date`, `contract_not_in_chain`, `non_positive_mid`) and the order is recorded in `RunResult.rejected_orders` instead. Filling at a zero mid would open a free position that is then marked to BSM, which fabricates P&L — this is the single most dangerous failure mode because the default `yfinance` chain covers at most one bar date.
- **Mark:** each bar, open contracts mark to chain mid when present; otherwise Black–Scholes via `lambdaclass.options.pricing` using last known IV, `prefs.defaults.risk_free_rate`, and calendar time to expiry.
- **Settle:** on the first bar with `bar_date >= expiry`, cash += `qty × 100 × intrinsic(side, strike, close)`; position removed; `option_trades` row with `action=expire`.
- **Equity:** `cash + stock_position × close + options_mtm` (new `options_mtm` column on the equity curve).
- **Artifacts:** `write_run_outputs` writes `option_trades.csv` and `rejected_orders.csv` when non-empty; CLI `run.log` includes `option_trades=N` and `rejected_orders=N`. `run` warns on stderr when the chain does not cover the window or when orders were rejected, and `--fail-on-rejected-orders` turns that into a non-zero exit.

Commission is split by instrument: `commission_per_contract` applies to options only, while stock uses `stock_commission_per_order` + `stock_commission_per_share` (both default `0.0`). Previously the per-contract rate was charged per *share*, which made stock backtests pay ~$65 on a 100-share order.

Pricing primitives live in `lambdaclass.options.pricing` (not under `reporting.dashboard`) so the engine does not depend on the Streamlit layer.

## Consequences

- **Pros:** Options P&L and equity become usable for multi-leg strategies; dashboard Run tab can show option fills.
- **Cons / known gaps:** European cash settlement only (no early exercise or assignment); no margin model; BSM fallback when a contract disappears from the chain *after* it was opened at a real price; fill at mid (no bid/ask adverse selection beyond flat slip bps).
- **Breaking:** runs recorded before this change may show option P&L from zero-price fills, and stock runs were over-charged commission. Their `metrics.json` is not comparable with new runs.
- **Watch:** Strategies must not re-enter every bar; sample options strategies track one open expiry until settlement.

## Out of scope

American exercise, dividends/borrow (`q=0`), portfolio margin, earnings-calendar strategies (follow-on work).
