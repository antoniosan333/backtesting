# `data_adapters` — agent context

## Adapter contract

`MarketDataAdapter` (`base.py`): Protocol with `get_stock_bars(symbol, start, end) -> DataFrame` and `get_option_chain(symbol, asof, expiry=None) -> DataFrame`.

Implemented: `yfinance_adapter.py` (`yfinance`). CLI resolves adapter name from `preferences.defaults.data_adapter`.

Also: `get_earnings_dates(symbol, limit=…)` → raw `earnings_date` / `timing` frame (empty if unavailable). Normalize via `lambdaclass.earnings.calendar.normalize_earnings_frame` before `DuckDBStore.write_earnings`. CLI: `fetch-earnings`.

## OptionsDX pipeline

Order: **parse** (`optionsdx_parser.py`) → **quality** (`optionsdx_quality.py`) → **normalize** (`optionsdx_normalize.py`, `run_normalize`).

- Raw input: default `zRawData/optionsdx/*.txt` (prefs `[optionsdx].input_dir`).
- **33-column** paired call/put rows are the common OptionsDX export shape.
- **28-column** files (e.g. per-contract EOD like `btc_eod_202106.txt`) are a separate schema path in the parser.

Normalization writes Parquet under `data/optionsdx/normalized/<SYMBOL>/<YYYY>/<MM>/<stem>.parquet`, per-file JSON under `reports/files/`, run summaries under `reports/runs/`, last run pointer `state/optionsdx_normalize_state.json` (skipped on `--dry-run`).

**Backtest bridge:** `optionsdx_chain_loader.load_normalized_optionsdx_chain` reads those Parquet files for a symbol, filters to bar `quote_date`s, and returns columns aligned with `YFinanceAdapter.get_option_chain` (plus `symbol`) for `run_backtest`.

## Quality flags (`apply_quality_rules`)

Pipe-separated string in `quality_flags` (sorted, unique):

| Flag | Meaning |
|------|---------|
| `DATE_MISMATCH` | `quote_date` vs date from `quote_unixtime` disagree |
| `INVALID_PRICE` | Negative bid or ask nulled |
| `CROSSED_MARKET` | bid > ask (both nulled) |
| `NEGATIVE_IV` | IV nulled |
| `NO_TRUSTED_MID` | Could not derive mid from bid/ask (non-crossed case) |

Derived booleans: `is_crossed_market`, `is_valid_iv`, `is_valid_price`, `is_kept`. Rows are kept; bad fields nulled where appropriate.

`FileQualityReport` aggregates counts (blank IV, negative IV nulled, crossed market, etc.) for JSON reports.

See root [AGENTS.md](../../../AGENTS.md) for CLI flags (`--dry-run`, gates, rates).
