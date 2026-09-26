# ADR-0002: Options chain source for backtests

**Status:** Accepted

## Context

Historical options came from two places: yfinance-backed Parquet under `data/options/<SYMBOL>.parquet` (from `fetch`) and OptionsDX-normalized Parquet under `[optionsdx].output_dir`. Strategies receive `options_chain` keyed by bar date (`asof`). The engine already grouped any frame by `asof`; it did not prescribe a single storage layout beyond column compatibility with the yfinance export shape.

## Decision

Add `[defaults].options_chain_source` with values `yfinance` (default) or `optionsdx`, plus CLI override `lambdaclass run … --options-source …`. When `optionsdx`, load all normalized Parquet under `output_dir/<SYMBOL>/**/`, filter rows to bar dates derived from `quote_date`, and map columns to the same contract-level schema used by `YFinanceAdapter.get_option_chain` (including synthetic `contract_symbol` when OptionsDX 33-col rows omit it).

## Consequences

- Users must align stock bar dates with normalized `quote_date` for meaningful chain joins.
- OptionsDX reads prune the symbol tree to the requested year/month partitions, then use DuckDB projection and date filtering. Legacy unpartitioned trees fall back to a recursive scan.
- yfinance exposes only a current chain. Historical stock fetches skip the chain rather than labelling current quotes with a historical `asof`; current chains are stamped with today's date.
- Frozen run snapshots include the new default key under `[defaults]` in `config.snapshot.toml` via existing `snapshot_preferences` behavior.
