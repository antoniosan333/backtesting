# `storage` — agent context

## `DuckDBStore`

Root: `Preferences.paths.data_dir` (default `data/`).

| Method | Parquet path | Dedupe key |
|--------|----------------|------------|
| `write_bars` / `read_bars` | `data/stocks/<SYMBOL>.parquet` | `(symbol, date)` — `keep="last"` |
| `write_chain` / `read_chain` | `data/options/<SYMBOL>.parquet` | `(symbol, contract_symbol, asof)` — `keep="last"` |
| `write_earnings` / `read_earnings` | `data/earnings/<SYMBOL>.parquet` | `(symbol, earnings_date)` — `keep="last"` |

Earnings columns: `symbol`, `earnings_date`, `timing` (`BMO`\|`AMC`\|`unknown`), `source`, `fetched_at`. See [ADR-0006](../../../docs/decisions/0006-earnings-calendar.md).

On append: read existing if present, `concat`, `drop_duplicates`, sort, write.

Reads use ephemeral `duckdb.connect()` + `read_parquet(?)` with optional `WHERE` on `date` / `asof` / `earnings_date`.

Dirs created in ctor: `stocks/`, `options/`, `earnings/`.

OptionsDX **normalized** Parquet layout is **not** under this class — it lives under `data/optionsdx/normalized/...` (see `data_adapters/optionsdx_normalize.py` and root AGENTS).

Parent: [AGENTS.md](../../../AGENTS.md).
