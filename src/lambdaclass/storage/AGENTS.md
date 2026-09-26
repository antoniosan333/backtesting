# `storage` — agent context

## `DuckDBStore`

Root: `Preferences.paths.data_dir` (default `data/`).

| Method | Parquet path | Dedupe key |
|--------|----------------|------------|
| `write_bars` / `read_bars` | `data/stocks/<SYMBOL>.parquet` | `(symbol, date)` — `keep="last"` |
| `write_chain` / `read_chain` | `data/options/<SYMBOL>.parquet` | `(symbol, contract_symbol, asof)` — `keep="last"` |
| `write_earnings` / `read_earnings` | `data/earnings/<SYMBOL>.parquet` | `(symbol, earnings_date)` — `keep="last"` |

| `write_universe` / `read_universe` | `data/universe/<name>.parquet` + `data/universe/snapshots/<name>/<date>.parquet` | replaced on write (adds `list_date`) |
| `write_earnings_day` / `read_earnings_day` / `cached_earnings_days` | `data/earnings/calendar/<YYYY>/<date>.parquet` (all reporters that day; empty days too) | replaced on write |
| `read_earnings_calendar(start, end, symbols)` | glob over the calendar cache (`union_by_name`) | — |
| `write_earnings_events` / `read_earnings_events` | `data/earnings/events/<name>.parquet` | replaced on write |

`bar_date_range(symbol)` returns stored `(min, max)` bar dates. `read_bars_many` / `read_earnings_many` read many symbols in one DuckDB query; prefer them in loops, since each `duckdb.connect()` costs several milliseconds. Dataset `name`s must match `^[a-z][a-z0-9_]{0,63}$` (they become file names). See [ADR-0010](../../../docs/decisions/0010-weekly-options-earnings-universe.md).

Bar columns: `date`, `open`, `high`, `low`, `close` (split-adjusted, not dividend-adjusted), `volume`, `dividends` (cash per share on the ex-date). `read_bars` backfills a missing `dividends` column with `0.0` and sets `bars.attrs["dividends_backfilled"] = True`.

Earnings columns: `symbol`, `earnings_date`, `timing` (`BMO`\|`AMC`\|`unknown`), `source`, `fetched_at`. See [ADR-0006](../../../docs/decisions/0006-earnings-calendar.md).

On append: read existing if present, `concat`, `drop_duplicates`, sort, write.

Reads use ephemeral `duckdb.connect()` + `read_parquet(?)` with optional `WHERE` on `date` / `asof` / `earnings_date`.

Dirs created in ctor: `stocks/`, `options/`, `earnings/`.

OptionsDX **normalized** Parquet layout is **not** under this class — it lives under `data/optionsdx/normalized/...` (see `data_adapters/optionsdx_normalize.py` and root AGENTS).

Parent: [AGENTS.md](../../../AGENTS.md).
