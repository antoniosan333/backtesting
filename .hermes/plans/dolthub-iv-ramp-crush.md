# Plan: DoltHub Options Data Integration for IV Ramp/Crush Analysis

## Goal

Fetch historical EOD option chain + volatility data from DoltHub's free
`dolthub/options` database, convert it to the pipeline's native format, and
feed it into the existing `build_vol_series()` → `earnings_cycle.py` analysis
engine to measure implied volatility ramp before earnings and crush after.

## Data Sources

### DoltHub `dolthub/options` (free, no API key)

API: `https://www.dolthub.com/api/v1alpha1/dolthub/options/master?q={SQL}`

**`option_chain` table** (13 cols):
`date, act_symbol, expiration, strike, call_put, bid, ask, vol, delta, gamma, theta, vega, rho`

**`volatility_history` table** (16 cols):
`date, act_symbol, hv_current, hv_week_ago, hv_month_ago, hv_year_high, hv_year_high_date, hv_year_low, hv_year_low_date, iv_current, iv_week_ago, iv_month_ago, iv_year_high, iv_year_high_date, iv_year_low, iv_year_low_date`

### yfinance (already wired)
- Stock bars: `lambdaclass fetch SYMBOL`
- Earnings calendar: `lambdaclass fetch-earnings SYMBOL`

## Architecture

```
DoltHub SQL API
       │
       ▼
┌──────────────────────┐
│ dolthub_adapter.py   │  ← new: query API, return DataFrames
│                      │
│  get_option_chain()  │  → matches CHAIN_COLUMNS schema
│  get_vol_history()   │  → new: IV/HV daily series
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ DuckDBStore          │  ← existing: write_chain / write_bars
│                      │
│  data/options/       │     SYM.parquet (chain)
│  data/stocks/        │     SYM.parquet (bars from yfinance)
│  data/earnings/      │     SYM.parquet (calendar from yfinance)
│  data/cache/vol/     │     SYM.parquet (vol history — new path)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ build_vol_series()   │  ← existing: volatility/series.py
│                      │
│  bars + chain +      │
│  earnings → daily    │
│  IV30, HV, term      │
│  structure, event    │
│  vol                 │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ earnings_cycle.py    │  ← existing
│                      │
│  align_events()      │
│  event_table()       │  → ramp_pct, crush_pct
│  aggregate_cycle()   │  → median IV path
│  cycle_summary()     │  → median ramp/crush
└──────────────────────┘
```

## Implementation Steps

### Phase 1 — DoltHub Adapter (new file)

**File**: `src/lambdaclass/data_adapters/dolthub_adapter.py`

1. **`DoltHubAdapter` class** implementing `MarketDataAdapter` protocol:
   - `get_option_chain(symbol, start, end) -> DataFrame`
     - SQL: `SELECT * FROM option_chain WHERE act_symbol='{SYM}' AND date >= '{start}' AND date <= '{end}'`
     - Paginate if needed (API may limit rows per response)
     - Map columns to `CHAIN_COLUMNS`:
       ```
       act_symbol    → symbol
       date          → asof
       expiration    → expiry
       strike        → strike (float)
       call_put      → side ("call"/"put")
       bid           → bid (float)
       ask           → ask (float)
       vol           → implied_volatility (float)
       delta/gamma/theta/vega/rho → pass through (optional)
       ```
     - Compute missing fields:
       - `dte`: `(expiration - date).days`
       - `underlying_last`: fill from stock bars `close` on matching date
       - `contract_symbol`: synthetic `{SYM}_{EXP}{C/P}_{STRIKE*1000}`
       - `last_price`: NaN (not in DoltHub)
       - `open_interest`: 0.0 (not in DoltHub)
       - `volume`: 0.0 (not in DoltHub)
     - Synthesize `SYMBOL_eod_YYYYMM.txt` in OptionsDX 33-col format OR
       write directly to `DuckDBStore.write_chain()`

   - `get_vol_history(symbol, start, end) -> DataFrame`
     - SQL: `SELECT * FROM volatility_history WHERE act_symbol='{SYM}' AND date >= '{start}' AND date <= '{end}'`
     - Return raw DataFrame with columns mapped:
       ```
       iv_current    → iv30
       hv_current    → hv20
       iv_year_high  → iv_year_high
       iv_year_low   → iv_year_low
       etc.
       ```

2. **Throttling**: DoltHub API is slow for large queries. Add:
   - Configurable page size (default 5000 rows per request)
   - Retry with backoff on timeout (3 retries, 5s/10s/20s)
   - Optional `dolt clone` local mode (see Phase 3)

3. **Tests**: `tests/test_dolthub_adapter.py`
   - Mock API responses (don't hit network in tests)
   - Column mapping correctness
   - DTE computation
   - Empty result handling
   - Pagination logic

### Phase 2 — CLI Commands

**File**: `src/lambdaclass/cli.py` (extend existing)

1. **`fetch-dolthub SYMBOL [--start] [--end] [--vol-only]`**
   - Fetch option chain from DoltHub for date range
   - Merge `underlying_last` from stored stock bars (warn if bars missing)
   - Write to `DuckDBStore.write_chain()` → `data/options/<SYMBOL>.parquet`
   - If `--vol-only`: fetch only `volatility_history` → `data/cache/vol/<SYMBOL>.parquet`
   - Default date range: last 3 years or stored bar range

2. **`fetch-dolthub-universe [--start] [--end] [--vol-only] [--pilot]`**
   - Iterate symbols from `data/weekly_options_stocks.csv` (or PILOT_SYMBOLS with `--pilot`)
   - For each: fetch chain (or vol history) + fetch bars (if missing) + fetch earnings (if missing)
   - Throttle: 2s delay between symbols
   - Progress: print `SYM: chain_rows=N vol_rows=N`
   - Resume: skip symbols that already have data for the date range

3. **`vol SYMBOL [--start] [--end]`** (wire the existing volatility branch code)
   - Read stored bars + chain + earnings + vol_history
   - Call `build_vol_series()` → write to `data/cache/vol/<SYMBOL>.parquet`
   - If `volatility_history` available, use it directly instead of computing IV from chain

4. **`vol-study [--symbols] [--start] [--end]`**
   - Run `align_events()` + `event_table()` + `aggregate_cycle()` across symbols
   - Output: `data/earnings/events/iv_ramp_crush_study.parquet`
   - Print `cycle_summary()` to console

### Phase 3 — Local Dolt Clone (optional, for speed)

**File**: `src/lambdaclass/data_adapters/dolthub_local.py`

1. **`dolt clone dolthub/options`** — download full database (~GB scale)
2. **`DoltLocalAdapter`** — query local MySQL-compatible Dolt server
   - Same interface as `DoltHubAdapter` but instant queries
   - `dolt sql-server` running on localhost:3306
3. **CLI flag**: `--source dolthub-api` (default) or `--source dolthub-local`
4. **Fallback**: if local server not running, fall back to API

### Phase 4 — Vol Series Integration

**File**: `src/lambdaclass/volatility/series.py` (extend existing)

1. **`build_vol_series_from_history()`** — new function
   - When `volatility_history` data is available, use it directly:
     - `iv_current` → `iv30` (no term structure interpolation needed)
     - `hv_current` → `hv20`
     - `iv_year_high` / `iv_year_low` → `iv_rank_252` (pre-computed!)
   - Still needs earnings calendar for `days_to_next_earnings` context
   - Falls back to chain-based `build_vol_series()` if no vol history

2. **`build_vol_series()`** — extend to accept `vol_history` parameter
   - If `vol_history` provided, merge pre-computed IV/HV columns
   - If not, compute from chain as before
   - Either way, add earnings context fields

### Phase 5 — Analysis Pipeline

**File**: `src/lambdaclass/earnings_cycle_runner.py` (new)

1. **`run_earnings_iv_study(symbols, start, end) -> dict`**
   - For each symbol:
     a. Load bars, earnings, vol series (or chain)
     b. Call `build_vol_series()` or `build_vol_series_from_history()`
     c. Call `align_events(series, earnings)`
     d. Collect aligned events
   - Across all symbols:
     a. Concatenate aligned events
     b. `normalize()` → express IV as ratio to pre-event base
     c. `aggregate_cycle()` → median IV path per rel_day
     d. `event_table()` → per-event ramp_pct, crush_pct
     e. `cycle_summary()` → medians
   - Return summary + write results to Parquet

2. **CLI**: `lambdaclass iv-study [--symbols] [--start] [--end] [--pilot]`
   - Run `run_earnings_iv_study()`
   - Print summary table to console
   - Write `data/earnings/events/iv_ramp_crush_study.parquet`

### Phase 6 — Dashboard Volatility Tab

**File**: `src/lambdaclass/reporting/dashboard/app.py` (extend existing)

1. Wire the Volatility tab (code exists from merged branch but needs integration):
   - Load `data/cache/vol/<SYMBOL>.parquet`
   - Show IV vs HV chart with earnings markers
   - Show earnings cycle study (median IV path, IQR bands)
   - Show per-event table (ramp_pct, crush_pct, realized_vs_implied)

## Execution Order

```
Phase 1 (adapter)     ──┐
                        ├── Phase 4 (vol integration) ── Phase 5 (study) ── Phase 6 (dashboard)
Phase 2 (CLI)         ──┘
                        
Phase 3 (local dolt)  ── optional, can be done later for speed
```

## Field Mapping: DoltHub → Pipeline

### option_chain → CHAIN_COLUMNS (for DuckDBStore.write_chain)

| DoltHub | Pipeline column | Transform |
|---------|----------------|-----------|
| `date` | `asof` | string[:10] |
| `act_symbol` | `symbol` | upper() |
| `expiration` | `expiry` | string[:10] |
| `strike` | `strike` | float() |
| `call_put` | `side` | lower() → "call"/"put" |
| `bid` | `bid` | float() |
| `ask` | `ask` | float() |
| `vol` | `implied_volatility` | float() |
| — | `contract_symbol` | synthetic: `{SYM}_{EXP}{C/P}_{STRIKE*1000}` |
| — | `last_price` | 0.0 (not available) |
| — | `open_interest` | 0.0 (not available) |
| — | `volume` | 0.0 (not available) |
| — | `underlying_last` | from stock bars `close` on matching date |
| — | `dte` | `(expiry - asof).days` |

### volatility_history → vol series (for data/cache/vol/)

| DoltHub | Pipeline column | Transform |
|---------|----------------|-----------|
| `date` | `date` | string[:10] |
| `act_symbol` | `symbol` | upper() |
| `iv_current` | `iv30` | float() |
| `hv_current` | `hv20` | float() |
| `iv_year_high` | `iv_year_high` | float() |
| `iv_year_low` | `iv_year_low` | float() |
| `iv_week_ago` | `iv_week_ago` | float() |
| `iv_month_ago` | `iv_month_ago` | float() |
| `hv_week_ago` | `hv_week_ago` | float() |
| `hv_month_ago` | `hv_month_ago` | float() |

## What's NOT Needed

- **No OptionsDX format conversion needed** — write directly to `DuckDBStore.write_chain()` which stores Parquet. The OptionsDX normalizer path is only for raw `.txt` files.
- **No `normalize-optionsdx` step** — DoltHub data is already clean (typed columns, no parsing errors).
- **No Greeks computation** — DoltHub provides delta/gamma/theta/vega/rho directly.
- **No volume/OI filtering** — accept this limitation; filter by bid>0 and spread width instead.

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| DoltHub API slow / timeouts | Paginate queries, retry with backoff, start with `--pilot` (25 symbols) |
| DoltHub API rate limits | Throttle requests (2s between symbols); use `dolt clone` for local access |
| No `underlying_last` in chain | Merge from yfinance stock bars (fetched separately) |
| Data starts ~2019 (not 2002) | Acceptable — 5+ years covers 4+ earnings cycles per quarter |
| Aggregate queries timeout | Query per-symbol with date range filters; avoid COUNT/MIN/MAX on full table |

## Estimated Effort

| Phase | Files | Effort |
|-------|-------|--------|
| 1 — DoltHub adapter | 2 new | ~2 hrs |
| 2 — CLI commands | 1 edit | ~1 hr |
| 4 — Vol series integration | 1 edit | ~1 hr |
| 5 — Analysis pipeline | 1 new | ~1.5 hrs |
| 6 — Dashboard tab | 1 edit | ~1 hr |
| 3 — Local Dolt (optional) | 1 new | ~1 hr |
| Tests | 2 new | ~1.5 hrs |
| **Total** | | **~9 hrs** |

## First Actionable Step

Start with Phase 1 + Phase 2 for `--vol-only` mode:
- `fetch-dolthub AAPL --vol-only --start 2020-01-01`
- This fetches only `volatility_history` (much smaller than full chain)
- Immediately usable for IV ramp/crush via `earnings_cycle.py`
- Then extend to full chain for straddle/term structure analysis