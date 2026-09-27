# ADR-0008: Historical vs implied volatility analysis and the earnings IV cycle

**Status:** Proposed

## Context

We want to know, per symbol and per date, whether options are cheap or expensive, and how implied volatility (IV) builds into an earnings release and collapses after it. Today the repo has the inputs but not the analysis:

- OHLC bars in `data/stocks/<SYMBOL>.parquet` (historical volatility source).
- Per-contract `iv`, `underlying_last`, `dte`, `strike`, `side`, bid/ask in normalized OptionsDX Parquet (implied volatility source). yfinance chains are not usable for history.
- `options/expected_move.py` already interpolates an ATM IV per expiry; `earnings/calendar.py` gives `days_to_next_earnings` / `days_since_last_earnings` with BMO/AMC handling.
- The dashboard has a Chain tab (single-date IV scatter) and an Earnings tab (price with markers, per-run event metrics). There is no time series of IV, no realized-vol series, and no cross-event view.

Design constraints: pure functions under `src/lambdaclass/`, Parquet + DuckDB reads, no new dependencies, Streamlit for charts, tests against the OptionsDX fixture.

## Decision

Add a `lambdaclass.volatility` package that builds a daily volatility series per symbol (realized + implied + relative-value metrics), an earnings event-study on that series, a cached artifact, a CLI command, a dashboard tab, and strategy-context fields.

### Universe and history window

Earnings symbols are the 626 stocks in `data/weekly_options_stocks.csv` (CBOE Available Weeklys, fetched 2026-09-27, on `master`). The 78 ETFs/ETNs in `data/weekly_options_universe.csv` stay out of the earnings study.

History starts at **2019-01-01**, the beginning of the free DoltHub option history. The first pass is these 25 names, all present in the stock file:

`AAPL`, `MSFT`, `NVDA`, `TSLA`, `AMD`, `META`, `AMZN`, `GOOGL`, `JPM`, `XOM`, `AVGO`, `CRM`, `COST`, `BA`, `DIS`, `INTC`, `QCOM`, `PYPL`, `COIN`, `UBER`, `SNOW`, `PANW`, `LLY`, `UNH`, `BAC`.

`lambdaclass vol --universe` later walks the rest of `weekly_options_stocks.csv`. A symbol missing from DoltHub is skipped and recorded.

### Market data sources

| Need | Source |
|------|--------|
| OHLC bars | Existing `fetch` (yfinance) |
| Historical chains and IV | DoltHub `post-no-preference/options`, table `option_chain`, paged by symbol and date. No account. |
| Earnings dates and BMO/AMC | SEC EDGAR submissions JSON: 8-K filings that contain Item 2.02. BMO vs AMC comes from the acceptance timestamp. |

`data_adapters/dolthub_chain.py` writes into the existing chain schema so `implied.py` stays vendor-neutral. `fetch-earnings --source edgar` writes the same `data/earnings/<SYMBOL>.parquet` the calendar helpers already read.

The SEC `User-Agent` contact address is read only from `LAMBDACLASS__edgar__contact_email`. It is never written to git, `preferences.toml`, config snapshots, or logs.

DoltHub has no underlying price and a reduced strike and expiry grid. Spot comes from the stock bars. If the two expiries needed to bracket a target DTE are absent, that IV point is null and the row carries a quality reason.

### 1. Realized (historical) volatility — `volatility/realized.py`

All estimators return annualized vol (`× √252`), right-aligned rolling windows, NaN until the window fills.

| Estimator | Inputs | Why |
|-----------|--------|-----|
| `close_to_close(close, window)` | close | Platform-comparable baseline (what thinkorswim / most sites call "HV") |
| `parkinson(high, low, window)` | H, L | Uses intraday range; ~5× more efficient than close-to-close |
| `garman_klass(open, high, low, close, window)` | OHLC | Range + open/close; assumes no drift, no gaps |
| `yang_zhang(open, high, low, close, window)` | OHLC | Handles overnight gaps and drift; the recommended primary estimator |

Options: `exclude_dates` — drop the return of the earnings reaction day so "ambient" HV is not inflated by the jump. Also `realized_forward(close, horizon)`: close-to-close vol over the *next* `horizon` days, used to test whether IV was actually rich.

Default windows: 10, 20, 30, 60, 252 trading days.

### 2. Implied volatility series — `volatility/implied.py`

Per quote date, from the chain:

- `atm_iv_term_structure(chain, spot) -> DataFrame[expiry, dte, atm_iv, straddle, n_quotes, quality]` — reuse the ATM interpolation from `expected_move.py` (promote `_interpolated_atm_iv` to a public helper, don't duplicate it). Treat `iv <= 0` as missing.
- `constant_maturity_iv(term, target_dte) -> float | None` — interpolate **linearly in total variance** (`σ²·T`) between the two expiries bracketing `target_dte` (the CBOE VIX convention), not linearly in IV. Targets: 7, 30, 60, 90. `iv30` is the headline number.
- `front_iv(term)` — ATM IV of the nearest expiry with `dte >= 1`; this is where the earnings ramp is most visible.
- `event_implied_vol(term, earnings_date)` — isolate the earnings jump by variance subtraction: pick the first expiry after the event (`T1`, `σ1`) and a reference expiry that also contains it but is further out (`T2`, `σ2`); ambient variance per day ≈ `(σ2²·T2 − σ1²·T1)/(T2 − T1)`; event variance = `σ1²·T1 − ambient·T1`. Returns `event_vol` and `implied_event_move = spot × √event_var` (a Market-Maker-Move-style number). Return `None` with a reason when the term structure is not inverted or expiries are missing.

### 3. Daily series builder — `volatility/series.py`

`build_vol_series(bars, chain_reader, earnings, *, windows, targets) -> DataFrame` with one row per bar date:

```
date, close,
hv10, hv20, hv30, hv60, hv252, hv_yz20, hv_yz60, hv20_ex_earnings,
iv7, iv30, iv60, iv90, iv_front, front_expiry, front_dte,
iv_rank_252, iv_pctile_252, iv_hv_spread, iv_hv_ratio,
rv_fwd21, iv30_minus_rv_fwd21,
days_to_next_earnings, days_since_last_earnings, event_vol, implied_event_move
```

Definitions:
- `iv_rank_252 = (iv30 − min(iv30, 252d)) / (max − min)`; `iv_pctile_252 = share of last 252 days with iv30 < today`. Percentile is the primary "cheap/expensive" gauge (rank is distorted by a single spike).
- `iv_hv_spread = iv30 − hv20`; `iv_hv_ratio = iv30 / hv20` (variance risk premium proxy).
- `rv_fwd21 = realized_forward(close, 21)`; `iv30_minus_rv_fwd21` is the ex-post premium — how much the market overpaid or underpaid for the month that followed.

Chain reading must not go through `load_normalized_optionsdx_chain` for multi-year ranges (it concatenates every Parquet file). Add `storage/optionsdx_reader.py::read_atm_slice(root, symbol, start, end, max_strike_distance_pct=0.15)` that uses DuckDB `read_parquet(glob)` with a `WHERE quote_date BETWEEN … AND abs(strike_distance_pct) <= …` projection of only the columns above. This cuts I/O by an order of magnitude and keeps memory flat.

Cache: `data/cache/vol/<SYMBOL>.parquet` plus `state/vol_series_markers.json` (last built date, chain root mtime, parameters hash). Rebuild incrementally from the last date; `--rebuild` forces a full pass. Forward-looking columns (`rv_fwd21`) are recomputed for the trailing 21 rows on each incremental build.

### 4. Earnings IV cycle — `volatility/earnings_cycle.py`

Event study on the series:

- `align_events(series, earnings, pre_days=30, post_days=10) -> DataFrame[earnings_date, timing, rel_day, iv30, iv_front, hv20, close, …]` where `rel_day` is in trading days relative to the **reaction session**: for AMC the reaction session is the next bar, for BMO it is the earnings bar itself. `rel_day = 0` is the last close *before* the reaction (the IV peak), `rel_day = 1` the first close after it.
- `normalize(aligned, base="rel_day=-30" | "ambient")` — express IV as a ratio to its level 30 days out (or to `hv20_ex_earnings`) so events at different vol regimes can be averaged.
- `aggregate_cycle(aligned) -> DataFrame[rel_day, median, q25, q75, mean, n]` for `iv30`, `iv_front`, and normalized versions.
- `event_table(aligned) -> DataFrame` one row per event: `iv_front_pre (rel_day 0)`, `iv_front_post (rel_day 1)`, `crush_pct = 1 − post/pre`, `iv30_pre`, `iv30_post`, `implied_event_move`, `expected_move_straddle` (from ADR-0007), `realized_move = |close₁ − close₀| / close₀`, `realized_over_implied`, `ramp_pct = iv_front_pre / iv_front(rel_day −20) − 1`.
- `cycle_summary(table) -> dict`: median ramp, median crush, share of events where realized < implied, median `realized_over_implied`.

### 5. CLI — `lambdaclass vol SYMBOL [--start] [--end] [--rebuild] [--csv PATH]`

Builds or updates the cached series, prints the latest row (iv30, hv20, spread, ratio, percentile, days to earnings) and the earnings-cycle summary, optionally exports CSV. Preferences section `[volatility]`: `hv_windows`, `iv_targets_dte`, `lookback_days = 252`, `max_strike_distance_pct`, `pre_days`, `post_days`.

### 6. Dashboard — new **Volatility** tab (`reporting/dashboard/vol_charts.py`)

1. **IV vs HV**: `iv30`, `hv20`, `hv_yz20` lines; earnings dates as vertical lines; shaded band where `iv_hv_ratio > 1.2` (rich) or `< 0.8` (cheap).
2. **Cheap / expensive gauge**: `iv_pctile_252` and `iv_rank_252` with 20/50/80 guide lines; current values as metrics.
3. **Premium realized**: scatter of `iv30` vs `rv_fwd21` with the 45° line, plus a histogram of `iv30_minus_rv_fwd21` with the current spread marked — shows how often, and by how much, IV has overpriced the following month.
4. **Term structure**: ATM IV by DTE for a chosen date, with the event-implied vol annotated when an earnings date is inside the front expiry.
5. **Earnings cycle**: median normalized IV path from `rel_day −30` to `+10` with the IQR band, individual events as faint lines, and the `event_table` beneath with download.

Guidance text under the gauge, kept short: "Options are expensive when IV percentile is high *and* IV/HV > 1.2; cheap when both are low. Before earnings, compare `implied_event_move` to the median realized move in the table instead."

### 7. Strategy access

Add `iv30`, `hv20`, `iv_rank_252`, `iv_pctile_252`, `iv_hv_ratio` to `StrategyContext` (all `None` when no series). `run_backtest` looks them up from the cached series when it exists; it never builds the series itself. This lets a strategy say "sell premium only when `iv_pctile_252 > 0.6`" or "enter the earnings straddle only when `implied_event_move` is below the median realized move".

### 8. Tests

- Realized: simulated GBM with known σ → all four estimators within tolerance; Yang-Zhang unaffected by a constant drift; `exclude_dates` removes exactly one return.
- Implied: constructed term structure → `constant_maturity_iv` reproduces a known variance interpolation; `event_implied_vol` recovers a planted jump; non-inverted structure returns `None`.
- Series: fixture `spy_eod_201201` end-to-end through `read_atm_slice` → non-null `iv30` on trading days, `iv_rank` in [0, 1].
- Earnings cycle: synthetic IV path with a ramp and crush around one AMC and one BMO event → `rel_day` alignment correct for both, `crush_pct` and `ramp_pct` match the planted values.
- Cache: incremental build appends only new dates and recomputes trailing forward columns.

### Delivery order

1. `realized.py`, `implied.py` (+ promote the ATM helper) with unit tests.
2. `fetch-earnings --source edgar` and `dolthub_chain.py`, then `series.py`, cache, and `vol` for the 25-name pilot. `optionsdx_reader.read_atm_slice` remains the reader for OptionsDX files when those exist.
3. `earnings_cycle.py` and tests.
4. Volatility tab.
5. `StrategyContext` fields.
6. `vol --universe` over the remaining names in `weekly_options_stocks.csv`.

## Consequences

- Easier: one cached Parquet per symbol answers "cheap or expensive?" instantly, and the same series feeds strategies, the dashboard, and the earnings event study. Everything is derived from data already on disk.
- Data coverage starts at 2019 and only for names DoltHub actually stores. The pilot proves the earnings cycle before the other ~600 names are fetched. OptionsDX files, when present, extend a single name back before 2019 through the same chain schema. `fetch-earnings --csv` remains available.
- Vendor IV: OptionsDX `iv` is vendor-computed from a pre-close snapshot; `iv30` will differ slightly from thinkorswim's number. Quality flags and `n_quotes` are carried so thin days are visible.
- Percentile/rank need a 252-day warm-up; the first year of a series has no gauge.
- Adding `StrategyContext` fields is additive; existing strategies are unaffected.
