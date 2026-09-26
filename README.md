# LambdaClass

Backtesting-first project for stock and options strategies with:

- Local DuckDB + Parquet storage
- Monthly strategy scaffolding
- Run outputs with reproducible config snapshots
- Preference-driven defaults for consistent execution

## Project memory

- **[AGENTS.md](AGENTS.md)** — repo map, CLI summary, conventions, links to package-level notes. The Cursor continual-learning hook may append durable facts below the marker at the bottom of that file.
- **[docs/decisions/](docs/decisions/README.md)** — ADRs for significant choices, including storage, data sources, the dashboard, options accounting, earnings, and atomic fills.

## Quick start

```bash
python -m pip install -e ".[dev]"
lambdaclass init
lambdaclass fetch SPY --start 2020-01-01
lambdaclass new-strategy demo
lambdaclass run demo
```

Options chain source (for strategies that use `context.options_chain`): set `[defaults] options_chain_source = "optionsdx"` after running `normalize-optionsdx`, or pass `lambdaclass run STRATEGY --options-source optionsdx`. Default remains `yfinance` (`data/options/<SYMBOL>.parquet` from `fetch`).

Override strategy params without editing the file: `lambdaclass run STRATEGY --param lots=2 --param width_inner=7.5`. Keys must exist in `StrategyImpl.params`, and values are converted to the type of the default (int, float, bool, or str). Each distinct param set gets its own config hash and run directory.

Sweep a grid of params: `lambdaclass sweep STRATEGY --grid width_inner=5,10,15 --grid lots=1:3:1 --metric sharpe --jobs 4`. Each combination is written as a normal run; the summary goes to `runs/<YYYY-MM>/<strategy>/_sweeps/<sweep_id>/sweep.parquet` and shows up in the dashboard's **Sweeps** tab. Grids over `--max-combos` (default 200) are refused. The best of many combinations is an optimistic estimate, so confirm it on data the sweep did not see.

## Earnings research universe

Build a history of earnings reactions for every stock and ETF with weekly options:

```bash
lambdaclass universe fetch-weeklys                      # Cboe list → data/universe/weekly_options.parquet
lambdaclass universe fetch-yahoo-earnings               # up to 100 quarters per stock, with announcement time
lambdaclass universe fetch-earnings --start 2024-01-01  # Nasdaq calendar per day: EPS, surprise, upcoming timing
lambdaclass universe fetch-bars --start 2001-10-01      # daily bars for the equities
lambdaclass universe build-events                       # merge sources → one row per report + per-symbol stats
```

Results live in `data/earnings/events/weekly_options.parquet` (per event: timing and its source, announcement time, gap, reaction and intraday returns, move in 20-day sigmas, volume ratio, 5-day run-up, 1/5/20-day drift, EPS surprise, `beat`) and `weekly_options_summary.parquet` (per symbol). Load them with `pd.read_parquet` or `DuckDBStore.read_earnings_events("weekly_options")`. The merged earnings dates are also written to `data/earnings/<SYMBOL>.parquet`, so `lambdaclass run --symbol <SYMBOL>` strategies see them in `StrategyContext`.

BMO/AMC comes from Yahoo's announcement timestamps (then Nasdaq's upcoming-day timing). For the few reports with no known time it is inferred from the overnight gaps and a per-symbol vote; filter on `timing_source == "vendor"` when you need exact timing. The Cboe list only shows current members, so older events carry survivorship bias (each fetch is kept as a dated snapshot). Details: [ADR-0010](docs/decisions/0010-weekly-options-earnings-universe.md).

Strategy files are executable Python. Only run strategies you trust; treat
`strategies/` as application code rather than an untrusted data directory.

## Backtest assumptions

- By default (`[defaults].fill_timing = "same_close"`) decisions observe a completed bar and fill at that bar's close or option-chain mid, which can introduce look-ahead bias. Set `fill_timing = "next_open"` to fill stock at the next bar's open and options at the next bar's chain mid.
- Stock is long-only unless `[risk].allow_short_stock = true`; shorts pay `[risk].short_borrow_rate` (annual, per calendar day) and dividends. A sell with nothing held is rejected as `no_position`.
- Multi-leg option decisions are all-or-none. `reduce_only` close legs cannot create a reverse position.
- Stock and option risk limits come from `[risk]`; cash cannot go negative unless `[defaults].allow_negative_cash = true`.
- `fetch --end` is inclusive. Bars store split-adjusted, dividend-unadjusted `close` plus a `dividends` column; cash dividends are credited to held shares on the ex-date. Bars fetched before dividend capture load with zero dividends (re-run `fetch` to fill them).
- Options use European-style cash settlement at intrinsic value. Early exercise, assignment, and portfolio margin are not modeled.
- Missing marks after a valid fill fall back to Black–Scholes.

See [ADR-0005](docs/decisions/0005-options-engine-accounting.md),
[ADR-0007](docs/decisions/0007-atomic-option-fills-and-risk-limits.md), and
[ADR-0008](docs/decisions/0008-fill-timing-dividends-and-short-stock.md).

## Dashboard

After `pip install -e ".[dev]"` (or a normal install so `streamlit` is present), from the repo root:

```bash
lambdaclass dashboard
```

This runs `streamlit` against the packaged app at `127.0.0.1:8501` by default (`--no-headless` opens a browser). The UI is read-only: it lists `runs/`, loads bars from `data/stocks/`, and can inspect OptionsDX chains under `[optionsdx].output_dir` in your preferences.

## OptionsDX raw data normalization

Use this after placing OptionsDX `*.txt` files under `zRawData/optionsdx` (or pass `--input-dir`). Defaults live in `config/preferences.toml` under `[optionsdx]`.

```bash
lambdaclass normalize-optionsdx
lambdaclass normalize-optionsdx --dry-run
lambdaclass normalize-optionsdx --fail-on-errors --fail-on-gates --max-negative-iv-rate 0.02 --max-crossed-market-rate 0.001
```

Outputs:

- Normalized Parquet: `data/optionsdx/normalized/<SYMBOL>/<YYYY>/<MM>/<stem>.parquet`
- Per-file JSON reports: `data/optionsdx/reports/files/<stem>.json`
- Run summary: `data/optionsdx/reports/runs/<UTC>.json`
- Last run pointer: `state/optionsdx_normalize_state.json`

### Follow-up checklist

- [ ] Run `normalize-optionsdx --dry-run` and skim `gate_failures` in stdout
- [ ] Tune `--max-negative-iv-rate` / `--max-crossed-market-rate` if you enforce CI gates
- [ ] Point strategies or loaders at `data/optionsdx/normalized` instead of raw `*.txt`
- [ ] Re-run normalization when new OptionsDX drops arrive (same command; Parquet files overwrite per stem)
