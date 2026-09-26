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

Strategy files are executable Python. Only run strategies you trust; treat
`strategies/` as application code rather than an untrusted data directory.

## Backtest assumptions

- Decisions observe a completed bar and fill at that bar's close or option-chain mid. This same-bar model can introduce look-ahead bias; use results for research rather than execution-quality simulation.
- Multi-leg option decisions are all-or-none. `reduce_only` close legs cannot create a reverse position.
- Stock and option risk limits come from `[risk]`; cash cannot go negative unless `[defaults].allow_negative_cash = true`.
- Options use European-style cash settlement at intrinsic value. Early exercise, assignment, dividends, borrow costs, and portfolio margin are not modeled.
- Missing marks after a valid fill fall back to Black–Scholes.

See [ADR-0005](docs/decisions/0005-options-engine-accounting.md) and
[ADR-0007](docs/decisions/0007-atomic-option-fills-and-risk-limits.md).

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
