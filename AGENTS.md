# LambdaClass — agent context

Stock and options backtesting: Typer CLI, TOML preferences, local Parquet + DuckDB reads, monthly strategy folders, run outputs with frozen config snapshots.

## Project map

| Area | Path |
|------|------|
| Package | `src/lambdaclass/` |
| CLI entry | `lambdaclass` → `lambdaclass.cli:main` |
| Preferences | `config/preferences.toml` (created by `init`) |
| Strategies (monthly) | `strategies/<YYYY-MM>/<name>.py` |
| Fetched data | `data/stocks/`, `data/options/`, `data/cache/`, `data/earnings/` |
| OptionsDX normalized | `data/optionsdx/normalized/<SYMBOL>/<YYYY>/<MM>/` |
| OptionsDX reports | `data/optionsdx/reports/files/`, `data/optionsdx/reports/runs/` |
| Backtest runs | `runs/<YYYY-MM>/<strategy>/<run_id>/` |
| App state JSON | `state/` (e.g. `fetch_markers.json`, `optionsdx_normalize_state.json`) |
| Raw OptionsDX | `zRawData/optionsdx/*.txt` |
| **Weekly options universe** | `data/weekly_options_universe.csv` (704 symbols), `data/weekly_options_stocks.csv` (626 stocks), `data/weekly_options_universe.json`, `data/weekly_options_universe_README.md` |
| Tests | `tests/` (fixtures `tests/fixtures/optionsdx/`) |

Deeper package notes: [data_adapters](src/lambdaclass/data_adapters/AGENTS.md), [storage](src/lambdaclass/storage/AGENTS.md), [strategies](src/lambdaclass/strategies/AGENTS.md), [backtest](src/lambdaclass/backtest/AGENTS.md), [reporting](src/lambdaclass/reporting/AGENTS.md).

Architecture decisions: [docs/decisions/](docs/decisions/README.md).

## Commands

All via `lambdaclass` (Python ≥ 3.11, `pip install -e ".[dev]"`).

| Command | Role |
|---------|------|
| `init` | Create dirs + default `config/preferences.toml` (`--force` overwrites prefs) |
| `fetch SYMBOL --start YYYY-MM-DD [--end]` | yfinance → Parquet append + dedupe; updates `state/fetch_markers.json` |
| `fetch-earnings SYMBOL [--csv PATH]` | Earnings calendar → `data/earnings/<SYMBOL>.parquet` (yfinance or CSV); `state/earnings_fetch_markers.json` |
| `new-strategy NAME` | Scaffold under current month folder |
| `run STRATEGY [--symbol] [--start/--end] [--options-source …] [--fail-on-rejected-orders]` | Backtest; `--options-source yfinance` or `optionsdx` (default: `[defaults].options_chain_source`) |
| `list-runs [--limit]` | Latest runs from `metrics.json` paths |
| `compare STRATEGY [--limit]` | Compare metrics for a strategy |
| `dashboard [--host] [--port] [--headless/--no-headless]` | Streamlit read-only review UI (`streamlit run` the packaged `reporting/dashboard/app.py`) |
| `normalize-optionsdx` | OptionsDX `*.txt` → normalized Parquet + JSON reports (prefs `[optionsdx]`) |

`normalize-optionsdx` flags: `--input-dir`, `--output-dir`, `--reports-dir`, `--dry-run`, `--fail-on-errors`, `--fail-on-gates`, `--max-negative-iv-rate`, `--max-crossed-market-rate`.

## Conventions and invariants

- **Preferences**: `[defaults]`, `[paths]`, `[reporting]`, `[risk]`, `[optionsdx]`. Env overrides: `LAMBDACLASS__section__key` (see `Preferences.load` in `config.py`).
- **Options chain for `run`**: `[defaults].options_chain_source` is `yfinance` (single Parquet per symbol under `data/options/`) or `optionsdx` (normalized tree under `[optionsdx].output_dir`, filtered to bar dates). Override per run with `--options-source`.
- **Strategy names**: Must match `^[A-Za-z][A-Za-z0-9_]*$` (CLI validation).
- **Strategy resolution**: Glob `strategies_dir/*/<name>.py`; pick last match; resolved path must stay under `strategies_dir` (no escape).
- **Strategy module**: File must define `StrategyImpl` subclass of `lambdaclass.strategies.base.Strategy`.
- **Run layout**: `runs/<YYYY-MM>/<strategy.name>/<run_id>/` where `run_id` includes UTC stamp, git short SHA, config hash.
- **Frozen snapshot**: `config.snapshot.toml` from `snapshot_preferences` — strategy params and CLI overrides redacted when keys match sensitive substrings (`token`, `secret`, `password`, `apikey`, `api_key`, `auth`, `credential`).
- **Config hash**: SHA1 of JSON-serialized full snapshot payload (preferences + raw strategy_params + cli_overrides), truncated — used in run_id, not the redacted file alone.
- **Fetch**: Retries (3) with backoff on adapter failures.
- **Storage**: `DuckDBStore` writes Parquet under `data/stocks/<SYMBOL>.parquet` and `data/options/<SYMBOL>.parquet`; append merges and dedupes by `(symbol, date)` / `(symbol, contract_symbol, asof)`.
- **Backtest engine**: Bar loop; options chain keyed by `asof` string matching bar `date`; stock + options commission/slippage from prefs; open-options ledger with MTM and expiry settlement ([ADR-0005](docs/decisions/0005-options-engine-accounting.md)).

## Where to look first

| Task | Start here |
|------|------------|
| CLI / commands | `src/lambdaclass/cli.py` |
| Types + prefs + snapshot | `src/lambdaclass/config.py` |
| OptionsDX pipeline | `src/lambdaclass/data_adapters/AGENTS.md` |
| Parquet I/O | `src/lambdaclass/storage/AGENTS.md` |
| Strategy API | `src/lambdaclass/strategies/base.py` |
| Run loop + outputs | `src/lambdaclass/backtest/engine.py` |
| BSM / Greeks / mid quotes | `src/lambdaclass/options/pricing.py` |
| Earnings calendar helpers | `src/lambdaclass/earnings/calendar.py` |
| Metrics / HTML report / dashboard | `src/lambdaclass/reporting/` (`reporting/dashboard/` for Streamlit) |
| **Weekly options universe (earnings symbols)** | `data/weekly_options_universe.csv` or `data/weekly_options_stocks.csv` |

## Weekly options universe (earnings research)

The tradable universe for earnings-related options backtesting. Sourced from the
CBOE Available Weeklys list — every US-listed symbol with weekly options expirations.

- **`data/weekly_options_universe.csv`** — 704 symbols (626 stocks + 78 ETFs/ETNs)
  with columns: `symbol`, `company_name`, `asset_type`, `sector`, `source`, `fetched_at`.
- **`data/weekly_options_stocks.csv`** — 626 stocks only (ETFs/ETNs excluded). Use this
  as the earnings universe since ETFs don't have earnings dates.
- **`data/weekly_options_universe.json`** — Same data as JSON with metadata envelope.
- **`data/weekly_options_universe_README.md`** — Schema docs, sector tags, source URL.

**Usage**: iterate symbols from `weekly_options_stocks.csv`, then for each:
```bash
lambdaclass fetch SYMBOL --start YYYY-MM-DD
lambdaclass fetch-earnings SYMBOL
lambdaclass run earnings_long_straddle --symbol SYMBOL --start YYYY-MM-DD
```

**Refresh**: re-download from `https://www.cboe.com/us/options/symboldir/weeklys_options/?download=csv`
and regenerate. These files are tracked in git (narrow `.gitignore` un-ignore); the rest
of `data/` remains local-only.

## Auto-appended by continual-learning

The Cursor continual-learning stop-hook may append high-signal, durable facts below this line. Keep hand-edited content above.

<!-- continual-learning:append-below -->

- On Windows, default `python` may resolve to the Microsoft Store stub; use a real Python 3.11+ on PATH (or the `py` launcher with an explicit version) before `pip install -e ".[dev]"` or `pytest`.
- Root `.gitignore` keeps `.cursor/hooks/state/`, `data/`, `runs/`, `zRawData/`, and `state/*.json` local-only; exceptions: `data/weekly_options_*.csv`, `data/weekly_options_*.json`, and `data/weekly_options_universe_README.md` are tracked (narrow un-ignore). Commit small OptionsDX samples under `tests/fixtures/optionsdx/` unless you add a narrow un-ignore.
- Streamlit dashboard defaults to loopback (`127.0.0.1`); to open it from another device on the same LAN use `lambdaclass dashboard --host 0.0.0.0`, browse to `http://<this-PC-LAN-IPv4>:<port>`, and allow the port through Windows Firewall if needed—`0.0.0.0` exposes the UI to the whole LAN.
- GitHub CI is pytest-only (`.github/workflows/ci.yml`: install `-e ".[dev]"` then `pytest`; no OptionsDX normalization or rate gates on runners).
- `py-vollib` on PyPI is now a deprecated alias that redirects to `vollib`; prefer `vollib` when touching `pyproject.toml` or the pricing imports.
- py-vollib/vollib analytical Greeks: `theta` is per calendar day (already ÷365) and `vega`/`rho` are per 1-percentage-point move (÷100) — do not label or rescale them as annual/per-unit values.
- Backtest engine options accounting ([ADR-0005](docs/decisions/0005-options-engine-accounting.md)): open ledger, chain/BSM marks, intrinsic expiry settlement, `option_trades.csv` in run dirs; equity includes `options_mtm`.
- Streamlit's first-run email prompt makes `lambdaclass dashboard --no-headless` exit immediately in non-interactive shells; a `%USERPROFILE%\.streamlit\credentials.toml` with an empty `email` enables non-interactive startup (default headless mode avoids the prompt).
