# ADR-0003: Backtest review UI (Streamlit dashboard, read-only, local-first)

**Status:** Accepted

## Context

The CLI emits per-run artifacts under `runs/<YYYY-MM>/<strategy>/<run_id>/` (`metrics.json`, `trades.csv`, `equity.parquet`, `config.snapshot.toml`, `run.log`, optional `report.html`). Reviewing a run today means opening `report.html` (a single equity line) and inspecting the JSON/CSV by hand. Comparing strategies relies on `lambdaclass list-runs` / `lambdaclass compare` text output. There is no visual review of price + signals, no cross-run overlay, and no inspection of OptionsDX chains for a given bar.

## Decision

Add a local web dashboard, launched via `lambdaclass dashboard`, built with **Streamlit** and **Plotly** (already a dep).

Properties:

- **Read-only.** Reads existing run artifacts and `data/stocks/`, `data/optionsdx/normalized/`. No writes, no engine calls, no new persistence layer.
- **Same Python package.** Lives under `src/lambdaclass/reporting/dashboard/` (`app.py`, `loader.py`, `charts.py`, `indicators.py`). Streamlit is a **core dependency** in `pyproject.toml`.
- **Indicators recomputed in the dashboard** from `data/stocks/<SYM>.parquet` plus `strategy_params` in `config.snapshot.toml` — the engine emits no new diagnostics for v1.
- **Cross-run aggregation via DuckDB** over `runs/**/metrics.json` (consistent with [ADR-0001](0001-local-duckdb-parquet.md)).
- **Three tabs:** Run (single-run review), Compare (multi-run overlay + side-by-side metrics), Chain (OptionsDX inspector for a chosen bar date, delegates to existing `data_adapters/optionsdx_chain_loader.py`).
- **No auth, no multi-user.** Binds to `127.0.0.1` by default.
- **DRY:** the existing per-run `report.html` writer (`reporting/tearsheet.py`) is refactored to share the equity figure factory with the dashboard.

## Consequences

- **Pros:** A real visual review surface ships in one PR. No new stack (Plotly already used). Engine and storage stay untouched. Loader is framework-agnostic, so a later migration to Dash/FastAPI is a UI rewrite, not a data rewrite.
- **Cons:** Streamlit pulls a sizable transitive dep tree on `pip install` (~50–150 MB) and adds ~30–60s cold-cache CI install time. Streamlit's top-down rerun model requires disciplined `@st.cache_data` use to stay snappy on large `runs/` trees.
- **Watch:** if dashboard indicator formulas drift from strategy logic, plots will mislead. Mitigated by pure functions in `indicators.py` with docstring formulas and unit tests. If complex strategies start needing engine-emitted diagnostics, that becomes a separate ADR.
- **Out of scope:** auth, hosted deployment, editing strategies / launching runs from the UI, page-level Streamlit testing via `AppTest`, migration to Dash.
