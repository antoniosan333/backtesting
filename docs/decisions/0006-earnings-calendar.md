# ADR-0006: Earnings calendar for event-driven options strategies

**Status:** Accepted

## Context

Options strategies around earnings need a calendar of release dates and timing (BMO/AMC). Equity bars and OptionsDX chains alone do not provide that. yfinance exposes a shallow history (~2–3 years); longer histories require CSV import.

## Decision

- Store calendars at `data/earnings/<SYMBOL>.parquet` with columns:
  `symbol`, `earnings_date` (YYYY-MM-DD), `timing` (`BMO` | `AMC` | `unknown`), `source` (`yfinance` | `csv`), `fetched_at` (ISO UTC).
- Dedupe on `(symbol, earnings_date)` keep last via `DuckDBStore.write_earnings` / `read_earnings`.
- Primary fetch: `lambdaclass fetch-earnings SYMBOL` using yfinance; optional `--csv PATH` for manual import.
- Session rules for strategy math:
  - **BMO** — move on open of `earnings_date`.
  - **AMC** — move after close of `earnings_date` (next bar is the first full reaction).
  - **unknown** — treat as AMC and flag in metrics.
- `StrategyContext` exposes `days_to_next_earnings`, `days_since_last_earnings`, `next_earnings_date`, `earnings_timing` (calendar-day counts). Empty calendar → all `None`; strategies hold.
- Dashboard Earnings tab is **read-only** (CLI remains the write path), consistent with [ADR-0003](0003-backtest-review-ui.md).
- Per-event stats land in `events.parquet` + `earnings_*` keys in `metrics.json` when a run trades around earnings.

## Consequences

- **Pros:** Event strategies become expressible; CSV fills yfinance gaps; same Parquet/local-first pattern as stocks.
- **Cons:** yfinance history is shallow and timing strings are noisy; ETFs often have empty calendars.
- **Watch:** Mis-labeled BMO/AMC shifts entry/exit by one session — document timing in metrics.

## Out of scope

Broker calendar APIs, OptionsDX-derived earnings, writing earnings from Streamlit, multi-symbol scanners.
