# Architecture Decision Records (ADRs)

Short, append-only records of **why** LambdaClass chose a path — not user guides (see root `README.md` and `AGENTS.md`).

## When to add an ADR

Write a new ADR when you change or lock in:

- Storage or on-disk layout (Parquet paths, DuckDB usage, normalization output)
- CLI surface or default behavior users rely on
- OptionsDX normalization rules or quality semantics
- A significant dependency or integration choice

Skip ADRs for one-line bugfixes unless they reverse a prior decision.

## Naming

`NNNN-kebab-title.md` — four-digit sequence (zero-padded), then a short kebab-case slug. Next free number after the highest existing file.

## Status

Use in the document frontmatter or first line:

- **Proposed** — under discussion
- **Accepted** — current truth for the codebase
- **Superseded** — link to the replacing ADR

## Template

Copy [0000-template.md](0000-template.md) and fill in sections.

## Index

| ADR | Title |
|-----|--------|
| [0001](0001-local-duckdb-parquet.md) | Local DuckDB + Parquet for market data |
| [0002](0002-options-chain-source.md) | Options chain source for backtests (`yfinance` vs `optionsdx`) |
| [0003](0003-backtest-review-ui.md) | Backtest review UI (Streamlit dashboard, read-only, local-first) |
| [0004](0004-options-strategy-builder.md) | Options strategy lab (chain-only legs, BSM + Greeks, Strategy tab) |
| [0005](0005-options-engine-accounting.md) | Options ledger, MTM, and expiry settlement in `run_backtest` |
| [0006](0006-earnings-calendar.md) | Earnings calendar fetch, context fields, event metrics, Earnings tab |
| [0007](0007-atomic-option-fills-and-risk-limits.md) | Atomic option structures, reduce-only closes, ledger context, and enforced risk limits |
| [0008](0008-fill-timing-dividends-and-short-stock.md) | Fill timing (`same_close` / `next_open`), cash dividends, and short stock |
| [0009](0009-parameter-sweeps.md) | `run --param` overrides, `lambdaclass.runs` orchestration, and `sweep` |
| [0010](0010-weekly-options-earnings-universe.md) | Weekly-options universe, Nasdaq earnings history, and earnings reaction events |
| [0011](0011-expected-move.md) | Expected move from historical option chains |
| [0012](0012-volatility-analysis.md) | Historical vs implied volatility analysis and the earnings IV cycle |
