# `reporting` — agent context

## `metrics.compute_metrics`

Input: equity curve `DataFrame` with `equity` column.

Output dict keys: `cagr`, `max_drawdown`, `sharpe`, `hit_rate`, `total_return`. Returns zeros if empty or missing column.

## `tearsheet.write_tearsheet`

Delegates equity plotting to `reporting.dashboard.charts.equity_with_drawdown(..., include_drawdown=False)` so the static HTML tearsheet matches the dashboard’s equity line; then `write_html(..., include_plotlyjs="cdn")`.

Theme default from prefs: `preferences.reporting.plot_theme` (e.g. `plotly_dark`). HTML written only when `preferences.reporting.save_html` is true (`cli` `run`).

## `reporting.dashboard` (Streamlit)

- **`app.py`**: Streamlit entry — sidebar filters, tabs **Run**, **Compare**, **Sweeps** (`loader.list_sweeps` / `load_sweep` / `sweep_pivot`, `charts.sweep_heatmap`), **Chain**, **Strategy**, **Earnings** (calendar table, price + earnings markers, `events.parquet` metrics for the active run).
- **`loader.py`**: Read-only `list_runs`, `load_run` (`RunBundle` includes `option_trades`), `load_bars`, `load_chain`, `load_earnings`, `load_events`, plus Strategy helpers. Cached in `app.py` by path + mtime.
- **`indicators.py`**: Pure SMA/EMA/RSI/Bollinger/monthly returns/rolling Sharpe for overlays.
- **`charts.py`**: Plotly factories including `strategy_pnl_chart` and `price_with_earnings`.
- **`option_strategies.py`**: Backward-compatible re-export shim for `lambdaclass.options`.

Options domain code: [`lambdaclass.options`](../options/). Earnings calendar math: [`lambdaclass.earnings.calendar`](../earnings/calendar.py). Per-event stats: [`reporting/earnings_metrics.py`](earnings_metrics.py).

CLI: `lambdaclass dashboard` (`cli.py`) runs `python -m streamlit run …/dashboard/app.py`.

Parent: [AGENTS.md](../../../AGENTS.md).
