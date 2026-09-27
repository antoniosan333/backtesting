# Weekly Options Universe

Reference universe of all symbols with CBOE-listed weekly options expirations.
Intended as the tradable universe for earnings-related options backtesting.

## Source

- **CBOE Available Weeklys** — `https://www.cboe.com/us/options/symboldir/weeklys_options/?download=csv`
- Fetched: see `fetched_at` column in each file

## Files

| File | Description |
|------|-------------|
| `weekly_options_universe.csv` | All 704 symbols (stocks + ETFs/ETNs) with asset type and sector |
| `weekly_options_universe.json` | Same data as JSON with metadata envelope |
| `weekly_options_stocks.csv` | 626 individual stocks only (ETFs/ETNs excluded) |
| `weekly_options_raw.csv` | Raw CBOE CSV as downloaded |

## Schema

| Column | Type | Values |
|--------|------|--------|
| `symbol` | str | Ticker symbol (e.g. AAPL, SPY) |
| `company_name` | str | Full company/ETF name from CBOE |
| `asset_type` | str | `Stock` or `ETF/ETN` |
| `sector` | str | Heuristic sector tag (see below) |
| `source` | str | `cboe` |
| `fetched_at` | str | ISO UTC timestamp |

## Sector tags

Sectors are inferred from company name keywords (not a formal taxonomy):

`semiconductor`, `software`, `internet`, `fintech`, `banking`, `biotech`,
`energy`, `defense`, `retail`, `auto`, `telecom`, `real_estate`, `index_fund` (ETFs), `other`

## Counts

- **704** total symbols with weekly options
- **626** individual stocks
- **78** ETFs/ETNs (SPY, QQQ, IWM, VIX, etc.)

## Usage with backtesting

To use a symbol from this universe in the earnings strategies:

```bash
# 1. Fetch stock + options data
lambdaclass fetch AAPL --start 2024-01-01

# 2. Fetch earnings calendar
lambdaclass fetch-earnings AAPL

# 3. Run an earnings strategy
lambdaclass run earnings_long_straddle --symbol AAPL --start 2024-01-01
```

## Notes

- ETFs/ETNs generally do not have earnings dates; they are included for
  non-earnings weekly options strategies.
- The CBOE list is updated periodically; re-fetch with
  `lambdaclass fetch` of this URL to refresh.
- yfinance earnings history is shallow (~2-3 years); for longer histories
  use CSV import (`lambdaclass fetch-earnings SYMBOL --csv PATH`).