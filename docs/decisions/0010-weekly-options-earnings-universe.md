# ADR-0010: Weekly-options universe and earnings reaction history

**Status:** Accepted

## Context

Earnings strategies need many events, not one symbol at a time. Names with
weekly option expirations are the natural universe: options are liquid enough
to trade the announcement, and a weekly expiry sits right after the report.
`fetch-earnings` handled one symbol through yfinance, whose earnings scraper
needs `lxml` (not a dependency) and fails quietly without it.

## Decision

**Universe.** `lambdaclass universe fetch-weeklys` downloads the Cboe
"Available Weeklys" CSV and keeps the *Exchange Traded Products* (`etp`) and
*Equity* (`equity`) sections; index schedules are dropped. Symbols use the
yfinance form (`BRK.B` → `BRK-B`), and anything that fails the symbol pattern is
discarded because symbols become file names. The list is written to
`data/universe/weekly_options.parquet` plus a dated copy under
`data/universe/snapshots/weekly_options/<date>.parquet`.

**Earnings history.** `universe fetch-earnings` queries the Nasdaq earnings
calendar one day at a time (`api.nasdaq.com/api/calendar/earnings?date=`),
which covers every US reporter per day with actual EPS, consensus, surprise,
and estimate count. Each day, including empty ones, is cached under
`data/earnings/calendar/<YYYY>/<date>.parquet`. Reruns skip cached days except
those within `--refresh-days` of today and future days. A refetch keeps a
previously known BMO/AMC timing when Nasdaq now reports none, and an empty
refetch never overwrites a non-empty cache. Rows for universe symbols are also
merged into `data/earnings/<SYMBOL>.parquet`, so `run` strategies see them via
`StrategyContext`.

**Bars.** `universe fetch-bars` loops the existing yfinance adapter over the
universe (retries, failures reported, symbols already covering the window
skipped).

**Events.** `universe build-events` writes one row per report to
`data/earnings/events/weekly_options.parquet` and per-symbol statistics to
`weekly_options_summary.parquet` (`lambdaclass.earnings.events`):

- The reaction session is the report day for BMO and the next session for AMC;
  weekend/holiday reports react on the next session.
- Nasdaq gives no time of day for past reports, so timing is inferred from the
  larger overnight gap (`timing_source = inferred_gap`, with a
  `timing_confidence` share). Vendor timing wins when known and is used to
  report how often the inference agrees. Inferred timings are copied back into
  the per-symbol earnings files with `timing_source = inferred_gap`.
- Per-event fields: gap, reaction, and intraday returns, absolute move, move in
  units of the prior 20-day daily volatility, volume ratio, 5-day run-up, 1/5/20
  day post-reaction drift, EPS actual/estimate/surprise, and `beat`. Bars are
  split-adjusted but not dividend-adjusted, so a dividend paid on the reaction
  day is added back.

## Consequences

- **Survivorship bias.** Cboe publishes today's list only. Backtests over past
  years use today's members; dated snapshots start the historical record from
  the first fetch.
- **Look-ahead in inferred timing.** The inference uses the reaction itself.
  Real announcement times are usually public in advance, so it is a fair label
  for research, but a strategy that trades *before* the report should rely on
  vendor timing (captured for upcoming days) where accuracy matters.
- Nasdaq's API is undocumented and may change or throttle; requests are paced
  (`--pause`, default 0.2 s) and retried, and failed days stay uncached so the
  next run retries them.
- Implied moves from options are not part of the events table yet; they need
  historical chains (OptionsDX) for each symbol.
