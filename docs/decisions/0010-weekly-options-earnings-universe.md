# ADR-0010: Weekly-options universe and earnings reaction history

**Status:** Accepted

## Context

Earnings strategies need many events, not one symbol at a time. Names with
weekly option expirations are the natural universe: options are liquid enough
to trade the announcement, and a weekly expiry sits right after the report.
`fetch-earnings` handled one symbol through yfinance, whose earnings scraper
needs `lxml` and silently returned nothing without it; it also discarded the
announcement time, so every timing was `unknown`.

## Decision

**Universe.** `lambdaclass universe fetch-weeklys` downloads the Cboe
"Available Weeklys" CSV and keeps the *Exchange Traded Products* (`etp`) and
*Equity* (`equity`) sections; index schedules are dropped. Symbols use the
yfinance form (`BRK.B` → `BRK-B`), and anything that fails the symbol pattern is
discarded because symbols become file names. The list is written to
`data/universe/weekly_options.parquet` plus a dated copy under
`data/universe/snapshots/weekly_options/<date>.parquet`.

**Two earnings sources**, merged by `build-events`:

- **Yahoo** (`universe fetch-yahoo-earnings`, `lxml` is now a dependency): up to
  100 quarters per symbol (back to ~2002) with the announcement timestamp, EPS
  estimate, reported EPS, and surprise. `parse_yahoo_earnings_dates` maps
  before 09:30 New York time to BMO, 16:00 or later to AMC, and in-session
  releases to `unknown`. A timestamp of exactly midnight UTC is Yahoo's "time
  unknown" marker (it reads as 19:00/20:00 the previous New York day), so it
  keeps its UTC date with `unknown` timing. Stored per symbol under
  `data/earnings/yahoo/<SYMBOL>.parquet`; symbols fetched within
  `--max-age-days` are skipped. The single-symbol `fetch-earnings` now gets real
  timing too, and adapter errors propagate to the retry logic instead of being
  swallowed.
- **Nasdaq** (`universe fetch-earnings`): the calendar one day at a time
  (`api.nasdaq.com/api/calendar/earnings?date=`), covering every US reporter
  with actual EPS, consensus, surprise, and estimate count, and BMO/AMC for
  upcoming days. Each day, including empty ones, is cached under
  `data/earnings/calendar/<YYYY>/<date>.parquet`. Reruns skip cached days except
  those within `--refresh-days` of today and future days. A refetch keeps a
  previously known timing when Nasdaq now reports none, and an empty refetch
  never overwrites a non-empty cache.

`combine_earnings_sources` pairs each Yahoo report with the nearest Nasdaq
report of the same symbol within 3 days. Timing comes from Yahoo, else Nasdaq
(`timing_vendor` says which); the date is Yahoo's when it knows the time of
day, else Nasdaq's; EPS fields are Nasdaq's, filled from Yahoo. Unpaired rows
from either source are kept (`source` = `yahoo`, `nasdaq`, or `nasdaq+yahoo`).

**Bars.** `universe fetch-bars` loops the existing yfinance adapter over the
universe (retries, failures reported, symbols already covering the window
skipped).

**Events.** `universe build-events` writes one row per report to
`data/earnings/events/weekly_options.parquet` and per-symbol statistics to
`weekly_options_summary.parquet` (`lambdaclass.earnings.events`):

- The reaction session is the report day for BMO and the next session for AMC;
  weekend/holiday reports react on the next session.
- Vendor timing (`timing_source = vendor`) is used whenever known. Otherwise
  timing is inferred from the larger overnight gap (`inferred_gap`, with a
  `timing_confidence` share), and a per-symbol vote (`symbol_timing`) overrides
  gap guesses below 0.9 confidence (`inferred_symbol`), because companies
  rarely change their slot. Vendor timings count 0.5 in the vote and each gap
  guess `confidence - 0.5`; the vote needs a net share of at least 20%.
- Checked against Yahoo timestamps, gap inference alone was right about 81% of
  the time and gap plus vote about 92%, so the inference is a fallback for the
  few reports Yahoo has no time for.
- Per-event fields: gap, reaction, and intraday returns, absolute move, move in
  units of the prior 20-day daily volatility, volume ratio, 5-day run-up, 1/5/20
  day post-reaction drift, EPS actual/estimate/surprise, `beat`,
  `announce_time`.
- Bars are split-adjusted but not dividend-adjusted, so a dividend paid on the
  reaction day is added back.

`build-events` also publishes the merged calendar, with resolved timing, to
`data/earnings/<SYMBOL>.parquet` so `run` strategies see it through
`StrategyContext`. Earlier universe rows (sources `nasdaq`, `yahoo`,
`nasdaq+yahoo`) are replaced; rows from other sources, such as CSV imports, are
kept.

## Consequences

- **Survivorship bias.** Cboe publishes today's list only. Backtests over past
  years use today's members; dated snapshots start the historical record from
  the first fetch. Yahoo's history makes this more visible: a 2005 event set
  only contains companies that still have weekly options today.
- **Look-ahead in inferred timing.** The fallback inference uses the reaction
  itself; filter on `timing_source == "vendor"` when a strategy trades before
  the report and timing accuracy matters.
- Both vendor endpoints are unofficial and may change or throttle. Requests are
  retried, Nasdaq days are paced (`--pause`, default 0.2 s), and failed days or
  symbols stay unfetched so the next run retries them.
- Implied moves from options are not part of the events table yet; they need
  historical chains (OptionsDX) for each symbol.
