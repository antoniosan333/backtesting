# ADR-0008: Fill timing, cash dividends, and short stock

**Status:** Accepted

## Context

ADR-0007 left three simplifications that biased results:

- Decisions filled at the same bar's close / chain mid, so a strategy could act
  on a price it had only just observed (look-ahead bias).
- Bars stored no dividend information, so stock holdings understated total
  return.
- `sell` was silently ignored without a long position, so strategies could not
  short stock and got no feedback when a sell did nothing.

## Decision

**Fill timing.** `[defaults].fill_timing` selects `same_close` (default, the
previous behavior) or `next_open`. With `next_open`, each bar's decision becomes
a pending order that fills on the following bar, after expiry settlement and
cash flows but before `on_bar`: stock at that bar's `open` (falling back to
`close` when missing), option legs at that bar's chain mid with the same
all-or-none rules. Chains are end-of-day snapshots, so the next bar's chain mid
is the earliest observable option price. A pending order on the final bar is
rejected with `no_next_bar`. Under `next_open`, `last_fills` / `last_rejections`
report what happened to the previous decision at the start of the current bar.

**Dividends.** Bars carry a `dividends` column (cash per share on the ex-date).
`close` stays split-adjusted but not dividend-adjusted so it remains comparable
with option strikes; dividend-adjusted closes would rewrite historical prices.
The engine applies `position × dividend` to cash on the ex-date, before
`on_bar`, and records a `dividend` row in `trades`. Short positions pay the
dividend. Bars fetched before this change load with zero dividends.

**Short stock.** `[risk].allow_short_stock` (default `false`) lets `sell` take
the position below zero; `buy` covers before going long. `[risk].short_borrow_rate`
is an annual rate charged per calendar day on the previous close's short
notional and recorded as `borrow_fee` rows in `trades`. `max_position_pct`
applies to absolute notional and only to orders that increase exposure. With
shorting disabled, a sell is capped at the held quantity and a sell with nothing
held is rejected as `no_position`.

## Consequences

- Default runs are unchanged except that no-op sells now appear in
  `rejected_orders` (and trip `--fail-on-rejected-orders`).
- `next_open` results differ from `same_close` for every strategy; compare runs
  only within one fill timing (the setting is part of the config hash).
- Round-trip trade statistics still pair `buy` → `sell`, so short round trips
  are not counted as trades; equity-based metrics include them.
- Early assignment, hard-to-borrow availability, and margin are still not
  modeled.
