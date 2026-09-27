# ADR-0007: Atomic option fills and enforced risk limits

**Status:** Accepted

## Context

The engine previously filled each option leg independently. If one leg of a
spread was unavailable, a strategy could believe the whole structure was open
and later submit closing quantities that created an unintended naked reverse
position. The configured stock and option risk limits were also not enforced.

## Decision

Option legs emitted in one `StrategyDecision` are one all-or-none structure.
The engine prices and validates every leg before applying any cash or ledger
change. If one leg fails, every leg is recorded in `rejected_orders` with a
`structure_rejected:*` reason and none are filled.

`OptionLeg.reduce_only` explicitly marks closing orders. Such a leg is rejected
when no opposite open quantity exists, so it cannot open or reverse a position.
`StrategyContext.open_options` is an immutable view of the engine ledger;
`last_fills` and `last_rejections` report the preceding bar's results.

Stock buys enforce `[risk].max_position_pct` and available cash unless
`[defaults].allow_negative_cash` is enabled. New option contracts enforce
`[risk].max_open_positions`. Rejections are written to the normal rejected
orders artifact.

## Consequences

- Multi-leg strategies cannot be left partially filled.
- Strategies can derive closing decisions from authoritative engine state
  instead of maintaining a shadow ledger.
- Existing strategies that intend to close positions should set
  `reduce_only=True`.
- Runs produced before this decision may not be comparable because partial
  structures and unconstrained positions could previously affect P&L.
- Fills still use the completed bar's close/chain mid. This is a deliberate
  research simplification and may introduce look-ahead bias; next-bar execution
  is available via `fill_timing = "next_open"` ([ADR-0008](0008-fill-timing-dividends-and-short-stock.md)).
