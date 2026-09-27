# ADR-0009: Parameter overrides and sweeps

**Status:** Accepted

## Context

Trying a different strategy parameter meant editing the strategy file, and
there was no way to compare a grid of parameter values. The `run` command also
mixed data loading, execution, artifact writing, and console output in one
function, so nothing else could reuse it.

## Decision

**Overrides.** `run --param key=value` sets `StrategyImpl.params` on the
instance. Keys must already exist in the strategy's params; values are
converted to the default's type (bool, int, float, str). The resolved params
are part of the config hash, so each variant gets its own run directory.

**Orchestration.** `lambdaclass.runs.runner` owns `load_run_inputs` (bars,
chain, earnings, and the per-date chain index) and `execute_run` (backtest plus
all run artifacts, no console output). It sits outside `backtest/` because
backtest modules must not import `reporting`.

**Sweeps.** `lambdaclass sweep STRATEGY --grid key=a,b --grid key=start:stop:step`
expands the Cartesian product (duplicates removed after type conversion) and
refuses grids larger than `--max-combos` (default 200). `--param` fixes
non-swept params.

- Each combination is an ordinary run directory under
  `runs/<YYYY-MM>/<strategy>/<run_id>/` with the same config hash an
  equivalent `run --param` would produce, so `list-runs`, `compare`, and the
  dashboard see them without changes. `run.log` records `sweep_id`.
- The sweep summary lives in `runs/<YYYY-MM>/<strategy>/_sweeps/<sweep_id>/`:
  `sweep.parquet` (one row per combination: `param_*` columns, all metrics,
  `status`, `error`, `run_dir`) and `sweep.json` (grid, fixed params,
  preferences, git SHA, counts).
- Inputs load once per sweep. `--jobs N` uses spawned worker processes that
  each reload the strategy file and inputs once; dynamically loaded strategy
  classes cannot be pickled, and spawn behaves the same on Linux, macOS, and
  Windows.
- A failing combination becomes an `error` row and the sweep continues;
  `--fail-fast` stops and exits non-zero.
- Every combination gets a fresh strategy instance, because strategies may keep
  state on `self`. HTML reports are skipped unless `--html` is passed.

The dashboard adds a read-only **Sweeps** tab: a results table, a metric
heatmap over one or two params with the others held fixed, and an equity
overlay of the top combinations.

## Consequences

- Choosing the best of many combinations overstates expected performance; the
  CLI and dashboard say so. Walk-forward / out-of-sample validation is the
  natural next step and can reuse `execute_run`.
- A sweep writes one full run directory per combination; `--max-combos` bounds
  disk use.
- Strategies must not mutate the shared chain frames in `context.options_chain`,
  since every combination in a process reuses them.
