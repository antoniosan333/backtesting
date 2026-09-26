# ADR-0004: Options strategy lab (Streamlit Strategy tab)

**Status:** Accepted

## Context

The dashboard already exposed normalized OptionsDX chains ([ADR-0003](0003-backtest-review-ui.md)) as a scatter + table. Users still lacked a way to compose multi-leg positions from **real** chain rows, stress IV, slide the mark-to-market date, and visualize **P&L at expiry vs. current** with aggregate Greeks.

## Decision

Add a fourth Streamlit tab **Strategy** alongside Run / Compare / Chain.

Properties:

- **Chain-only legs.** Strikes, expiries, sides, IV, and mids come from normalized OptionsDX Parquet (`load_chain` / `optionsdx_chain_loader`). The UI may only change **quantities** (contracts); other leg fields stay disabled in `st.data_editor`.
- **Pricing / Greeks:** Black–Scholes via existing dependency `py-vollib` (`black_scholes`, `greeks.analytical`). Per-leg P&L at evaluation: `qty × 100 × (BSM(S, K, T_eval, r, σ+shock) − mid)`; at expiry: intrinsic minus mid. `T_eval` is calendar days from the chosen evaluation date to expiry, divided by 365.
- **Presets:** Long/short call/put, vertical, ratio, straddle, strangle, call butterfly, iron condor, iron butterfly — each resolves to concrete rows via nearest-strike `snap_to_chain`.
- **Breakevens / tail flags:** Numerical zero-crossings on a dense spot grid (default ±30%, 401 points). “Unbounded” tails use edge slope + local flatness heuristics to avoid false positives on defined-risk spreads.
- **No persistence:** Download legs as CSV only; no writes to `runs/` or new DB.

## Consequences

- **Pros:** Read-only, reuses existing chain + bars loaders; pure math module is unit-testable without Streamlit; `py-vollib` already in `pyproject.toml`.
- **Cons:** European BSM only (`q=0`); no per-strike smile beyond a single **IV shock** added to each leg’s chain IV; breakevens can be missed or mis-counted if the grid is too narrow or the payoff is pathological.
- **Watch:** Dashboard indicator drift risk does not apply here, but **model risk** does — chain mids vs. model marks can disagree. Mitigate by documenting limitations in this ADR and in-tab copy.

## Out of scope

- American exercise / dividends / borrow.
- Saving named strategies or pushing orders.
- Vol-surface or local-vol repricing.
