# ADR-0007: Expected move from historical option chains

**Status:** Accepted

## Context

Strategies and review tooling need the options-implied **expected move** per expiration (what thinkorswim shows next to each series in the option chain), computed point-in-time for every backtest bar.

Tradier is not a usable source for this: it has no expected-move field, its IV/greeks (ORATS, hourly) exist only on live chains, and `/markets/history` returns OHLCV for unexpired contracts only. Historical expected move must therefore be derived from the chains we already store.

Inputs already available per contract in normalized OptionsDX Parquet: `quote_date`, `underlying_last`, `expire_date`, `dte`, `strike`, `side`, `bid`, `ask`, `iv`, plus quality flags. The backtest chain frame (`CHAIN_COLUMNS` in `optionsdx_chain_loader.py`) currently drops `underlying_last` and `dte`, and fills missing IV with `0.0`.

thinkorswim's exact formula is proprietary. Published conventions:

| Method | Formula | Meaning |
|--------|---------|---------|
| `iv_1sd` | `S × IV_atm × √(DTE/365)` | ≈68% (1σ) range |
| `tos` | `iv_1sd × skew_factor` (default 0.85) | Community reproduction of thinkorswim chain value |
| `straddle` | `ATM call mid + ATM put mid` | ≈0.8σ; ×0.85 is the earnings rule of thumb |
| `straddle_1sd` | `straddle × 1.25` | 1σ via Brenner–Subrahmanyam (straddle ≈ 0.798·S·σ·√T) |
| `weighted` | `0.6·ATM straddle + 0.3·1st OTM strangle + 0.1·2nd OTM strangle` | tastytrade's skew-aware ≈68% |

## Decision

Add a pure, vendor-neutral module `lambdaclass/options/expected_move.py` that computes all methods from a single-date chain frame, and wire it into chain loading, the strategy context, the engine outputs, and the dashboard. No new dependencies.

### 1. Chain frame (`optionsdx_chain_loader.py`)

- Append `underlying_last` and `dte` to `CHAIN_COLUMNS` (optional for yfinance frames; engine and dashboard must tolerate absence).
- Stop coercing missing IV to `0.0` in the loader, or have the expected-move code treat `iv <= 0` as missing. Prefer the latter to avoid changing engine fill behavior (`_fill_quote` already clamps).

### 2. Core module (`options/expected_move.py`)

```python
@dataclass(frozen=True)
class ExpectedMove:
    expiry: str
    dte: float              # calendar days, from chain `dte` or asof→expiry
    spot: float
    atm_strike: float
    atm_iv: float | None    # interpolated, see below
    straddle: float | None
    iv_1sd: float | None
    tos: float | None
    straddle_1sd: float | None
    weighted: float | None
    quality: str            # "ok" | pipe-joined reasons, e.g. "WIDE_SPREAD|MISSING_PUT"

def expected_move_for_expiry(chain, expiry, spot, *, skew_factor=0.85, max_spread_pct=0.5) -> ExpectedMove
def expected_moves(chain, spot, *, expiries=None, **kw) -> pd.DataFrame     # one row per expiry
def expected_move_for_horizon(chain, spot, target_dte, **kw) -> ExpectedMove  # nearest expiry ≥ target_dte
```

Rules:

- **Spot:** `underlying_last` median for the date if present, else the caller-supplied bar close. `underlying_last` is preferred because OptionsDX quotes are snapshotted before the close.
- **Mids:** reuse `pricing.safe_option_mid`; drop quotes with `bid <= 0`, crossed markets, or `(ask − bid)/mid > max_spread_pct`.
- **ATM strike:** strike minimizing `|K − S|` that has both a valid call and put mid.
- **ATM IV:** linear interpolation in strike between the two strikes bracketing `S`, averaging call and put IV at each; fall back to the nearest single strike.
- **Strangles for `weighted`:** 1st/2nd OTM = next strike above ATM for the call and next below for the put. Return `None` (and a quality reason) if either wing is missing rather than substituting.
- **Time:** calendar days / 365, with a floor of 1 day for 0-DTE snapshots so values stay finite.
- Everything vectorized per `(asof, expiry)` group; no row-wise `apply`.

### 3. Strategy access

- Add `expected_moves: pd.DataFrame | None = None` to `StrategyContext` (defaulted, so existing strategies are unaffected).
- `run_backtest` computes it lazily per bar only when a chain exists for that date. Precompute once per date with a single `groupby("asof")` when `run_backtest` starts, to avoid recomputing inside the bar loop.

### 4. Run outputs

- Write `expected_moves.parquet` in the run dir (`asof, expiry, dte, spot, atm_iv, straddle, iv_1sd, tos, straddle_1sd, weighted, quality`) for the front expiry and the expiry nearest 30 DTE. This keeps files small.
- Add `expected_move` preferences with `skew_factor`, `max_spread_pct` and `horizons_dte = [0, 7, 30]`. They land in `config.snapshot.toml` and therefore the config hash.

### 5. Earnings metrics fix (`reporting/earnings_metrics.py`)

- Replace the premium-derived `implied_move_pct` with the pre-event `straddle` from `expected_moves.parquet` (entry date, first expiry after the event), and emit both `implied_move_pct` (straddle) and `implied_move_1sd_pct`. `beat_implied` compares against the straddle value (the conventional earnings test) and is documented as such.

### 6. Dashboard

- **Chain tab:** table of expected moves per expiry for the chosen date, and a ±band on the IV scatter.
- **Run tab:** optional overlay of `spot ± tos` (front expiry) on the price chart, plus a realized-vs-expected series (`|close[t+DTE] − close[t]| / expected`) to show calibration.

### 7. Tests (`tests/test_expected_move.py`)

- Synthetic BSM chain (generated with `pricing.black_scholes_price`, flat σ): `straddle_1sd ≈ iv_1sd` within 2%, `straddle ≈ 0.798 · iv_1sd`, `tos == 0.85 · iv_1sd`.
- The worked tastytrade example (S=121, straddle 4.40, strangles 3.46/2.66) gives `weighted ≈ 3.944` (tastytrade rounds each term and shows 3.95).
- Missing put / crossed / wide-spread cases return `None` with the right quality reason.
- Fixture regression on `tests/fixtures/optionsdx/spy_eod_201201` through the normalized loader.
- Engine test: `StrategyContext.expected_moves` is populated when a chain exists and is `None` otherwise.

### Delivery order

1. Core module and unit tests (no behavior change elsewhere).
2. Loader columns and `StrategyContext` field, with the engine precompute and `expected_moves.parquet`.
3. Earnings metrics switch.
4. Dashboard panels.

## Consequences

- Easier: expected move is available historically for any symbol with normalized OptionsDX data, is reproducible (inputs frozen in the snapshot), and is vendor-neutral. A live Tradier feed can later reuse the same function if it writes into `CHAIN_COLUMNS`.
- Accuracy: `tos` is an approximation (thinkorswim's skew handling is proprietary). Calibrate `skew_factor` against a handful of recorded thinkorswim values before relying on it for strike selection.
- Coverage: yfinance chains are only present for dates that were actually fetched, and yfinance IV is unreliable. Treat `optionsdx` as the supported source for this feature.
- Data quality: wide or stale quotes on illiquid names produce noisy straddles. The `quality` column and spread filter make that visible rather than hidden.
- Changing `earnings_metrics.implied_move_pct` alters numbers in existing `events.parquet` outputs. Old runs keep their files; only new runs change.
