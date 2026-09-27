from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lambdaclass.data_adapters.optionsdx_normalize import NormalizeOptions, run_normalize
from lambdaclass.storage.optionsdx_reader import read_atm_slice

from lambdaclass.data_adapters.dolthub_chain import fetch_dolthub_chain
from lambdaclass.data_adapters.edgar_earnings import (
    earnings_rows_from_submission,
    fetch_earnings_frame,
    timing_from_acceptance,
)
from lambdaclass.volatility.earnings_cycle import (
    align_events,
    cycle_summary,
    event_table,
)
from lambdaclass.volatility.implied import (
    constant_maturity_iv,
    event_implied_vol,
    front_iv,
)
from lambdaclass.volatility.series import _iv_percentile, _iv_rank, merge_vol_cache
from lambdaclass.volatility.realized import (
    close_to_close,
    garman_klass,
    parkinson,
    yang_zhang,
)


def _ohlc_from_path(prices: np.ndarray) -> pd.DataFrame:
    days, _steps = prices.shape
    return pd.DataFrame(
        {
            "open": prices[:, 0],
            "high": prices.max(axis=1),
            "low": prices.min(axis=1),
            "close": prices[:, -1],
        }
    )


def test_estimators_recover_simulated_volatility() -> None:
    rng = np.random.default_rng(7)
    sigma = 0.20
    n_days = 800
    steps = 48
    dt = 1.0 / (252 * steps)
    shocks = rng.normal(0.0, sigma * math.sqrt(dt), size=n_days * steps)
    prices = 100.0 * np.exp(np.cumsum(shocks)).reshape(n_days, steps)
    bars = _ohlc_from_path(prices)
    window = 252

    close_vol = close_to_close(bars["close"], window).iloc[-1]
    range_vol = parkinson(bars["high"], bars["low"], window).iloc[-1]
    gk_vol = garman_klass(bars["open"], bars["high"], bars["low"], bars["close"], window).iloc[-1]
    yz_vol = yang_zhang(bars["open"], bars["high"], bars["low"], bars["close"], window).iloc[-1]

    assert close_vol == pytest.approx(sigma, abs=0.03)
    assert range_vol == pytest.approx(sigma, abs=0.05)
    assert gk_vol == pytest.approx(sigma, abs=0.05)
    assert yz_vol == pytest.approx(sigma, abs=0.05)


def test_yang_zhang_is_stable_when_a_constant_drift_is_added() -> None:
    rng = np.random.default_rng(11)
    n_days = 80
    steps = 8
    shocks = rng.normal(0.0, 0.01, size=n_days * steps)
    flat = (100.0 * np.exp(np.cumsum(shocks))).reshape(n_days, steps)
    base = _ohlc_from_path(flat)
    moved = base.copy()
    factor = np.exp(0.01 * np.arange(n_days))
    for column in ("open", "high", "low", "close"):
        moved[column] = moved[column] * factor

    base_vol = yang_zhang(base["open"], base["high"], base["low"], base["close"], 20).iloc[-1]
    moved_vol = yang_zhang(moved["open"], moved["high"], moved["low"], moved["close"], 20).iloc[-1]

    assert moved_vol == pytest.approx(base_vol, rel=0.05)


def test_exclude_dates_removes_one_return() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = pd.Series([100.0, 110.0, 90.0, 100.0, 100.0, 100.0])
    excluded = close_to_close(close, 5, dates=dates, exclude_dates=[dates[2]])
    returns = np.log(close / close.shift(1)).to_numpy()
    expected = float(np.std(returns[[1, 3, 4, 5]], ddof=1) * math.sqrt(252))

    assert excluded.iloc[-1] == pytest.approx(expected)
    assert close_to_close(close, 5).iloc[-1] != pytest.approx(expected)


def test_constant_maturity_iv_interpolates_total_variance() -> None:
    term = pd.DataFrame(
        {
            "expiry": ["2024-01-20", "2024-02-09"],
            "dte": [20.0, 40.0],
            "atm_iv": [0.20, 0.30],
        }
    )
    left_t = 20 / 365
    right_t = 40 / 365
    target_t = 30 / 365
    variance = (0.20**2 * left_t + 0.30**2 * right_t) / 2

    assert constant_maturity_iv(term, 30) == pytest.approx(math.sqrt(variance / target_t))
    assert constant_maturity_iv(term, 10) is None


def test_event_implied_vol_recovers_a_planted_jump() -> None:
    spot = 100.0
    ambient = 0.20**2
    jump_var = 0.10**2
    rows = []
    for dte in (10.0, 40.0):
        total = ambient * (dte / 365.0) + jump_var
        rows.append(
            {
                "expiry": f"2024-02-{int(dte):02d}",
                "dte": dte,
                "atm_iv": math.sqrt(total / (dte / 365.0)),
            }
        )
    term = pd.DataFrame(rows)
    term.loc[0, "expiry"] = "2024-01-20"
    term.loc[1, "expiry"] = "2024-02-19"

    event = event_implied_vol(term, "2024-01-15", spot, asof="2024-01-10")

    assert event.reason is None
    assert event.event_vol == pytest.approx(0.10, abs=1e-6)
    assert event.implied_event_move == pytest.approx(10.0, abs=1e-4)


def test_flat_term_structure_has_no_event_variance() -> None:
    term = pd.DataFrame(
        {
            "expiry": ["2024-01-20", "2024-02-19"],
            "dte": [10.0, 40.0],
            "atm_iv": [0.25, 0.25],
        }
    )

    event = event_implied_vol(term, "2024-01-15", 100.0, asof="2024-01-10")

    assert event.event_vol is None
    assert event.reason == "NOT_INVERTED"


def test_front_iv_uses_the_nearest_live_expiry() -> None:
    term = pd.DataFrame(
        {
            "expiry": ["2024-01-12", "2024-01-19"],
            "dte": [0.0, 7.0],
            "atm_iv": [0.40, 0.22],
        }
    )

    iv, expiry, dte = front_iv(term)

    assert iv == pytest.approx(0.22)
    assert expiry == "2024-01-19"
    assert dte == pytest.approx(7.0)


def test_optionsdx_slice_reads_the_spy_fixture(tmp_path: Path) -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "optionsdx"
    output = tmp_path / "normalized"
    run_normalize(
        NormalizeOptions(
            input_root=fixture_root,
            output_root=output,
            reports_dir=tmp_path / "reports",
            state_path=tmp_path / "state.json",
            dry_run=False,
            fail_on_errors=False,
        )
    )

    spy = read_atm_slice(output, "SPY", "2012-01-03", "2012-01-03")

    assert not spy.empty
    assert spy["asof"].eq("2012-01-03").all()
    assert pd.to_numeric(spy["implied_volatility"], errors="coerce").notna().any()


def test_iv_rank_and_percentile_use_a_trailing_window() -> None:
    iv = pd.Series([0.10, 0.20, 0.30])

    assert _iv_rank(iv, 3).iloc[-1] == pytest.approx(1.0)
    assert _iv_percentile(iv, 3).iloc[-1] == pytest.approx(2.0 / 3.0)


def test_cache_merge_refreshes_the_forward_tail_and_appends() -> None:
    previous = pd.DataFrame({"date": [f"2024-01-{day:02d}" for day in range(1, 31)], "rv_fwd21": 1.0})
    updated = pd.DataFrame({"date": [f"2024-01-{day:02d}" for day in range(1, 32)], "rv_fwd21": 2.0})

    merged = merge_vol_cache(previous, updated, forward_horizon=21)

    assert merged["date"].iloc[-1] == "2024-01-31"
    assert merged["rv_fwd21"].iloc[:10].eq(1.0).all()
    assert merged["rv_fwd21"].iloc[-21:].eq(2.0).all()


def test_earnings_cycle_aligns_bmo_and_amc_and_measures_crush() -> None:
    days = pd.bdate_range("2024-01-02", periods=80)
    series = pd.DataFrame(
        {
            "date": days.strftime("%Y-%m-%d"),
            "close": 100.0,
            "iv_front": 0.20,
            "iv30": 0.20,
            "implied_event_move": 8.0,
            "front_straddle": 7.0,
        }
    )
    amc = days[40].strftime("%Y-%m-%d")
    bmo = days[50].strftime("%Y-%m-%d")
    series.loc[20, "iv_front"] = 0.20
    series.loc[40, ["iv_front", "close"]] = [0.40, 100.0]
    series.loc[41, ["iv_front", "close"]] = [0.22, 110.0]
    series.loc[49, "iv_front"] = 0.30
    series.loc[50, "iv_front"] = 0.15
    earnings = pd.DataFrame(
        {"earnings_date": [amc, bmo], "timing": ["AMC", "BMO"], "symbol": ["ZZZ", "ZZZ"]}
    )

    aligned = align_events(series, earnings, pre_days=30, post_days=10)
    amc_rows = aligned[aligned["earnings_date"] == amc]
    bmo_rows = aligned[aligned["earnings_date"] == bmo]
    table = event_table(aligned)

    assert amc_rows.loc[amc_rows["rel_day"] == 0, "date"].iloc[0] == amc
    assert bmo_rows.loc[bmo_rows["rel_day"] == 1, "date"].iloc[0] == bmo
    amc_event = table[table["earnings_date"] == amc].iloc[0]
    assert amc_event["crush_pct"] == pytest.approx(1.0 - 0.22 / 0.40)
    assert amc_event["ramp_pct"] == pytest.approx(1.0)
    assert cycle_summary(table)["median_crush_pct"] == pytest.approx(
        pd.Series([1.0 - 0.22 / 0.40, 1.0 - 0.15 / 0.30]).median()
    )


def test_edgar_maps_acceptance_time_and_item_202() -> None:
    assert timing_from_acceptance("2026-01-29T21:30:33.000Z") == "AMC"
    assert timing_from_acceptance("2026-01-29T12:00:00.000Z") == "BMO"
    payload = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q"],
                "items": ["2.02,9.01", ""],
                "filingDate": ["2024-02-01", "2024-02-02"],
                "acceptanceDateTime": ["2024-02-01T21:30:00.000Z", ""],
            }
        }
    }

    rows = earnings_rows_from_submission(payload)

    assert rows == [{"earnings_date": "2024-02-01", "timing": "AMC"}]


def test_edgar_fetch_uses_the_injected_client() -> None:
    def http_get(url: str, _email: str) -> dict:
        if url.endswith("company_tickers.json"):
            return {"0": {"cik_str": 320193, "ticker": "AAPL"}}
        return {
            "filings": {
                "recent": {
                    "form": ["8-K"],
                    "items": ["2.02"],
                    "filingDate": ["2024-02-01"],
                    "acceptanceDateTime": ["2024-02-01T21:30:00.000Z"],
                },
                "files": [],
            }
        }

    frame = fetch_earnings_frame("AAPL", http_get=http_get, email="research@example.com")

    assert frame.iloc[0]["earnings_date"] == "2024-02-01"
    assert frame.iloc[0]["timing"] == "AMC"
    assert frame.iloc[0]["source"] == "edgar"


def test_dolthub_chain_maps_rows_into_the_backtest_schema() -> None:
    def http_get(_url: str) -> dict:
        return {
            "query_execution_status": "Success",
            "rows": [
                {
                    "date": "2024-01-02",
                    "expiration": "2024-01-19",
                    "strike": "190.00",
                    "call_put": "Call",
                    "bid": "1.00",
                    "ask": "1.20",
                    "vol": "0.2500",
                }
            ],
        }

    chain = fetch_dolthub_chain("AAPL", "2024-01-02", "2024-01-02", http_get=http_get)

    assert chain.iloc[0]["side"] == "call"
    assert chain.iloc[0]["implied_volatility"] == pytest.approx(0.25)
    assert chain.iloc[0]["asof"] == "2024-01-02"
    assert str(chain.iloc[0]["contract_symbol"]).startswith("AAPL_")
