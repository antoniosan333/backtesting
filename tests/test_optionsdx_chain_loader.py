from __future__ import annotations

from pathlib import Path

import pandas as pd

from lambdaclass.data_adapters.optionsdx_chain_loader import CHAIN_COLUMNS, load_normalized_optionsdx_chain
from lambdaclass.data_adapters.optionsdx_normalize import NormalizeOptions, run_normalize


def test_load_chain_from_normalized_matches_engine_columns(tmp_path: Path) -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "optionsdx"
    out = tmp_path / "normalized"
    rep = tmp_path / "reports"
    state = tmp_path / "state" / "optionsdx_normalize_state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    run_normalize(
        NormalizeOptions(
            input_root=fixture_root,
            output_root=out,
            reports_dir=rep,
            state_path=state,
            dry_run=False,
            fail_on_errors=False,
        )
    )

    spy = load_normalized_optionsdx_chain(out, "SPY", ["2012-01-03"])
    assert not spy.empty
    assert list(spy.columns) == CHAIN_COLUMNS
    assert spy["asof"].eq("2012-01-03").all()
    assert spy["symbol"].eq("SPY").all()
    assert spy["contract_symbol"].astype(str).str.len().gt(0).all()

    btc = load_normalized_optionsdx_chain(out, "BTC", ["2021-06-01"])
    assert not btc.empty
    assert btc["symbol"].eq("BTC").all()
    assert btc["contract_symbol"].iloc[0] == "BTC-1JUN21-34000-C"

    empty = load_normalized_optionsdx_chain(out, "SPY", ["2099-01-01"])
    assert empty.empty
    assert list(empty.columns) == CHAIN_COLUMNS


def test_load_chain_empty_when_symbol_dir_missing(tmp_path: Path) -> None:
    root = tmp_path / "empty_norm"
    root.mkdir()
    df = load_normalized_optionsdx_chain(root, "ZZZ", ["2020-01-01"])
    assert df.empty and list(df.columns) == CHAIN_COLUMNS


def _normalized_row(quote_date: str, contract_symbol: object = None) -> dict[str, object]:
    return {
        "symbol": "SPY",
        "quote_date": quote_date,
        "expire_date": "2024-02-16",
        "side": "call",
        "strike": 500.0,
        "last": 4.25,
        "bid": 4.0,
        "ask": 4.5,
        "iv": 0.2,
        "volume": 12,
        "contract_symbol": contract_symbol,
    }


def test_load_chain_prunes_unrequested_year_month_partitions(tmp_path: Path) -> None:
    root = tmp_path / "normalized"
    requested = root / "SPY" / "2024" / "01"
    requested.mkdir(parents=True)
    pd.DataFrame([_normalized_row("2024-01-05")]).to_parquet(requested / "january.parquet")

    unrelated = root / "SPY" / "2024" / "02"
    unrelated.mkdir(parents=True)
    # A broad recursive read would try to parse this deliberately invalid file.
    (unrelated / "must-not-be-read.parquet").write_bytes(b"not parquet")

    out = load_normalized_optionsdx_chain(root, "spy", ["2024-01-05"])

    assert len(out) == 1
    assert out.iloc[0].to_dict() == {
        "contract_symbol": "SPY_20240216_C_500000",
        "side": "call",
        "strike": 500.0,
        "last_price": 4.25,
        "bid": 4.0,
        "ask": 4.5,
        "implied_volatility": 0.2,
        "open_interest": 0.0,
        "volume": 12,
        "expiry": "2024-02-16",
        "asof": "2024-01-05",
        "symbol": "SPY",
    }


def test_load_chain_keeps_legacy_unpartitioned_files_as_fallback(tmp_path: Path) -> None:
    root = tmp_path / "normalized"
    symbol_dir = root / "SPY"
    symbol_dir.mkdir(parents=True)
    pd.DataFrame([_normalized_row("2023-12-29", "LEGACY-C")]).to_parquet(symbol_dir / "legacy.parquet")

    out = load_normalized_optionsdx_chain(root, "SPY", ["2023-12-29"])

    assert out["contract_symbol"].tolist() == ["LEGACY-C"]
    assert list(out.columns) == CHAIN_COLUMNS
