from __future__ import annotations

from pathlib import Path

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
