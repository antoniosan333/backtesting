from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

import lambdaclass.reporting.dashboard.loader as dashboard_loader
from lambdaclass.reporting.dashboard import loader


def _write_run(
    runs_root: Path,
    *,
    month: str,
    strategy: str,
    run_id: str,
    metrics: dict[str, float] | None = None,
    bars: list[tuple[str, float]] | None = None,
    trades: list[dict[str, object]] | None = None,
    snapshot_extra: str = "",
    log_lines: list[str] | None = None,
) -> Path:
    run_dir = runs_root / month / strategy / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = metrics or {
        "total_return": 0.1,
        "sharpe": 1.2,
        "max_drawdown": -0.05,
        "hit_rate": 0.55,
        "cagr": 0.08,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if bars:
        df = pd.DataFrame(bars, columns=["date", "equity"])
        df.to_parquet(run_dir / "equity.parquet", index=False)
    else:
        pd.DataFrame({"date": ["2024-01-02"], "equity": [100000.0]}).to_parquet(
            run_dir / "equity.parquet", index=False
        )
    cols = ["date", "action", "quantity", "price", "cash_after"]
    df_trades = pd.DataFrame(trades or [], columns=cols)
    df_trades.to_csv(run_dir / "trades.csv", index=False)
    snapshot = (
        '[meta]\nconfig_hash = "abc1234567"\n'
        '[strategy_params]\nfast = "10"\nslow = "30"\n'
        '[cli_overrides]\nstart = "2024-01-02"\nend = "2024-01-05"\n' + snapshot_extra
    )
    (run_dir / "config.snapshot.toml").write_text(snapshot, encoding="utf-8")
    log = "\n".join(
        log_lines or [f"strategy={strategy}", "symbol=SPY", "rows=4", "trades=0", "config_hash=abc1234567"]
    )
    (run_dir / "run.log").write_text(log, encoding="utf-8")
    return run_dir


def test_list_runs_orders_newest_first(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    older = _write_run(runs_root, month="2024-01", strategy="alpha", run_id="20240102T000000Z-aaa-1111111111")
    time.sleep(0.05)
    newer = _write_run(runs_root, month="2024-01", strategy="alpha", run_id="20240103T000000Z-aaa-2222222222")
    runs = loader.list_runs(runs_root)
    assert runs[0] == newer
    assert runs[1] == older
    assert loader.list_strategies(runs_root) == ["alpha"]
    assert loader.runs_for_strategy(runs_root, "alpha") == [newer, older]


def test_load_run_returns_bundle_with_snapshot_fields(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = _write_run(
        runs_root,
        month="2024-01",
        strategy="alpha",
        run_id="20240103T000000Z-aaa-2222222222",
        bars=[("2024-01-02", 100000.0), ("2024-01-03", 100500.0)],
        trades=[
            {"date": "2024-01-02", "action": "buy", "quantity": 10, "price": 100.0, "cash_after": 99000.0},
            {"date": "2024-01-03", "action": "sell", "quantity": 10, "price": 110.0, "cash_after": 100100.0},
        ],
    )
    bundle = loader.load_run(run_dir)
    assert bundle.strategy == "alpha"
    assert bundle.run_id == run_dir.name
    assert bundle.symbol == "SPY"
    assert bundle.config_hash == "abc1234567"
    assert bundle.start == "2024-01-02"
    assert bundle.end == "2024-01-05"
    assert bundle.snapshot["strategy_params"]["fast"] == "10"
    assert len(bundle.equity_curve) == 2
    assert len(bundle.trades) == 2
    assert bundle.option_trades.empty


def test_load_run_uses_snapshot_symbol_and_treats_legacy_none_bounds_as_absent(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(
        tmp_path / "runs",
        month="2024-01",
        strategy="alpha",
        run_id="legacy",
        bars=[("2024-01-02", 100000.0), ("2024-01-03", 100500.0)],
        log_lines=["symbol=SPY"],
    )
    (run_dir / "config.snapshot.toml").write_text(
        '[meta]\nconfig_hash = "abc1234567"\n[cli_overrides]\nstart = "None"\nend = "None"\nsymbol = "qqq"\n',
        encoding="utf-8",
    )

    bundle = loader.load_run(run_dir)

    assert bundle.symbol == "QQQ"
    assert bundle.start == "2024-01-02"
    assert bundle.end == "2024-01-03"


def test_load_run_falls_back_to_log_symbol_for_legacy_snapshot(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path / "runs",
        month="2024-01",
        strategy="alpha",
        run_id="legacy",
        log_lines=["symbol=iwm"],
    )

    assert loader.load_run(run_dir).symbol == "IWM"


def test_aggregate_metrics_returns_one_row_per_run(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    a = _write_run(
        runs_root,
        month="2024-01",
        strategy="alpha",
        run_id="20240102T000000Z-aaa-1111111111",
        metrics={"sharpe": 1.0, "total_return": 0.05},
    )
    b = _write_run(
        runs_root,
        month="2024-01",
        strategy="beta",
        run_id="20240103T000000Z-bbb-2222222222",
        metrics={"sharpe": 0.7, "total_return": 0.02},
    )
    table = loader.aggregate_metrics(runs_root, [a, b])
    assert len(table) == 2
    assert set(table["strategy"]) == {"alpha", "beta"}
    assert "sharpe" in table.columns and "total_return" in table.columns


def test_load_bars_filters_by_date_range(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    stocks = data_dir / "stocks"
    stocks.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000.0, 1500.0, 2000.0],
        }
    ).to_parquet(stocks / "SPY.parquet", index=False)
    full = loader.load_bars(data_dir, "SPY")
    filtered = loader.load_bars(data_dir, "SPY", start="2024-01-03", end="2024-01-04")
    assert len(full) == 3
    assert filtered["date"].tolist() == ["2024-01-03", "2024-01-04"]
    assert loader.load_bars(data_dir, "MSFT").empty


@pytest.mark.parametrize("load_name", ["load_bars", "load_chain", "load_earnings"])
def test_market_data_loaders_reject_unsafe_symbols(tmp_path: Path, load_name: str) -> None:
    load = getattr(loader, load_name)
    args = (tmp_path, "../secret", []) if load_name == "load_chain" else (tmp_path, "../secret")

    with pytest.raises(ValueError, match="symbol"):
        load(*args)


def test_load_chain_normalizes_symbol_before_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[str] = []

    def fake_load(root: Path, symbol: str, dates: list[str]) -> pd.DataFrame:
        recorded.append(symbol)
        return pd.DataFrame()

    monkeypatch.setattr(dashboard_loader, "load_normalized_optionsdx_chain", fake_load)

    loader.load_chain(tmp_path, "brk.b", ["2024-01-02"])

    assert recorded == ["BRK.B"]


def test_derived_trade_stats() -> None:
    trades = pd.DataFrame(
        [
            {"date": "2024-01-02", "action": "buy", "quantity": 10, "price": 100.0, "cash_after": 0.0},
            {"date": "2024-01-05", "action": "sell", "quantity": 10, "price": 110.0, "cash_after": 1100.0},
            {"date": "2024-01-10", "action": "buy", "quantity": 5, "price": 120.0, "cash_after": 500.0},
            {"date": "2024-01-12", "action": "sell", "quantity": 5, "price": 110.0, "cash_after": 1050.0},
        ]
    )
    assert loader.num_trades(trades) == 4
    assert loader.avg_holding_days(trades) == 2.5
    assert loader.win_rate_per_trade(trades) == 0.5
    empty = pd.DataFrame(columns=["date", "action", "quantity", "price", "cash_after"])
    assert loader.num_trades(empty) == 0
    assert loader.avg_holding_days(empty) == 0.0
    assert loader.win_rate_per_trade(empty) == 0.0


def test_run_dir_mtime_ns_changes_with_writes(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = _write_run(
        runs_root, month="2024-01", strategy="alpha", run_id="20240102T000000Z-aaa-1111111111"
    )
    mt1 = loader.run_dir_mtime_ns(run_dir)
    time.sleep(0.05)
    (run_dir / "metrics.json").write_text(json.dumps({"total_return": 0.2}), encoding="utf-8")
    mt2 = loader.run_dir_mtime_ns(run_dir)
    assert mt2 > mt1


def test_chain_expiries_sorted_unique() -> None:
    ch = pd.DataFrame(
        {
            "expiry": ["2026-06-01", "2026-03-01", "2026-06-01"],
            "strike": [1.0, 2.0, 3.0],
        }
    )
    assert loader.chain_expiries(ch) == ["2026-03-01", "2026-06-01"]


def test_chain_spot_estimate_last_close_on_or_before_asof() -> None:
    bars = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-05", "2024-01-08"],
            "close": [100.0, 101.5, 102.0],
        }
    )
    assert loader.chain_spot_estimate(bars, "2024-01-06") == pytest.approx(101.5)
