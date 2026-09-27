"""Streamlit entry point for `lambdaclass dashboard`.

Launch via the CLI; do not run this file directly outside of `streamlit run`.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import lambdaclass.options as option_strategies
from lambdaclass.config import DEFAULT_PREFERENCES, Preferences
from lambdaclass.reporting.dashboard import charts, indicators, loader
from lambdaclass.runs.layout import PARAM_PREFIX, SWEEP_RESULTS_FILE
from lambdaclass.symbols import validate_symbol

st.set_page_config(page_title="LambdaClass Dashboard", layout="wide")


def _repo_root() -> Path:
    return Path.cwd()


def _load_preferences() -> Preferences:
    path = _repo_root() / "config" / "preferences.toml"
    if not path.exists():
        return DEFAULT_PREFERENCES
    return Preferences.load(path)


@st.cache_data(show_spinner=False)
def _list_strategies(runs_root_str: str) -> list[str]:
    return loader.list_strategies(Path(runs_root_str))


@st.cache_data(show_spinner=False)
def _runs_for_strategy(runs_root_str: str, strategy: str) -> list[str]:
    return [str(p) for p in loader.runs_for_strategy(Path(runs_root_str), strategy)]


@st.cache_data(show_spinner=False)
def _load_run_cached(run_dir_str: str, mtime_ns: int) -> dict[str, Any]:
    bundle = loader.load_run(Path(run_dir_str))
    return {
        "run_dir": str(bundle.run_dir),
        "strategy": bundle.strategy,
        "run_id": bundle.run_id,
        "metrics": bundle.metrics,
        "equity_curve": bundle.equity_curve,
        "trades": bundle.trades,
        "option_trades": bundle.option_trades,
        "expected_moves": bundle.expected_moves,
        "snapshot": bundle.snapshot,
        "symbol": bundle.symbol,
        "start": bundle.start,
        "end": bundle.end,
        "config_hash": bundle.config_hash,
    }


@st.cache_data(show_spinner=False)
def _load_bars_cached(data_dir_str: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    return loader.load_bars(Path(data_dir_str), symbol, start or None, end or None)


@st.cache_data(show_spinner=False)
def _aggregate_metrics_cached(runs_root_str: str, run_dir_strs: tuple[str, ...]) -> pd.DataFrame:
    return loader.aggregate_metrics(Path(runs_root_str), [Path(p) for p in run_dir_strs])


@st.cache_data(show_spinner=False)
def _load_chain_cached(normalized_root_str: str, symbol: str, dates: tuple[str, ...]) -> pd.DataFrame:
    return loader.load_chain(Path(normalized_root_str), symbol, list(dates))


@st.cache_data(show_spinner=False)
def _load_earnings_cached(data_dir_str: str, symbol: str) -> pd.DataFrame:
    return loader.load_earnings(Path(data_dir_str), symbol)


@st.cache_data(show_spinner=False)
def _load_events_cached(run_dir_str: str, mtime_ns: int) -> pd.DataFrame:
    return loader.load_events(Path(run_dir_str))


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def _format_run_label(run_dir_str: str) -> str:
    return Path(run_dir_str).name


def _render_run_tab(
    bundle_dict: dict[str, Any], bars: pd.DataFrame, indicator_overlays: dict[str, pd.Series]
) -> None:
    st.subheader(f"{bundle_dict['strategy']} - {bundle_dict['run_id']}")
    cols = st.columns(3)
    cols[0].metric("Symbol", bundle_dict["symbol"] or "-")
    cols[1].metric("Date range", f"{bundle_dict['start']} -> {bundle_dict['end']}")
    cols[2].metric("Config hash", bundle_dict["config_hash"] or "-")

    with st.expander("config.snapshot.toml"):
        snapshot = bundle_dict["snapshot"]
        if snapshot:
            st.json(snapshot)
        else:
            st.info("No snapshot file found.")

    trades_df = bundle_dict["trades"]
    equity_df = bundle_dict["equity_curve"]
    derived = indicators.derived_run_stats(
        bundle_dict["metrics"],
        trades_df,
        loader.num_trades(trades_df),
        loader.avg_holding_days(trades_df),
        loader.win_rate_per_trade(trades_df),
    )
    st.markdown("### Metrics")
    metrics_cols = st.columns(4)
    metric_keys = [
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "annualized_volatility",
        "max_drawdown",
        "calmar",
        "up_bar_ratio",
        "num_trades",
        "avg_holding_days",
        "win_rate_per_trade",
    ]
    for i, key in enumerate(metric_keys):
        if key in derived:
            value = derived[key]
            label = key.replace("_", " ").title()
            metrics_cols[i % 4].metric(label, f"{value:.4f}" if isinstance(value, float) else str(value))

    st.markdown("### Price + Signals")
    st.plotly_chart(
        charts.price_with_signals(
            bars,
            trades_df,
            indicator_overlays,
            bundle_dict.get("expected_moves"),
        ),
        use_container_width=True,
    )

    st.markdown("### Equity + Drawdown")
    st.plotly_chart(
        charts.equity_with_drawdown(equity_df, include_drawdown=True),
        use_container_width=True,
    )

    st.markdown("### Rolling Sharpe")
    window = st.slider("Window (bars)", min_value=5, max_value=120, value=21, step=1)
    if not equity_df.empty and "equity" in equity_df.columns:
        rs = indicators.rolling_sharpe(equity_df["equity"], window=window)
        rs.index = equity_df["date"] if "date" in equity_df.columns else rs.index
        st.plotly_chart(charts.rolling_sharpe_chart(rs), use_container_width=True)
    else:
        st.info("No equity data to compute rolling Sharpe.")

    st.markdown("### Trades")
    if trades_df.empty:
        st.info("No trades recorded.")
    else:
        st.dataframe(trades_df, use_container_width=True)
        st.download_button(
            "Download trades.csv",
            data=trades_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{bundle_dict['run_id']}-trades.csv",
            mime="text/csv",
        )

    option_trades_df = bundle_dict.get("option_trades")
    if option_trades_df is None:
        option_trades_df = pd.DataFrame()
    st.markdown("### Option trades")
    if option_trades_df.empty:
        st.caption("No option fills for this run.")
    else:
        st.dataframe(option_trades_df, use_container_width=True)
        st.download_button(
            "Download option_trades.csv",
            data=option_trades_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{bundle_dict['run_id']}-option_trades.csv",
            mime="text/csv",
            key="dl_option_trades",
        )


def _render_compare_tab(bundle_dicts: list[dict[str, Any]], runs_root: Path) -> None:
    if len(bundle_dicts) < 2:
        st.info("Select at least 2 runs in the sidebar to enable comparison.")
        return
    runs_for_overlay = {f"{b['strategy']}/{b['run_id']}": b["equity_curve"] for b in bundle_dicts}
    normalize = st.checkbox("Normalize equity to 1.0 at start", value=True)
    st.plotly_chart(
        charts.equity_overlay(runs_for_overlay, normalize=normalize),
        use_container_width=True,
    )

    st.markdown("### Side-by-side metrics")
    table = _aggregate_metrics_cached(str(runs_root), tuple(b["run_dir"] for b in bundle_dicts))
    if table.empty:
        st.info("No metrics available for selected runs.")
    else:
        st.dataframe(table, use_container_width=True)

    st.markdown("### Monthly returns (per run)")
    grid_cols = st.columns(min(2, len(bundle_dicts)))
    for i, b in enumerate(bundle_dicts):
        with grid_cols[i % len(grid_cols)]:
            st.caption(f"{b['strategy']}/{b['run_id']}")
            heat = indicators.monthly_returns_table(b["equity_curve"])
            st.plotly_chart(
                charts.monthly_heatmap(heat, title=""),
                use_container_width=True,
            )


def _render_chain_tab(
    active_bundle: dict[str, Any], bars: pd.DataFrame, prefs: Preferences, root: Path
) -> None:
    if active_bundle["symbol"] == "" or bars.empty:
        st.info("Need a run with a known symbol and fetched bars to inspect the chain.")
        return
    available_dates = bars["date"].astype(str).tolist()
    if not available_dates:
        st.info("No bar dates available for this run.")
        return
    chosen_date = st.selectbox("Bar date", options=available_dates, index=len(available_dates) - 1)
    normalized_root = _resolve_path(root, prefs.optionsdx.output_dir)
    chain = _load_chain_cached(str(normalized_root), active_bundle["symbol"], (chosen_date,))
    if chain.empty:
        st.warning(
            f"No normalized OptionsDX rows for {active_bundle['symbol']} on {chosen_date} "
            f"under {normalized_root}."
        )
        return

    spot = loader.chain_spot_estimate(bars, chosen_date)
    moves = expected_moves(
        chain,
        spot,
        skew_factor=prefs.expected_move.skew_factor,
        max_spread_pct=prefs.expected_move.max_spread_pct,
    )
    st.markdown("### Expected move by expiration")
    if moves.empty:
        st.caption("No expiration could be calculated.")
    else:
        st.dataframe(
            moves[
                [
                    "expiry",
                    "dte",
                    "spot",
                    "atm_strike",
                    "atm_iv",
                    "straddle",
                    "iv_1sd",
                    "tos",
                    "weighted",
                    "quality",
                ]
            ],
            use_container_width=True,
        )

    sides = sorted(chain["side"].dropna().unique().tolist())
    side_filter = st.multiselect("Side", options=sides, default=sides)
    expiries = sorted(chain["expiry"].dropna().unique().tolist())
    expiry_filter = st.multiselect("Expiry", options=expiries, default=expiries)

    strike_min = float(chain["strike"].min())
    strike_max = float(chain["strike"].max())
    if strike_max > strike_min:
        strike_range = st.slider(
            "Strike range",
            min_value=strike_min,
            max_value=strike_max,
            value=(strike_min, strike_max),
        )
    else:
        strike_range = (strike_min, strike_max)

    min_oi = st.number_input("Min open interest", min_value=0, value=0, step=1)

    filtered = chain[
        chain["side"].isin(side_filter)
        & chain["expiry"].isin(expiry_filter)
        & (chain["strike"].between(strike_range[0], strike_range[1]))
        & (chain["open_interest"] >= float(min_oi))
    ]
    st.caption(f"{len(filtered)} contracts after filters (of {len(chain)} loaded)")
    st.plotly_chart(charts.chain_iv_scatter(filtered), use_container_width=True)
    st.dataframe(filtered, use_container_width=True)


def _fmt_pnl_money(x: float) -> str:
    if math.isinf(x) and x > 0:
        return "∞"
    if math.isinf(x) and x < 0:
        return "-∞"
    return f"{x:,.2f}"


def _render_strategy_tab(symbol: str, bars: pd.DataFrame, prefs: Preferences, root: Path) -> None:
    """Options strategy P&L / Greeks from normalized OptionsDX chain (chain-only legs)."""
    st.subheader("Strategy lab (chain-only)")
    if symbol == "" or bars.empty:
        st.info("Set a symbol and date range with fetched bars (see sidebar).")
        return
    available_dates = bars["date"].astype(str).tolist()
    if not available_dates:
        st.info("No bar dates available.")
        return
    chosen_date = st.selectbox(
        "As-of (bar date)", options=available_dates, index=len(available_dates) - 1, key="strat_asof"
    )
    normalized_root = _resolve_path(root, prefs.optionsdx.output_dir)
    chain = _load_chain_cached(str(normalized_root), symbol, (chosen_date,))
    if chain.empty:
        st.warning(
            f"No normalized OptionsDX rows for {symbol} on {chosen_date} under {normalized_root}. "
            "Run `lambdaclass normalize-optionsdx` and ensure data exists for this symbol/date."
        )
        return

    expiries = loader.chain_expiries(chain)
    if not expiries:
        st.info("No expiries in chain for this date.")
        return
    expiry = st.selectbox("Expiry", options=expiries, index=len(expiries) // 2, key="strat_expiry")
    spot = loader.chain_spot_estimate(bars, chosen_date)
    if spot <= 0.0:
        st.warning("Could not estimate spot (close); check bars.")
        return

    preset_options = ["custom", *list(option_strategies.PRESETS.keys())]

    def _preset_label(k: str) -> str:
        if k == "custom":
            return "Custom (add legs from chain)"
        return option_strategies.PRESETS[k].label

    preset = st.selectbox("Preset", options=preset_options, format_func=_preset_label, key="strat_preset")

    params: dict[str, Any] = {}
    if preset != "custom":
        spec = option_strategies.PRESETS[preset]
        if spec.params:
            cols = st.columns(max(len(spec.params), 1))
            for i, p in enumerate(spec.params):
                col = cols[i % len(cols)]
                key = f"strat_p_{preset}_{p.name}"
                if p.kind == "choice":
                    choices = list(p.choices or ())
                    default_idx = choices.index(p.default) if p.default in choices else 0
                    params[p.name] = col.selectbox(p.label, choices, index=default_idx, key=key)
                elif p.kind == "int":
                    params[p.name] = col.number_input(
                        p.label,
                        min_value=int(p.min_value if p.min_value is not None else 1),
                        value=int(p.default),
                        step=int(p.step if p.step is not None else 1),
                        key=key,
                    )
                else:
                    params[p.name] = col.number_input(
                        p.label,
                        min_value=float(p.min_value if p.min_value is not None else 0.0),
                        value=float(p.default),
                        step=float(p.step if p.step is not None else 0.5),
                        key=key,
                    )

    if preset == "custom":
        sig: tuple[Any, ...] = ("custom", chosen_date)
    else:
        sig = (preset, chosen_date, expiry, float(spot), json.dumps(params, sort_keys=True))

    sig_key = "strategy_preset_sig"
    df_key = "strategy_legs_editor_df"
    if sig_key not in st.session_state:
        st.session_state[sig_key] = None
    if st.session_state[sig_key] != sig:
        st.session_state[sig_key] = sig
        try:
            if preset == "custom":
                legs0: list[option_strategies.Leg] = []
            else:
                legs0 = option_strategies.build_preset_legs(preset, chain, expiry, spot, params)
        except (ValueError, KeyError) as exc:
            st.warning(f"Could not build preset legs: {exc}")
            legs0 = []
        st.session_state[df_key] = option_strategies.legs_to_dataframe(legs0)

    if preset == "custom":
        st.markdown("#### Add leg (snapped to chain)")
        a1, a2, a3, a4, a5 = st.columns(5)
        add_side = a1.selectbox("Side", ["call", "put"], key="strat_add_side")
        add_expiry = a2.selectbox("Leg expiry", options=expiries, key="strat_add_exp")
        add_strike = a3.number_input(
            "Target strike", min_value=0.01, value=float(spot), step=0.5, key="strat_add_k"
        )
        add_qty = a4.number_input("Qty", value=1, step=1, key="strat_add_q")
        try:
            preview = option_strategies.snap_to_chain(chain, add_side, add_expiry, float(add_strike))
            a5.caption(f"→ {preview.contract_symbol} @ {preview.strike:.2f}")
        except ValueError as exc:
            preview = None
            a5.caption(str(exc))
        if st.button("Add leg", key="strat_add_btn") and preview is not None:
            cur = st.session_state.get(df_key)
            if cur is None or (isinstance(cur, pd.DataFrame) and cur.empty):
                cur = pd.DataFrame()
            else:
                cur = cur.drop(columns=["remove"], errors="ignore")
            new_leg = option_strategies.Leg(
                contract_symbol=preview.contract_symbol,
                side=preview.side,
                strike=preview.strike,
                expiry=preview.expiry,
                iv=preview.iv,
                mid_price=preview.mid_price,
                quantity=int(add_qty),
            )
            added = option_strategies.legs_to_dataframe([new_leg])
            st.session_state[df_key] = pd.concat([cur, added], ignore_index=True) if not cur.empty else added
            st.rerun()

    df0 = st.session_state.get(df_key)
    if df0 is None:
        df0 = option_strategies.legs_to_dataframe([])
    if isinstance(df0, pd.DataFrame) and not df0.empty and "remove" not in df0.columns:
        df0 = df0.copy()
        df0["remove"] = False
        st.session_state[df_key] = df0

    if df0 is None or df0.empty:
        st.info("No legs yet. Choose a preset or add custom legs from the chain.")
        return

    editor_key = f"strat_leg_editor_{hash(sig)}"
    st.markdown("#### Legs (quantity editable; other fields from chain)")
    edited = st.data_editor(
        df0,
        key=editor_key,
        hide_index=True,
        column_config={
            "contract_symbol": st.column_config.TextColumn("Contract", disabled=True),
            "side": st.column_config.TextColumn(disabled=True),
            "strike": st.column_config.NumberColumn("Strike", disabled=True, format="%.2f"),
            "expiry": st.column_config.TextColumn(disabled=True),
            "iv": st.column_config.NumberColumn("IV", disabled=True, format="%.4f"),
            "mid_price": st.column_config.NumberColumn("Mid", disabled=True, format="%.4f"),
            "quantity": st.column_config.NumberColumn("Qty (contracts)", step=1),
            "remove": st.column_config.CheckboxColumn("Remove"),
        },
    )
    if st.button("Apply removals", key="strat_rm_btn"):
        legs_kept = option_strategies.legs_from_dataframe(edited)
        st.session_state[df_key] = option_strategies.legs_to_dataframe(legs_kept)
        st.rerun()

    st.session_state[df_key] = edited.copy()

    legs = [lg for lg in option_strategies.legs_from_dataframe(edited) if lg.quantity != 0]
    if not legs:
        st.warning("Set at least one leg with non-zero quantity.")
        return

    distinct_expiries = {lg.expiry for lg in legs}
    multi_expiry = len(distinct_expiries) > 1

    exp_d = option_strategies.parse_option_date(str(expiry)[:10])
    asof_d = option_strategies.parse_option_date(str(chosen_date)[:10])
    if multi_expiry:
        latest_exp = max(option_strategies.parse_option_date(e) for e in distinct_expiries)
        eval_max = latest_exp - timedelta(days=1) if latest_exp > asof_d else asof_d
    else:
        eval_max = exp_d - timedelta(days=1) if exp_d > asof_d else asof_d
    if eval_max < asof_d:
        eval_max = asof_d
    eval_default = min(asof_d, eval_max)
    eval_dt = st.date_input(
        "Evaluation date (mark-to-market)",
        value=eval_default,
        min_value=asof_d,
        max_value=eval_max,
        key="strat_eval",
    )
    eval_s = eval_dt.isoformat() if isinstance(eval_dt, date) else str(eval_dt)[:10]

    c_r, c_iv = st.columns(2)
    risk_free = c_r.number_input(
        "Risk-free rate r",
        min_value=0.0,
        max_value=0.25,
        value=float(prefs.defaults.risk_free_rate),
        step=0.005,
        format="%.4f",
    )
    iv_shift = c_iv.number_input(
        "IV shock (add to each leg σ)", min_value=-0.5, max_value=0.5, value=0.0, step=0.01
    )

    try:
        out = option_strategies.position_pnl(
            legs,
            eval_date=eval_s,
            r=float(risk_free),
            iv_shift=float(iv_shift),
            spot=float(spot),
        )
    except Exception as exc:  # pragma: no cover - defensive UI
        st.error(f"P&L computation failed: {exc}")
        return

    g = out["net_greeks"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Δ", f"{g['delta']:.4f}")
    m2.metric("Net Γ", f"{g['gamma']:.6f}")
    m3.metric("Net Θ ($/day)", f"{g['theta']:.4f}")
    m4.metric("Net ν ($ / 1 vol pt)", f"{g['vega']:.4f}")

    n1, n2, n3 = st.columns(3)
    n1.metric("Net premium ($)", _fmt_pnl_money(float(out["net_premium"])))
    n2.metric("Max profit (expiry, grid)", _fmt_pnl_money(float(out["max_profit"])))
    n3.metric("Max loss (expiry, grid)", _fmt_pnl_money(float(out["max_loss"])))

    be = out["breakevens"]
    flags = []
    if out.get("unbounded_up"):
        flags.append("tail risk / uncapped P&L toward higher spot (off grid)")
    if out.get("unbounded_down"):
        flags.append("tail risk / uncapped P&L toward lower spot (off grid)")
    st.caption(
        "Breakevens (expiry): "
        + (", ".join(f"{x:.2f}" for x in be) if be else "none in range")
        + (" | " + "; ".join(flags) if flags else "")
    )

    theme = prefs.reporting.plot_theme
    title_label = option_strategies.PRESETS[preset].label if preset in option_strategies.PRESETS else "Custom"
    if multi_expiry:
        st.info(
            "Legs span multiple expiries — the 'at expiration' curve (all legs intrinsic at once) "
            "is hidden; calendar-spread expiry P&L is out of scope for v1."
        )
        fig = charts.strategy_pnl_chart(
            out["S_grid"],
            out["pnl_now"],
            out["pnl_now"],
            spot=float(spot),
            breakevens=[],
            theme=theme,
            title=f"{symbol} — {title_label} @ {chosen_date} (mark only)",
        )
        # Keep only the evaluation-date line (duplicate traces otherwise)
        keep = [t for t in fig.data if getattr(t, "name", None) == "At evaluation date"]
        if keep:
            fig.data = tuple(keep[:1])
    else:
        fig = charts.strategy_pnl_chart(
            out["S_grid"],
            out["pnl_expiry"],
            out["pnl_now"],
            spot=float(spot),
            breakevens=be,
            theme=theme,
            title=f"{symbol} — {title_label} @ {chosen_date} → expiry {expiry}",
        )
    st.plotly_chart(fig, use_container_width=True)

    dl = edited[edited["quantity"] != 0] if "quantity" in edited.columns else edited
    if "remove" in dl.columns:
        dl = dl[~dl["remove"].astype(bool)].drop(columns=["remove"], errors="ignore")
    st.download_button(
        "Download legs.csv",
        data=dl.to_csv(index=False).encode("utf-8"),
        file_name=f"{symbol}-strategy-legs.csv",
        mime="text/csv",
        key="strat_dl",
    )


def _render_earnings_tab(
    symbol: str,
    bars: pd.DataFrame,
    active_bundle: dict[str, Any],
    data_dir: Path,
    prefs: Preferences,
) -> None:
    """Earnings calendar, price markers, and per-run event metrics."""
    st.subheader("Earnings")
    if not symbol:
        st.info("Set a symbol in the sidebar.")
        return
    earnings = _load_earnings_cached(str(data_dir), symbol)
    if earnings.empty:
        st.warning(
            f"No earnings calendar for {symbol}. Run `lambdaclass fetch-earnings {symbol}` (or `--csv PATH`)."
        )
    else:
        st.caption(
            f"{len(earnings)} events · source={earnings['source'].iloc[-1] if 'source' in earnings.columns else '?'}"
        )
        st.dataframe(earnings, use_container_width=True)

    if bars.empty:
        st.info("No bars loaded for price chart.")
    else:
        theme = prefs.reporting.plot_theme
        st.plotly_chart(
            charts.price_with_earnings(
                bars,
                earnings if not earnings.empty else None,
                theme=theme,
                title=f"{symbol} — price with earnings",
            ),
            use_container_width=True,
        )

    st.markdown("### Event metrics (active run)")
    run_dir = Path(active_bundle["run_dir"])
    events_path = run_dir / "events.parquet"
    mtime = events_path.stat().st_mtime_ns if events_path.is_file() else 0
    events = _load_events_cached(str(run_dir), mtime)
    metrics = active_bundle.get("metrics") or {}
    ekeys = [k for k in metrics if str(k).startswith("earnings_")]
    if ekeys:
        cols = st.columns(min(4, len(ekeys)))
        for i, k in enumerate(sorted(ekeys)):
            cols[i % len(cols)].metric(k.replace("_", " ").title(), f"{metrics[k]:.4f}")
    if events.empty:
        st.caption("No events.parquet for this run (strategy did not trade around earnings, or no calendar).")
    else:
        st.dataframe(events, use_container_width=True)
        st.download_button(
            "Download events.csv",
            data=events.to_csv(index=False).encode("utf-8"),
            file_name=f"{active_bundle['run_id']}-events.csv",
            mime="text/csv",
            key="dl_events",
        )


@st.cache_data(show_spinner=False)
def _load_sweep_cached(sweep_dir_str: str, mtime_ns: int) -> dict[str, Any]:
    bundle = loader.load_sweep(Path(sweep_dir_str))
    return {
        "sweep_id": bundle.sweep_id,
        "strategy": bundle.strategy,
        "manifest": bundle.manifest,
        "results": bundle.results,
    }


def _sweep_mtime_ns(sweep_dir: Path) -> int:
    path = sweep_dir / SWEEP_RESULTS_FILE
    return path.stat().st_mtime_ns if path.is_file() else 0


def _render_sweeps_tab(runs_root: Path, selected_strategies: list[str], prefs: Preferences) -> None:
    sweep_dirs = loader.list_sweeps(runs_root)
    if not sweep_dirs:
        st.info("No sweeps yet. Run `lambdaclass sweep STRATEGY --grid key=a,b` to create one.")
        return
    sweeps = [_load_sweep_cached(str(path), _sweep_mtime_ns(path)) for path in sweep_dirs]
    in_selection = [index for index, sweep in enumerate(sweeps) if sweep["strategy"] in selected_strategies]
    choice = st.selectbox(
        "Sweep",
        options=list(range(len(sweeps))),
        index=in_selection[0] if in_selection else 0,
        format_func=lambda index: (
            f"{sweeps[index]['strategy']} · {sweeps[index]['sweep_id']} · "
            f"{len(sweeps[index]['results'])} combos"
        ),
    )
    sweep = sweeps[choice]
    manifest: dict[str, Any] = sweep["manifest"]
    results: pd.DataFrame = sweep["results"]
    fill_timing = (manifest.get("preferences") or {}).get("defaults", {}).get("fill_timing", "same_close")
    st.caption(
        f"Symbol {manifest.get('symbol', '?')} · window {manifest.get('start') or 'start'} → "
        f"{manifest.get('end') or 'end'} · chain {manifest.get('options_source', '?')} · "
        f"fills {fill_timing} · {manifest.get('completed', 0)} ok / {manifest.get('failed', 0)} failed"
    )
    if manifest.get("fixed_params"):
        st.caption("Fixed params: " + ", ".join(f"{k}={v}" for k, v in manifest["fixed_params"].items()))
    if results.empty:
        st.warning("This sweep has no results.")
        return

    params = loader.sweep_param_names(results)
    metrics = loader.sweep_metric_names(results)
    if not metrics:
        st.warning("No numeric metrics in this sweep (all combinations failed?).")
        st.dataframe(results, use_container_width=True)
        return
    manifest_metric = str(manifest.get("metric", ""))
    default_metric = manifest_metric if manifest_metric in metrics else metrics[0]
    col_metric, col_order = st.columns([3, 1])
    metric = col_metric.selectbox("Metric", options=metrics, index=metrics.index(default_metric))
    minimize = col_order.checkbox("Lower is better", value=bool(manifest.get("minimize", False)))

    ranked = results.sort_values(metric, ascending=minimize, kind="stable", na_position="last")
    st.markdown("### Results")
    st.dataframe(ranked, use_container_width=True, hide_index=True)
    st.download_button(
        "Download sweep.csv",
        data=ranked.to_csv(index=False).encode("utf-8"),
        file_name=f"{sweep['sweep_id']}-sweep.csv",
        mime="text/csv",
        key="dl_sweep",
    )

    st.markdown("### Heatmap")
    col_x, col_y = st.columns(2)
    x_param = col_x.selectbox("X param", options=params, index=0)
    y_options: list[str | None] = [None, *[name for name in params if name != x_param]]
    y_param = col_y.selectbox(
        "Y param",
        options=y_options,
        index=1 if len(y_options) > 1 else 0,
        format_func=lambda name: "(none)" if name is None else name,
    )
    fixed: dict[str, Any] = {}
    others = [name for name in params if name not in (x_param, y_param)]
    if others:
        fixed_cols = st.columns(len(others))
        for column, name in zip(fixed_cols, others, strict=True):
            values = sorted(results[f"{PARAM_PREFIX}{name}"].dropna().unique().tolist())
            fixed[name] = column.selectbox(f"Hold {name} at", options=values, key=f"sweep_fixed_{name}")
    pivot = loader.sweep_pivot(results, x=x_param, y=y_param, metric=metric, fixed=fixed)
    st.plotly_chart(
        charts.sweep_heatmap(
            pivot, metric=metric, x_label=x_param, y_label=y_param, theme=prefs.reporting.plot_theme
        ),
        use_container_width=True,
    )

    ok = ranked[ranked["status"] == "ok"]
    top_n = int(
        st.number_input("Equity overlay: top N", min_value=1, max_value=20, value=min(5, max(len(ok), 1)))
    )
    overlay: dict[str, pd.DataFrame] = {}
    for _, row in ok.head(top_n).iterrows():
        run_dir = Path(str(row["run_dir"]))
        if not (run_dir / "metrics.json").is_file():
            continue
        bundle = _load_run_cached(str(run_dir), loader.run_dir_mtime_ns(run_dir))
        label = " ".join(f"{name}={row[PARAM_PREFIX + name]}" for name in params)
        overlay[label] = bundle["equity_curve"]
    if overlay:
        st.plotly_chart(charts.equity_overlay(overlay, normalize=True), use_container_width=True)
    if len(ok) > 1:
        st.caption(
            "The best of many combinations is an optimistic estimate; confirm it on data the sweep did not see."
        )


def main() -> None:
    st.title("LambdaClass - Backtest Review")
    root = _repo_root()
    prefs = _load_preferences()
    runs_root = _resolve_path(root, prefs.paths.runs_dir)
    data_dir = _resolve_path(root, prefs.paths.data_dir)

    st.sidebar.header("Selection")
    strategies = _list_strategies(str(runs_root))
    if not strategies:
        st.warning(f"No runs found under {runs_root}. Run `lambdaclass run STRATEGY` first.")
        return

    selected_strategies = st.sidebar.multiselect("Strategies", options=strategies, default=strategies[:1])
    if not selected_strategies:
        st.info("Pick at least one strategy in the sidebar.")
        return

    candidate_runs: list[str] = []
    for strat in selected_strategies:
        candidate_runs.extend(_runs_for_strategy(str(runs_root), strat))
    if not candidate_runs:
        st.warning("No runs for selected strategies.")
        return

    if len(selected_strategies) == 1:
        chosen_run = st.sidebar.selectbox(
            "Run",
            options=candidate_runs,
            index=0,
            format_func=_format_run_label,
        )
        chosen_runs = [chosen_run]
    else:
        chosen_runs = st.sidebar.multiselect(
            "Runs",
            options=candidate_runs,
            default=candidate_runs[: min(4, len(candidate_runs))],
            format_func=_format_run_label,
        )
        if not chosen_runs:
            st.info("Pick at least one run.")
            return

    bundles: list[dict[str, Any]] = []
    for run_dir_str in chosen_runs:
        bundle = _load_run_cached(run_dir_str, loader.run_dir_mtime_ns(Path(run_dir_str)))
        bundles.append(bundle)

    active = bundles[0]
    symbol_input = st.sidebar.text_input("Symbol", value=active["symbol"] or "SPY")
    try:
        symbol_override = validate_symbol(symbol_input)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        st.stop()
    start_override = st.sidebar.text_input("Start (YYYY-MM-DD)", value=active["start"])
    end_override = st.sidebar.text_input("End (YYYY-MM-DD)", value=active["end"])

    bars = _load_bars_cached(str(data_dir), symbol_override, start_override, end_override)
    if bars.empty:
        st.sidebar.warning(
            f"No bars at data/stocks/{symbol_override}.parquet. Run `lambdaclass fetch {symbol_override}`."
        )

    snapshot_params = (
        active["snapshot"].get("strategy_params", {}) if isinstance(active["snapshot"], dict) else {}
    )

    def _param_int(name: str, default: int) -> int:
        raw = snapshot_params.get(name)
        if raw in (None, "", "***REDACTED***"):
            return default
        try:
            return int(float(str(raw)))
        except ValueError:
            return default

    fast_default = _param_int("fast", 10)
    slow_default = _param_int("slow", 30)

    show_sma = st.sidebar.checkbox("Show SMA overlays", value=True)
    fast_window = st.sidebar.number_input("SMA fast window", min_value=2, max_value=400, value=fast_default)
    slow_window = st.sidebar.number_input("SMA slow window", min_value=3, max_value=400, value=slow_default)

    indicator_overlays: dict[str, pd.Series] = {}
    if show_sma and not bars.empty and "close" in bars.columns:
        indicator_overlays[f"SMA({fast_window})"] = indicators.sma(bars["close"], int(fast_window))
        indicator_overlays[f"SMA({slow_window})"] = indicators.sma(bars["close"], int(slow_window))

    if any(v == "***REDACTED***" for v in snapshot_params.values()):
        st.sidebar.warning("Snapshot has redacted strategy_params; some overlays use defaults.")

    report_path = Path(active["run_dir"]) / "report.html"
    if report_path.is_file():
        st.sidebar.download_button(
            "Download report.html",
            data=report_path.read_bytes(),
            file_name=f"{active['run_id']}-report.html",
            mime="text/html",
        )
    else:
        st.sidebar.info("report.html not generated for this run.")

    tab_run, tab_compare, tab_sweeps, tab_chain, tab_strategy, tab_earnings = st.tabs(
        ["Run", "Compare", "Sweeps", "Chain", "Strategy", "Earnings"]
    )
    with tab_run:
        _render_run_tab(active, bars, indicator_overlays)
    with tab_compare:
        _render_compare_tab(bundles, runs_root)
    with tab_sweeps:
        _render_sweeps_tab(runs_root, selected_strategies, prefs)
    with tab_chain:
        _render_chain_tab(active, bars, prefs, root)
    with tab_strategy:
        _render_strategy_tab(symbol_override, bars, prefs, root)
    with tab_earnings:
        _render_earnings_tab(symbol_override, bars, active, data_dir, prefs)
    with tab_volatility:
        _render_volatility_tab(symbol_override, prefs, data_dir)


main()
