"""Plotly figure factories shared by the dashboard and the per-run HTML tearsheet."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lambdaclass.reporting.dashboard.indicators import equity_drawdown


def _empty_figure(title: str, theme: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, template=theme)
    return fig


def equity_with_drawdown(
    equity_curve: pd.DataFrame,
    *,
    theme: str = "plotly_dark",
    include_drawdown: bool = True,
    title: str = "Equity Curve",
) -> go.Figure:
    """Equity line on top, drawdown area below (when ``include_drawdown=True``).

    With ``include_drawdown=False`` the figure has only the equity line, matching the
    legacy single-panel tearsheet so existing ``report.html`` rendering is unchanged.
    """
    if equity_curve is None or equity_curve.empty or "equity" not in equity_curve.columns:
        return _empty_figure(title, theme)

    df = equity_curve.copy()
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={df.columns[0]: "date"})

    if not include_drawdown:
        fig = px.line(df, x="date", y="equity", title=title)
        fig.update_layout(template=theme)
        return fig

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(title, "Drawdown"),
    )
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["equity"], mode="lines", name="Equity"),
        row=1,
        col=1,
    )
    dd = equity_drawdown(df["equity"])
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=dd,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_layout(template=theme, showlegend=False, hovermode="x unified")
    return fig


def price_with_signals(
    bars: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    indicators: Mapping[str, pd.Series] | None = None,
    *,
    theme: str = "plotly_dark",
    title: str = "Price with Signals",
) -> go.Figure:
    """Candlestick chart with optional indicator overlays and buy/sell triangle markers."""
    if bars is None or bars.empty:
        return _empty_figure(title, theme)
    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(bars.columns):
        return _empty_figure(title, theme)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=bars["date"],
            open=bars["open"],
            high=bars["high"],
            low=bars["low"],
            close=bars["close"],
            name="OHLC",
            showlegend=False,
        )
    )

    if indicators:
        for name, series in indicators.items():
            if series is None or len(series) == 0:
                continue
            fig.add_trace(go.Scatter(x=bars["date"], y=series, mode="lines", name=name))

    if trades is not None and not trades.empty and {"date", "action", "price"}.issubset(trades.columns):
        buys = trades[trades["action"].str.lower() == "buy"]
        sells = trades[trades["action"].str.lower() == "sell"]
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["date"],
                    y=buys["price"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12),
                    name="Buy",
                )
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["date"],
                    y=sells["price"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=12),
                    name="Sell",
                )
            )

    fig.update_layout(
        template=theme,
        title=title,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    return fig


def equity_overlay(
    runs: Mapping[str, pd.DataFrame],
    *,
    normalize: bool = True,
    theme: str = "plotly_dark",
    title: str = "Equity Overlay",
) -> go.Figure:
    """One trace per run. ``runs`` maps label -> equity_curve DataFrame."""
    if not runs:
        return _empty_figure(title, theme)
    fig = go.Figure()
    any_added = False
    for label, df in runs.items():
        if df is None or df.empty or "equity" not in df.columns:
            continue
        x = df["date"] if "date" in df.columns else df.index
        y = df["equity"].astype(float)
        if normalize and float(y.iloc[0]) != 0.0:
            y = y / float(y.iloc[0])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label))
        any_added = True
    if not any_added:
        return _empty_figure(title, theme)
    fig.update_layout(template=theme, title=title, hovermode="x unified")
    return fig


def monthly_heatmap(
    returns_table: pd.DataFrame,
    *,
    theme: str = "plotly_dark",
    title: str = "Monthly Returns",
) -> go.Figure:
    if returns_table is None or returns_table.empty:
        return _empty_figure(title, theme)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    z = returns_table.values
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=months,
            y=[str(int(y)) for y in returns_table.index],
            colorscale="RdYlGn",
            zmid=0.0,
            colorbar=dict(tickformat=".1%"),
        )
    )
    fig.update_layout(template=theme, title=title)
    return fig


def sweep_heatmap(
    pivot: pd.DataFrame,
    *,
    metric: str,
    x_label: str,
    y_label: str | None,
    theme: str = "plotly_dark",
) -> go.Figure:
    """Metric over one or two swept params (``sweep_pivot`` output)."""
    title = f"{metric} by {x_label}" + (f" × {y_label}" if y_label else "")
    if pivot is None or pivot.empty:
        return _empty_figure(title, theme)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(value) for value in pivot.columns],
            y=[str(value) for value in pivot.index],
            colorscale="RdYlGn",
            text=[[f"{value:.4g}" for value in row] for row in pivot.values],
            texttemplate="%{text}",
            hovertemplate=f"{x_label}=%{{x}}<br>{y_label or ''}=%{{y}}<br>{metric}=%{{z:.4f}}<extra></extra>",
        )
    )
    fig.update_layout(template=theme, title=title, xaxis_title=x_label, yaxis_title=y_label or "")
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return fig


def rolling_sharpe_chart(
    series: pd.Series,
    *,
    theme: str = "plotly_dark",
    title: str = "Rolling Sharpe",
) -> go.Figure:
    if series is None or len(series) == 0:
        return _empty_figure(title, theme)
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines", name="Sharpe"))
    fig.update_layout(template=theme, title=title, hovermode="x unified")
    return fig


def strategy_pnl_chart(
    S_grid: Sequence[float] | np.ndarray,
    pnl_expiry: Sequence[float] | np.ndarray,
    pnl_now: Sequence[float] | np.ndarray,
    *,
    spot: float,
    breakevens: Sequence[float],
    theme: str = "plotly_dark",
    title: str = "Strategy P&L",
) -> go.Figure:
    """P&L at expiration vs mark at evaluation date; spot reference and breakeven markers."""
    S = np.asarray(S_grid, dtype=float)
    pe = np.asarray(pnl_expiry, dtype=float)
    pn = np.asarray(pnl_now, dtype=float)
    if S.size == 0 or pe.size != S.size or pn.size != S.size:
        return _empty_figure(title, theme)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S, y=pe, mode="lines", name="At expiration"))
    fig.add_trace(go.Scatter(x=S, y=pn, mode="lines", name="At evaluation date"))
    be = [float(b) for b in breakevens]
    if be:
        fig.add_trace(
            go.Scatter(
                x=be,
                y=[0.0] * len(be),
                mode="markers+text",
                name="Breakevens",
                text=[f"{x:.2f}" for x in be],
                textposition="top center",
                marker=dict(size=10, symbol="diamond"),
            )
        )
    y_min = float(min(pe.min(), pn.min(), 0.0))
    y_max = float(max(pe.max(), pn.max(), 0.0))
    pad = max((y_max - y_min) * 0.05, 1.0)
    fig.add_shape(
        type="line",
        xref="x",
        yref="y",
        x0=float(spot),
        x1=float(spot),
        y0=y_min - pad,
        y1=y_max + pad,
        line=dict(color="rgba(255,255,255,0.45)", width=2, dash="dash"),
    )
    fig.add_hline(y=0.0, line_color="rgba(200,200,200,0.5)", line_width=1)
    fig.update_layout(
        template=theme,
        title=title,
        hovermode="x unified",
        xaxis_title="Underlying price",
        yaxis_title="P&L ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def chain_iv_scatter(
    chain: pd.DataFrame,
    *,
    theme: str = "plotly_dark",
    title: str = "Strike vs Implied Volatility",
) -> go.Figure:
    if chain is None or chain.empty or not {"strike", "implied_volatility", "side"}.issubset(chain.columns):
        return _empty_figure(title, theme)
    fig = px.scatter(
        chain,
        x="strike",
        y="implied_volatility",
        color="side",
        hover_data=[
            c for c in ("contract_symbol", "expiry", "open_interest", "volume") if c in chain.columns
        ],
        title=title,
    )
    fig.update_layout(template=theme)
    return fig


def price_with_earnings(
    bars: pd.DataFrame,
    earnings: pd.DataFrame | None = None,
    *,
    theme: str = "plotly_dark",
    title: str = "Price with Earnings",
) -> go.Figure:
    """Close line with vertical markers on earnings dates (color by timing)."""
    if bars is None or bars.empty or not {"date", "close"}.issubset(bars.columns):
        return _empty_figure(title, theme)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bars["date"], y=bars["close"], mode="lines", name="Close"))
    if earnings is not None and not earnings.empty and "earnings_date" in earnings.columns:
        closes = bars.copy()
        closes["date"] = closes["date"].astype(str).str[:10]
        y_min = float(bars["close"].min())
        y_max = float(bars["close"].max())
        pad = max((y_max - y_min) * 0.05, 1.0)
        for _, row in earnings.iterrows():
            ed = str(row["earnings_date"])[:10]
            timing = str(row.get("timing", "unknown")).upper()
            color = {"BMO": "rgba(80,180,255,0.7)", "AMC": "rgba(255,160,80,0.7)"}.get(
                timing, "rgba(200,200,200,0.5)"
            )
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=ed,
                x1=ed,
                y0=y_min - pad,
                y1=y_max + pad,
                line=dict(color=color, width=1, dash="dot"),
            )
            # Marker at nearest close
            sub = closes[closes["date"] <= ed]
            y = float(sub["close"].iloc[-1]) if not sub.empty else y_max
            fig.add_trace(
                go.Scatter(
                    x=[ed],
                    y=[y],
                    mode="markers+text",
                    name=f"E:{timing}",
                    text=[timing],
                    textposition="top center",
                    marker=dict(size=9, symbol="diamond", color=color),
                    showlegend=False,
                )
            )
    fig.update_layout(template=theme, title=title, hovermode="x unified")
    return fig
