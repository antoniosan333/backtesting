"""Plotly charts for the volatility dashboard tab."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from lambdaclass.reporting.dashboard.charts import _empty_figure


def iv_versus_hv(
    series: pd.DataFrame,
    earnings: pd.DataFrame | None = None,
    *,
    theme: str = "plotly_dark",
) -> go.Figure:
    """IV30 against trailing realized volatility, with earnings dates marked."""
    if series is None or series.empty:
        return _empty_figure("Implied vs historical volatility", theme)
    fig = go.Figure()
    for column, name in (("iv30", "IV30"), ("hv20", "HV20"), ("hv_yz20", "Yang-Zhang 20")):
        if column in series.columns:
            fig.add_trace(go.Scatter(x=series["date"], y=series[column], mode="lines", name=name))
    if earnings is not None and not earnings.empty and "earnings_date" in earnings.columns:
        for day in earnings["earnings_date"].astype(str).str[:10]:
            fig.add_vline(x=day, line_width=1, line_dash="dot", opacity=0.4)
    fig.update_layout(template=theme, title="Implied vs historical volatility", hovermode="x unified")
    return fig


def iv_gauge(series: pd.DataFrame, *, theme: str = "plotly_dark") -> go.Figure:
    """252-day IV percentile and rank."""
    if series is None or series.empty:
        return _empty_figure("IV percentile", theme)
    fig = go.Figure()
    for column, name in (("iv_pctile_252", "IV percentile"), ("iv_rank_252", "IV rank")):
        if column in series.columns:
            fig.add_trace(go.Scatter(x=series["date"], y=series[column], mode="lines", name=name))
    for level in (0.2, 0.5, 0.8):
        fig.add_hline(y=level, line_dash="dot", line_width=1)
    fig.update_layout(template=theme, title="Cheap / expensive gauge", hovermode="x unified", yaxis_range=[0, 1])
    return fig


def premium_realized(series: pd.DataFrame, *, theme: str = "plotly_dark") -> go.Figure:
    """IV30 against the realized volatility that followed."""
    if series is None or series.empty or "rv_fwd21" not in series.columns:
        return _empty_figure("Implied versus realized", theme)
    points = series.dropna(subset=["iv30", "rv_fwd21"])
    fig = go.Figure(
        go.Scatter(
            x=points["iv30"],
            y=points["rv_fwd21"],
            mode="markers",
            name="Months",
            marker={"size": 6, "opacity": 0.6},
        )
    )
    if not points.empty:
        limit = float(max(points["iv30"].max(), points["rv_fwd21"].max()))
        fig.add_trace(go.Scatter(x=[0, limit], y=[0, limit], mode="lines", name="IV = realized"))
    fig.update_layout(
        template=theme,
        title="IV30 versus following 21-day realized vol",
        xaxis_title="IV30",
        yaxis_title="Realized vol",
    )
    return fig


def term_structure(term: pd.DataFrame, *, theme: str = "plotly_dark") -> go.Figure:
    """ATM implied volatility by days to expiry."""
    if term is None or term.empty:
        return _empty_figure("Term structure", theme)
    fig = go.Figure(go.Scatter(x=term["dte"], y=term["atm_iv"], mode="lines+markers", name="ATM IV"))
    fig.update_layout(template=theme, title="ATM implied volatility term structure", xaxis_title="DTE")
    return fig


def earnings_cycle_chart(cycle: pd.DataFrame, *, theme: str = "plotly_dark") -> go.Figure:
    """Median normalized front IV around earnings, with the interquartile band."""
    if cycle is None or cycle.empty:
        return _empty_figure("Earnings IV cycle", theme)
    front = cycle[cycle["metric"] == "iv_front_norm"]
    if front.empty:
        front = cycle[cycle["metric"] == "iv_front"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=front["rel_day"], y=front["q75"], mode="lines", line={"width": 0}, showlegend=False)
    )
    fig.add_trace(
        go.Scatter(
            x=front["rel_day"],
            y=front["q25"],
            mode="lines",
            fill="tonexty",
            name="IQR",
            line={"width": 0},
        )
    )
    fig.add_trace(go.Scatter(x=front["rel_day"], y=front["median"], mode="lines", name="Median"))
    fig.update_layout(template=theme, title="Earnings IV cycle", xaxis_title="Sessions vs reaction")
    return fig
