from __future__ import annotations

from pathlib import Path

import pandas as pd

from lambdaclass.reporting.dashboard.charts import equity_with_drawdown


def write_tearsheet(equity_curve: pd.DataFrame, destination: Path, theme: str = "plotly_dark") -> Path:
    fig = equity_with_drawdown(equity_curve, theme=theme, include_drawdown=False, title="Equity Curve")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(destination), include_plotlyjs="cdn")
    return destination
