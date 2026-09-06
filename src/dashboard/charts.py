"""Plotly chart builders for recorded operational monitoring."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


COLORS = {
    "power": "#2563eb",
    "solar": "#f59e0b",
    "score": "#7c3aed",
    "threshold": "#dc2626",
    "selected": "#0f172a",
    "alert": "#dc2626",
    "temperature": "#ea580c",
    "humidity": "#0891b2",
    "wind": "#16a34a",
}


def _selected_indicator(
    figure: go.Figure, frame: pd.DataFrame, selected_index: int, column: str
) -> None:
    row = frame.iloc[selected_index]
    figure.add_vline(
        x=row["Timestamp"].timestamp() * 1000,
        line_width=1,
        line_dash="dot",
        line_color=COLORS["selected"],
    )
    figure.add_trace(
        go.Scatter(
            x=[row["Timestamp"]],
            y=[row[column]],
            mode="markers",
            marker={"size": 10, "color": COLORS["selected"], "symbol": "diamond"},
            name="Selected reading",
            hovertemplate="Selected<br>%{x}<br>%{y:.4f}<extra></extra>",
        )
    )


def measurement_chart(
    frame: pd.DataFrame,
    column: str,
    *,
    title: str,
    y_axis_title: str,
    selected_index: int,
    color: str,
) -> go.Figure:
    """Build one selected-reading-aware operational measurement chart."""

    figure = go.Figure(
        go.Scatter(
            x=frame["Timestamp"],
            y=frame[column],
            mode="lines",
            line={"color": color, "width": 2},
            name=title,
            hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
        )
    )
    _selected_indicator(figure, frame, selected_index, column)
    figure.update_layout(
        title=title,
        xaxis_title="Recorded timestamp (timezone not specified)",
        yaxis_title=y_axis_title,
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        height=340,
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return figure


def anomaly_chart(frame: pd.DataFrame, selected_index: int) -> go.Figure:
    """Plot actual anomaly scores, the frozen threshold, and real alert rows."""

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["Timestamp"], y=frame["anomaly_score"], mode="lines",
            line={"color": COLORS["score"], "width": 2}, name="Anomaly score",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["Timestamp"], y=frame["threshold"], mode="lines",
            line={"color": COLORS["threshold"], "width": 2, "dash": "dash"},
            name="Frozen threshold",
        )
    )
    alerts = frame.loc[frame["status"].eq("ALERT")]
    if not alerts.empty:
        figure.add_trace(
            go.Scatter(
                x=alerts["Timestamp"], y=alerts["anomaly_score"], mode="markers",
                marker={"color": COLORS["alert"], "size": 9}, name="ALERT",
            )
        )
    _selected_indicator(figure, frame, selected_index, "anomaly_score")
    figure.update_layout(
        title="Anomaly score and frozen threshold",
        xaxis_title="Recorded timestamp (timezone not specified)",
        yaxis_title="Anomaly score",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        height=370,
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return figure
