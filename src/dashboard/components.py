"""Reusable Streamlit presentation components for the prototype dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard.data_service import clamp_replay_index, move_replay_index


def render_header() -> None:
    """Render the application identity without implying production readiness."""

    title, badge = st.columns([5, 1])
    with title:
        st.title("Solar Panel Monitoring and Fault Detection System")
        st.caption(
            "Prototype combining recorded operational monitoring, anomaly "
            "detection, and prepared visual solar-panel inspection."
        )
    with badge:
        st.markdown("### `PROTOTYPE`")


def _step_replay(delta: int, row_count: int) -> None:
    st.session_state.replay_position = move_replay_index(
        st.session_state.get("replay_position", 0), delta, row_count
    )


def _reset_replay() -> None:
    st.session_state.replay_position = 0


def render_replay_controls(row_count: int) -> int:
    """Render bounded sequential controls and return the active row position."""

    if "replay_position" not in st.session_state:
        st.session_state.replay_position = 0
    st.session_state.replay_position = clamp_replay_index(
        st.session_state.replay_position, row_count
    )
    previous, following, reset = st.columns(3)
    previous.button(
        "Previous Reading",
        on_click=_step_replay,
        args=(-1, row_count),
        disabled=st.session_state.replay_position == 0,
        width="stretch",
    )
    following.button(
        "Next Reading",
        on_click=_step_replay,
        args=(1, row_count),
        disabled=st.session_state.replay_position == row_count - 1,
        width="stretch",
    )
    reset.button(
        "Reset Replay",
        on_click=_reset_replay,
        disabled=st.session_state.replay_position == 0,
        width="stretch",
    )
    return st.slider(
        "Replay position",
        min_value=0,
        max_value=row_count - 1,
        key="replay_position",
        help="Move through recorded readings in their original chronological order.",
    )


def render_current_reading(row: pd.Series) -> None:
    """Separate measurements from frozen model output in two concise panels."""

    st.subheader("Selected recorded measurement")
    st.caption(f"Timestamp: {row['Timestamp']:%Y-%m-%d %H:%M:%S} — timezone not specified")
    columns = st.columns(5)
    columns[0].metric("Power Generated", f"{row['Power_Generated']:.2f} W")
    columns[1].metric("Solar Radiation", f"{row['Solar_Radiation']:.2f} W/m²")
    columns[2].metric("Air Temperature", f"{row['Air_Temp']:.2f} °C")
    columns[3].metric("Relative Humidity", f"{row['Relative_Humidity']:.2f} %RH")
    columns[4].metric("Wind Speed", f"{row['Wind_Speed']:.2f} m/s")

    st.subheader("Frozen anomaly-model result")
    score, threshold, status, rtd_mean, rtd_std = st.columns(5)
    score.metric("Anomaly Score", f"{row['anomaly_score']:.6f}")
    threshold.metric("Frozen Threshold", f"{row['threshold']:.6f}")
    status.metric("Status", str(row["status"]))
    rtd_mean.metric("RTD Mean", f"{row['rtd_mean']:.3f}")
    rtd_std.metric("RTD Std", f"{row['rtd_std']:.3f}")


def render_model_information(threshold: float) -> None:
    """Explain the frozen operational prototype and its negative result."""

    with st.expander("Prototype Model Information"):
        st.markdown(
            f"""
- **Method:** Isolation Forest using seven operational features.
- **Frozen threshold:** `{threshold:.15f}` with strict `score > threshold` alerts.
- **Status:** operational anomaly prototype complete; no fitting occurs in the dashboard.
- **Known limitation:** the final controlled evaluation did not detect any of
  the three moderate synthetic events. The prototype is not validated for real
  photovoltaic fault diagnosis or production alarms.
"""
        )


def render_events(events: pd.DataFrame) -> None:
    """Render actual grouped alert events or an explicit zero-event message."""

    st.subheader("Threshold-crossing events")
    if events.empty:
        st.info("No threshold-crossing anomaly events were detected in this replay.")
        return
    display = events.rename(
        columns={
            "event_id": "Event",
            "start_timestamp": "Start",
            "end_timestamp": "End",
            "row_count": "Alert readings",
            "duration_minutes": "Duration (minutes)",
            "maximum_anomaly_score": "Maximum score",
            "mean_anomaly_score": "Mean score",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")


def format_metric_value(value: Any, unit: str = "") -> str:
    """Format dashboard metrics consistently while keeping units explicit."""

    return f"{float(value):.2f}{unit}"
