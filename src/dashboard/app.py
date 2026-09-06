"""Streamlit entry point for the solar monitoring prototype dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


# Keep ``streamlit run src/dashboard/app.py`` reliable from the repository root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.charts import COLORS, anomaly_chart, measurement_chart
from src.dashboard.components import (
    render_current_reading,
    render_events,
    render_header,
    render_model_information,
    render_replay_controls,
)
from src.dashboard.data_service import (
    DashboardDataError,
    build_dashboard_data,
    build_event_table,
    dashboard_summary,
    select_current_row,
)


PAGES = (
    "Overview",
    "Operational Monitoring",
    "Panel Inspection",
    "Drone Video Inspection",
)


@st.cache_data(show_spinner=False)
def load_verified_replay():
    """Cache data plus frozen-score verification across Streamlit reruns."""

    return build_dashboard_data(verify_frozen_scores=True)


def render_overview(frame, selected_index: int) -> None:
    """Render truthful system-level metrics and module readiness."""

    st.header("System Overview")
    st.info(
        "Recorded Operational Dataset Replay — these measurements are historical "
        "project data, not live plant telemetry."
    )
    selected = select_current_row(frame, selected_index)
    summary = dashboard_summary(frame)
    metrics = st.columns(5)
    metrics[0].metric("Monitored Readings", f"{summary['readings']:,}")
    metrics[1].metric("Selected Power", f"{selected['Power_Generated']:.2f} W")
    metrics[2].metric("Selected Solar Radiation", f"{selected['Solar_Radiation']:.2f} W/m²")
    metrics[3].metric("Detected Alerts", f"{summary['alert_rows']:,}")
    metrics[4].metric("System Status", str(selected["status"]))

    st.subheader("Replay selection")
    selected_index = render_replay_controls(len(frame))
    selected = select_current_row(frame, selected_index)
    st.caption(
        f"Selected recorded timestamp: {selected['Timestamp']:%Y-%m-%d %H:%M:%S} "
        "(timezone not specified)"
    )

    operational, vision = st.columns(2)
    with operational:
        st.subheader("Operational anomaly module")
        st.success("PROTOTYPE COMPLETE")
        st.write(
            "Frozen seven-feature Isolation Forest with recorded score replay, "
            "strict NORMAL/ALERT mapping, and grouped event output."
        )
        st.metric("Alert Events", summary["alert_events"])
    with vision:
        st.subheader("Computer Vision Module")
        st.success("READY FOR INTEGRATION")
        st.write(
            "Frozen YOLO11n prototype with six visual classes and reusable image, "
            "frame, and video inference modules."
        )
        st.warning(
            "Prototype performance is weak for dusty panels and remains limited "
            "for electrical and physical damage."
        )

    st.subheader("Dataset information")
    st.write(
        f"{len(frame)} replay readings from recorded evaluation-baseline data; "
        "the original cadence is approximately two minutes. The source covers "
        "only a short period and does not represent long-term or seasonal operation."
    )
    render_model_information(float(frame["threshold"].iloc[0]))


def render_operational_monitoring(frame, selected_index: int) -> None:
    """Render replay controls, measurements, charts, and real alert events."""

    st.header("Operational Monitoring")
    st.caption(
        "Monitoring Data Replay — sequential review of recorded measurements and "
        "unchanged frozen-model outputs."
    )
    selected_index = render_replay_controls(len(frame))
    selected = select_current_row(frame, selected_index)
    render_current_reading(selected)

    st.subheader("Power and solar input")
    st.plotly_chart(
        measurement_chart(
            frame,
            "Power_Generated",
            title="Generated power over recorded time",
            y_axis_title="Power Generated (W)",
            selected_index=selected_index,
            color=COLORS["power"],
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.plotly_chart(
        measurement_chart(
            frame,
            "Solar_Radiation",
            title="Solar radiation over recorded time",
            y_axis_title="Solar Radiation (W/m²)",
            selected_index=selected_index,
            color=COLORS["solar"],
        ),
        width="stretch",
        config={"displaylogo": False},
    )

    st.subheader("Anomaly monitoring")
    st.plotly_chart(
        anomaly_chart(frame, selected_index),
        width="stretch",
        config={"displaylogo": False},
    )
    alert_count = int(frame["status"].eq("ALERT").sum())
    alert_metric, event_metric = st.columns(2)
    events = build_event_table(frame)
    alert_metric.metric("ALERT readings", alert_count)
    event_metric.metric("Grouped alert events", len(events))
    render_events(events)

    st.subheader("Environmental conditions")
    environmental_options = {
        "Air Temperature": ("Air_Temp", "Air Temperature (°C)", COLORS["temperature"]),
        "Relative Humidity": (
            "Relative_Humidity", "Relative Humidity (%RH)", COLORS["humidity"]
        ),
        "Wind Speed": ("Wind_Speed", "Wind Speed (m/s)", COLORS["wind"]),
    }
    choice = st.selectbox(
        "Environmental metric",
        tuple(environmental_options),
        help="Each measurement is shown on its own correctly labelled axis.",
    )
    column, axis_title, color = environmental_options[choice]
    st.plotly_chart(
        measurement_chart(
            frame,
            column,
            title=f"{choice} over recorded time",
            y_axis_title=axis_title,
            selected_index=selected_index,
            color=color,
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    render_model_information(float(frame["threshold"].iloc[0]))


def render_panel_placeholder() -> None:
    """Describe prepared image support without exposing inactive controls."""

    st.header("Solar Panel Image Inspection")
    st.info(
        "Computer-vision integration prepared — available in the next "
        "system-integration stage."
    )
    st.subheader("Supported visual classes")
    st.write(
        "bird-drop · clean · dusty · electrical-damage · physical-damage · snow-covered"
    )
    st.write(
        "Image upload and inference integration will be connected in the next "
        "system-integration stage. No prediction is generated on this page."
    )
    st.warning(
        "The frozen detector is a limited prototype, with especially weak dusty "
        "recall and limited damage-class performance."
    )


def render_video_placeholder() -> None:
    """Describe prepared video processing without implying bundled drone data."""

    st.header("Drone Video Inspection")
    st.info(
        "Video interface integration is prepared for the next system-integration stage."
    )
    st.write(
        "The existing processor supports sequential frame inference, annotated "
        "output video, class counts, frame summaries, timestamps, and a detection log."
    )
    st.warning(
        "No real drone footage is bundled with this prototype. The engineering "
        "smoke test used a clearly labelled sequence created from validation images."
    )


def run_app() -> None:
    """Configure and render the selected dashboard section."""

    st.set_page_config(
        page_title="Solar Panel Monitoring and Fault Detection System",
        page_icon="☀️",
        layout="wide",
    )
    render_header()
    with st.sidebar:
        st.subheader("Navigation")
        page = st.radio("Section", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("Data mode")
        st.write("**Recorded Operational Dataset Replay**")
        st.caption("Timezone: not specified by source")

    try:
        with st.spinner("Verifying recorded feed and frozen anomaly artifacts…"):
            frame = load_verified_replay()
    except (DashboardDataError, FileNotFoundError, OSError, ValueError) as error:
        st.error(f"Dashboard data could not be loaded: {error}")
        st.info(
            "Check the recorded feed, evaluation-baseline CSV, and frozen anomaly "
            "artifact files, then reload the application."
        )
        st.stop()

    selected_index = min(
        max(int(st.session_state.get("replay_position", 0)), 0), len(frame) - 1
    )
    if page == "Overview":
        render_overview(frame, selected_index)
    elif page == "Operational Monitoring":
        render_operational_monitoring(frame, selected_index)
    elif page == "Panel Inspection":
        render_panel_placeholder()
    else:
        render_video_placeholder()

    st.divider()
    st.caption(
        "TECHNOVOLT university project prototype — recorded replay, not live plant "
        "telemetry and not production-ready."
    )


if __name__ == "__main__":
    run_app()
