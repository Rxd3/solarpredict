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
    render_cv_model_information,
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
from src.dashboard.vision_service import (
    VisionUploadError,
    analyze_image_upload,
    decode_image_upload,
    image_detection_table,
    inspect_video_upload,
    process_video_upload,
    sha256_bytes,
    verify_frozen_checkpoint,
)
from src.computer_vision.inference import CLASS_NAMES, load_detector


PAGES = (
    "Overview",
    "Operational Monitoring",
    "Panel Inspection",
    "Video Inspection",
)


@st.cache_data(show_spinner=False)
def load_verified_replay():
    """Cache data plus frozen-score verification across Streamlit reruns."""

    return build_dashboard_data(verify_frozen_scores=True)


@st.cache_resource(show_spinner=False, max_entries=1)
def load_frozen_cv_detector():
    """Verify and cache the selected detector through its existing loader."""

    verify_frozen_checkpoint()
    return load_detector()


@st.cache_data(show_spinner=False, max_entries=4)
def inspect_uploaded_video(content: bytes, filename: str):
    """Cache validated container metadata for unchanged uploaded bytes."""

    return inspect_video_upload(content, filename)


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

    st.subheader("Module status")
    operational, anomaly, image_module, video_module = st.columns(4, border=True)
    with operational:
        st.markdown("**Operational monitoring**")
        st.badge("READY", icon=":material/check:", color="green")
        st.caption("Recorded replay and measurement charts")
    with anomaly:
        st.markdown("**Anomaly detection**")
        st.badge("PROTOTYPE COMPLETE", icon=":material/check:", color="blue")
        st.caption(f"{summary['alert_events']} grouped alert events")
    with image_module:
        st.markdown("**Image inspection**")
        st.badge("READY", icon=":material/check:", color="green")
        st.caption("Frozen six-class YOLO inference")
    with video_module:
        st.markdown("**Video inspection**")
        st.badge("READY", icon=":material/check:", color="green")
        st.caption("Frame processing and downloadable results")

    st.warning(
        "Computer-vision performance varies substantially by class: dusty recall "
        "is especially weak, and electrical/physical damage remain limited."
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


def _show_cv_status(summary: dict[str, object]) -> None:
    status = str(summary["overall_status"])
    message = str(summary.get("message", ""))
    if status == "ATTENTION":
        st.warning(f"**ATTENTION** — {message}", icon=":material/warning:")
    elif status == "CLEAN":
        st.success(f"**CLEAN model output** — {message}", icon=":material/check_circle:")
    else:
        st.info(f"**NO_DETECTION** — {message}", icon=":material/info:")


def render_panel_inspection() -> None:
    """Decode an uploaded image and run deliberate frozen-model inference."""

    st.header("Solar panel image inspection")
    st.caption(
        "Upload a JPG, JPEG, or PNG image for screening with the frozen YOLO11n "
        "prototype. Uploaded bytes are not permanently saved."
    )
    st.warning(
        "Performance varies substantially by class. Dusty recall is particularly "
        "low, electrical and physical damage also have limited recall, and results "
        "must not replace professional inspection."
    )
    uploaded = st.file_uploader(
        "Solar-panel image",
        type=["jpg", "jpeg", "png"],
        key="panel_image_upload",
        max_upload_size=20,
        help="The file extension is filtered by the browser and the actual bytes are decoded by OpenCV.",
    )
    confidence = st.slider(
        "Display/deployment confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        key="panel_image_confidence",
        help=(
            "Controls which boxes are displayed for this upload. It is not model "
            "accuracy and does not alter frozen evaluation metrics."
        ),
    )
    if uploaded is None:
        st.info("Upload an image to preview it. Inference starts only after Analyze Image.")
        render_cv_model_information()
        return

    content = uploaded.getvalue()
    try:
        decoded = decode_image_upload(content, uploaded.name)
    except (VisionUploadError, OSError, ValueError) as error:
        st.error(f"Image upload could not be used: {error}", icon=":material/error:")
        render_cv_model_information()
        return

    st.write(f"**File:** `{decoded['filename']}`")
    st.caption(f"Dimensions: {decoded['width']} × {decoded['height']} pixels")
    st.image(
        decoded["frame"],
        caption="Validated uploaded image — not yet analyzed",
        channels="BGR",
        width="stretch",
    )

    signature = (decoded["sha256"], float(confidence))
    if st.session_state.get("panel_image_result_signature") != signature:
        st.session_state.pop("panel_image_result", None)
    if st.button(
        "Analyze Image",
        type="primary",
        icon=":material/search:",
        key="analyze_image",
    ):
        progress = st.status("Running frozen image inference…", expanded=True)
        try:
            progress.write("Verifying and loading the selected checkpoint.")
            detector = load_frozen_cv_detector()
            progress.write("Running CPU inference and rendering actual detections.")
            result = analyze_image_upload(
                content,
                uploaded.name,
                detector=detector,
                confidence=float(confidence),
            )
        except (VisionUploadError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            progress.update(label="Image analysis failed", state="error", expanded=True)
            st.error(f"The image could not be analyzed: {error}", icon=":material/error:")
        else:
            st.session_state.panel_image_result = result
            st.session_state.panel_image_result_signature = signature
            progress.update(label="Image analysis complete", state="complete", expanded=False)

    result = st.session_state.get("panel_image_result")
    if result is None or st.session_state.get("panel_image_result_signature") != signature:
        render_cv_model_information()
        return

    prediction = result["prediction"]
    condition = prediction["condition_summary"]
    with st.container(horizontal=True):
        st.metric("Total detections", prediction["detection_count"], border=True)
        highest = prediction["highest_confidence"]
        st.metric(
            "Highest confidence",
            "—" if highest is None else f"{float(highest):.3f}",
            border=True,
        )
        detected = condition["detected_conditions"]
        st.metric("Conditions detected", len(detected), border=True)
        st.metric("Inspection status", condition["overall_status"], border=True)
    _show_cv_status(
        {
            **condition,
            "message": {
                "ATTENTION": "One or more non-clean visual conditions were detected for review.",
                "CLEAN": "Only clean-class detections were produced; this is not certification.",
                "NO_DETECTION": (
                    "No visual conditions were detected by the model. This does not "
                    "certify that the panel is healthy or fault-free."
                ),
            }[condition["overall_status"]],
        }
    )

    original_column, annotated_column = st.columns(2)
    with original_column:
        st.subheader("Original upload")
        st.image(decoded["frame"], channels="BGR", width="stretch")
    with annotated_column:
        st.subheader("Annotated model output")
        st.image(result["annotated_png"], width="stretch")

    detection_table = image_detection_table(prediction)
    st.subheader("Detections")
    if detection_table.empty:
        st.info("No detection rows are available at the selected display threshold.")
    else:
        st.dataframe(
            detection_table,
            hide_index=True,
            width="stretch",
            column_config={
                "Confidence": st.column_config.NumberColumn(format="%.3f"),
                "x1": st.column_config.NumberColumn(format="%.1f"),
                "y1": st.column_config.NumberColumn(format="%.1f"),
                "x2": st.column_config.NumberColumn(format="%.1f"),
                "y2": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        counts = [
            {"Class": name, "Detections": condition["counts_by_class"][name]}
            for name in CLASS_NAMES
            if condition["counts_by_class"][name] > 0
        ]
        st.subheader("Detected class counts")
        st.dataframe(counts, hide_index=True, width="content")
    st.download_button(
        "Download annotated image",
        data=result["annotated_png"],
        file_name=result["annotated_filename"],
        mime="image/png",
        icon=":material/download:",
        on_click="ignore",
    )
    render_cv_model_information()


def render_video_inspection() -> None:
    """Validate and deliberately process uploaded solar-panel video."""

    st.header("Video inspection")
    st.caption(
        "Process uploaded drone footage or other solar-panel video frame by frame. "
        "An upload is not described as drone footage unless you identify it that way."
    )
    st.info(
        "Frame stride 1 analyzes every frame. Higher values reduce CPU inference "
        "work; skipped inference frames remain in the annotated output and are "
        "explicitly marked in its frame log."
    )
    uploaded = st.file_uploader(
        "Solar-panel video",
        type=["mp4", "avi", "mov"],
        key="panel_video_upload",
        max_upload_size=200,
        help="Uploads are processed in a generated temporary workspace and removed afterward.",
    )
    confidence = st.slider(
        "Video display/deployment confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        key="panel_video_confidence",
        help="This viewing threshold is not model accuracy and was not tuned from test data.",
    )
    frame_stride = st.number_input(
        "Frame stride",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="panel_video_stride",
    )
    if uploaded is None:
        st.info("Upload a video to validate it. Processing starts only after Process Video.")
        st.caption(
            "No real drone footage is bundled. The existing demo is a sequence "
            "created from validation images for video-pipeline testing."
        )
        render_cv_model_information()
        return

    content = uploaded.getvalue()
    try:
        metadata = inspect_uploaded_video(content, uploaded.name)
    except (VisionUploadError, OSError, ValueError) as error:
        st.error(f"Video upload could not be used: {error}", icon=":material/error:")
        render_cv_model_information()
        return

    st.write(f"**File:** `{metadata['filename']}`")
    st.caption(
        f"Validated video: {metadata['reported_frame_count']} frames · "
        f"{metadata['fps']:.3f} FPS · {metadata['width']} × {metadata['height']} · "
        f"approximately {metadata['duration_seconds']:.2f} seconds"
    )
    signature = (metadata["sha256"], float(confidence), int(frame_stride))
    if st.session_state.get("panel_video_result_signature") != signature:
        st.session_state.pop("panel_video_result", None)
    if st.button(
        "Process Video",
        type="primary",
        icon=":material/play_arrow:",
        key="process_video",
    ):
        progress = st.status("Processing uploaded video…", expanded=True)
        try:
            progress.write("Verifying and loading the frozen detector.")
            detector = load_frozen_cv_detector()
            progress.write("Analyzing selected frames and writing every output frame.")
            result = process_video_upload(
                content,
                uploaded.name,
                detector=detector,
                confidence=float(confidence),
                frame_stride=int(frame_stride),
            )
            progress.write("Validating logs and preparing browser-compatible playback.")
        except (VisionUploadError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            progress.update(label="Video processing failed", state="error", expanded=True)
            st.error(f"The video could not be processed: {error}", icon=":material/error:")
        else:
            st.session_state.panel_video_result = result
            st.session_state.panel_video_result_signature = signature
            progress.update(label="Video processing complete", state="complete", expanded=False)

    result = st.session_state.get("panel_video_result")
    if result is None or st.session_state.get("panel_video_result_signature") != signature:
        render_cv_model_information()
        return

    summary = result["summary"]
    with st.container(horizontal=True):
        st.metric("Total frames", summary["total_frames"], border=True)
        st.metric("Analyzed frames", summary["analyzed_frames"], border=True)
        st.metric("Frame stride", summary["frame_stride"], border=True)
        st.metric("Total detections", summary["total_detections"], border=True)
        st.metric("Processing time", f"{summary['processing_duration_seconds']:.2f} s", border=True)
        inference_fps = summary["approximate_inference_fps"]
        st.metric(
            "Approx. inference FPS",
            "—" if inference_fps is None else f"{inference_fps:.2f}",
            border=True,
        )
    _show_cv_status(result["condition_summary"])

    st.subheader("Annotated video")
    st.video(result["annotated_video_bytes"], format="video/mp4", width="stretch")
    if result["browser_playback_ready"]:
        st.success(
            "A browser-compatible H.264 playback copy was created and validated.",
            icon=":material/check_circle:",
        )
    else:
        st.warning(
            "Browser-compatible conversion was unavailable. The original OpenCV "
            "output was validated and is downloadable, but browser playback may fail."
        )

    class_rows = [
        {
            "Class": name,
            "Detections": summary["detections_by_class"][name],
            "Frames containing class": summary["frames_containing_each_class"][name],
            "Maximum confidence": summary["maximum_confidence_by_class"][name],
        }
        for name in CLASS_NAMES
    ]
    st.subheader("Class summary")
    st.dataframe(
        class_rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Maximum confidence": st.column_config.NumberColumn(format="%.3f")
        },
    )

    st.subheader("Detection log preview")
    detections = result["detections"]
    if detections.empty:
        st.info("The generated detection log contains no rows.")
    else:
        preview = detections.loc[
            :, ["frame_index", "video_time_seconds", "class_name", "confidence"]
        ].head(100).rename(
            columns={
                "frame_index": "Frame",
                "video_time_seconds": "Time (seconds)",
                "class_name": "Class",
                "confidence": "Confidence",
            }
        )
        st.dataframe(
            preview,
            hide_index=True,
            width="stretch",
            column_config={
                "Time (seconds)": st.column_config.NumberColumn(format="%.3f"),
                "Confidence": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        if len(detections) > len(preview):
            st.caption(f"Showing the first {len(preview)} of {len(detections)} detections.")

    st.subheader("Download generated outputs")
    with st.container(horizontal=True):
        st.download_button(
            "Annotated video",
            data=result["annotated_video_bytes"],
            file_name=result["download_names"]["annotated_video"],
            mime="video/mp4",
            icon=":material/download:",
            on_click="ignore",
        )
        st.download_button(
            "Detection CSV",
            data=result["detection_csv_bytes"],
            file_name=result["download_names"]["detection_csv"],
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
        )
        st.download_button(
            "Frame JSON",
            data=result["frame_json_bytes"],
            file_name=result["download_names"]["frame_json"],
            mime="application/json",
            icon=":material/download:",
            on_click="ignore",
        )
        st.download_button(
            "Summary JSON",
            data=result["summary_json_bytes"],
            file_name=result["download_names"]["summary_json"],
            mime="application/json",
            icon=":material/download:",
            on_click="ignore",
        )
    render_cv_model_information()


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

    if page in {"Overview", "Operational Monitoring"}:
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
        else:
            render_operational_monitoring(frame, selected_index)
    elif page == "Panel Inspection":
        render_panel_inspection()
    else:
        render_video_inspection()

    st.divider()
    st.caption(
        "TECHNOVOLT university project prototype — recorded replay, not live plant "
        "telemetry and not production-ready."
    )


if __name__ == "__main__":
    run_app()
