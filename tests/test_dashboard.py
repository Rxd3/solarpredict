"""Data-contract and application smoke tests for the Streamlit dashboard."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard.data_service import (
    DashboardDataError,
    FEED_COLUMNS,
    anomaly_status,
    build_dashboard_data,
    build_event_table,
    clamp_replay_index,
    dashboard_summary,
    load_anomaly_feed,
    move_replay_index,
    select_current_row,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_HASHES = {
    "models/comparisons/no_time_standard_scaler.joblib":
        "b8b04db85a3bae4a0187b8ede66e859e811b7ee85b995976c9dc808b67b9aefb",
    "models/comparisons/no_time_isolation_forest.joblib":
        "5eb151856bb1d0776587140bb2eb201ba450f40293d540a90db44b3599ef963c",
    "config/frozen_anomaly_detector.yaml":
        "825cdc02e000352611de77bc19fce1e550f0dd8c7bd8a791e5cc2f1da79dd8b3",
    "config/frozen_anomaly_threshold.yaml":
        "019d251482160e64da89e33c7276c6aad864696bfa907a087a7d7c300b0139b7",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def dashboard_data() -> pd.DataFrame:
    return build_dashboard_data(verify_frozen_scores=True)


def test_feed_loading_timestamp_parsing_and_zero_alerts(
    dashboard_data: pd.DataFrame,
) -> None:
    feed = load_anomaly_feed()
    assert tuple(feed.columns) == FEED_COLUMNS
    assert feed["Timestamp"].dtype.kind == "M"
    assert len(feed) == len(dashboard_data) == 337
    assert dashboard_data["Timestamp"].is_monotonic_increasing
    assert dashboard_data["status"].eq("NORMAL").all()
    summary = dashboard_summary(dashboard_data)
    assert summary["alert_rows"] == summary["alert_events"] == 0
    assert build_event_table(dashboard_data).empty


def test_replay_selection_and_bounds(dashboard_data: pd.DataFrame) -> None:
    assert clamp_replay_index(-10, len(dashboard_data)) == 0
    assert clamp_replay_index(999, len(dashboard_data)) == 336
    assert move_replay_index(0, -1, len(dashboard_data)) == 0
    assert move_replay_index(0, 1, len(dashboard_data)) == 1
    assert move_replay_index(336, 1, len(dashboard_data)) == 336
    assert select_current_row(dashboard_data, 999).equals(dashboard_data.iloc[336])
    with pytest.raises(DashboardDataError, match="no readings"):
        clamp_replay_index(0, 0)


def test_strict_anomaly_status_and_event_generation() -> None:
    threshold = 0.7
    assert anomaly_status(threshold, threshold) == "NORMAL"
    assert anomaly_status(threshold + 1e-6, threshold) == "ALERT"
    scored = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                ["2022-01-01 00:00", "2022-01-01 00:02", "2022-01-01 00:06"]
            ),
            "anomaly_score": [0.8, 0.9, 0.85],
            "status": ["ALERT", "ALERT", "ALERT"],
        }
    )
    events = build_event_table(scored)
    assert events.event_id.tolist() == ["alert-001", "alert-002"]
    assert events.row_count.tolist() == [2, 1]
    assert events.duration_minutes.tolist() == [4, 2]


@pytest.mark.parametrize("problem", ["missing_column", "bad_timestamp"])
def test_feed_validation_errors_are_readable(tmp_path: Path, problem: str) -> None:
    feed = pd.read_csv(ROOT / "outputs/dashboard_anomaly_feed_example.csv")
    if problem == "missing_column":
        feed = feed.drop(columns="anomaly_score")
        match = "missing required columns: anomaly_score"
    else:
        feed.loc[0, "Timestamp"] = "not-a-timestamp"
        match = "cannot be parsed"
    path = tmp_path / "bad_feed.csv"
    feed.to_csv(path, index=False)
    with pytest.raises(DashboardDataError, match=match):
        load_anomaly_feed(path)


def test_streamlit_pages_replay_controls_and_charts_render() -> None:
    app = AppTest.from_file(str(ROOT / "src/dashboard/app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Solar Panel Monitoring and Fault Detection System"
    assert app.header[0].value == "System Overview"
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Monitored Readings"] == "337"
    assert metrics["Detected Alerts"] == "0"
    assert metrics["System Status"] == "NORMAL"

    app.radio[0].set_value("Operational Monitoring").run()
    assert not app.exception
    assert app.header[0].value == "Operational Monitoring"
    assert len(app.get("plotly_chart")) == 4
    assert app.slider[0].value == 0
    next(button for button in app.button if button.label == "Next Reading").click()
    app.run()
    assert app.slider[0].value == 1
    app.slider[0].set_value(336).run()
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Status"] == "NORMAL"
    assert metrics["ALERT readings"] == "0"
    assert metrics["Grouped alert events"] == "0"

    app.radio[0].set_value("Panel Inspection").run()
    assert not app.exception
    assert app.header[0].value == "Solar Panel Image Inspection"
    assert not app.get("file_uploader")
    app.radio[0].set_value("Drone Video Inspection").run()
    assert not app.exception
    assert app.header[0].value == "Drone Video Inspection"
    assert not app.get("file_uploader")


def test_dashboard_has_no_training_calls_or_displayed_day_numbers() -> None:
    sources = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "src/dashboard/app.py",
            "src/dashboard/data_service.py",
            "src/dashboard/components.py",
            "src/dashboard/charts.py",
        )
    )
    assert ".train(" not in sources and ".fit(" not in sources
    assert "Day 27" not in sources and "Day 28" not in sources
    assert {relative: _sha256(ROOT / relative) for relative in FROZEN_HASHES} == FROZEN_HASHES
