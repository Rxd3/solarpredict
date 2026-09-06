"""Validated data access for the recorded operational dashboard replay."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.anomaly_detection.inference import (
    FrozenDetector,
    group_alert_events,
    load_frozen_detector,
    score_dataframe,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPERATIONAL_DATA = ROOT / "data/model_ready/core_evaluation_baseline.csv"
DEFAULT_ANOMALY_FEED = ROOT / "outputs/dashboard_anomaly_feed_example.csv"
TIMESTAMP_COLUMN = "Timestamp"
OPERATIONAL_COLUMNS = (
    TIMESTAMP_COLUMN,
    "Power_Generated",
    "Solar_Radiation",
    "Air_Temp",
    "Relative_Humidity",
    "Wind_Speed",
    "rtd_mean",
    "rtd_std",
)
FEED_COLUMNS = (
    TIMESTAMP_COLUMN,
    "Power_Generated",
    "Solar_Radiation",
    "anomaly_score",
    "threshold",
    "threshold_margin",
    "status",
)
NUMERIC_OPERATIONAL_COLUMNS = OPERATIONAL_COLUMNS[1:]
NUMERIC_FEED_COLUMNS = FEED_COLUMNS[1:6]


class DashboardDataError(ValueError):
    """Raised when dashboard input cannot satisfy the replay contract."""


def validate_required_columns(
    frame: pd.DataFrame, required: tuple[str, ...], *, source_name: str
) -> None:
    """Require a known schema while permitting documented extra columns."""

    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DashboardDataError(
            f"{source_name} is missing required columns: {', '.join(missing)}"
        )


def _read_csv(path: str | Path, *, source_name: str) -> pd.DataFrame:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{source_name} was not found: {resolved}")
    try:
        return pd.read_csv(resolved)
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        raise DashboardDataError(f"Could not read {source_name}: {error}") from error


def _parse_timestamp(frame: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    result = frame.copy()
    parsed = pd.to_datetime(
        result[TIMESTAMP_COLUMN], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    if parsed.isna().any():
        count = int(parsed.isna().sum())
        raise DashboardDataError(
            f"{source_name} contains {count} timestamp value(s) that cannot be parsed."
        )
    if parsed.dt.tz is not None:
        raise DashboardDataError(
            f"{source_name} timestamps must remain timezone-naive."
        )
    if parsed.duplicated().any():
        raise DashboardDataError(f"{source_name} contains duplicate timestamps.")
    if not parsed.is_monotonic_increasing:
        raise DashboardDataError(
            f"{source_name} timestamps are not in recorded chronological order."
        )
    result[TIMESTAMP_COLUMN] = parsed
    return result


def _validate_numeric(
    frame: pd.DataFrame, columns: tuple[str, ...], *, source_name: str
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        converted = pd.to_numeric(result[column], errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise DashboardDataError(
                f"{source_name} column {column!r} contains missing or non-numeric values."
            )
        result[column] = converted
    return result


def load_recorded_operational_data(
    path: str | Path = DEFAULT_OPERATIONAL_DATA,
) -> pd.DataFrame:
    """Load the untouched evaluation-baseline measurements for replay."""

    frame = _read_csv(path, source_name="Recorded operational data")
    validate_required_columns(frame, OPERATIONAL_COLUMNS, source_name="Recorded operational data")
    frame = _parse_timestamp(frame, source_name="Recorded operational data")
    return _validate_numeric(
        frame, NUMERIC_OPERATIONAL_COLUMNS, source_name="Recorded operational data"
    )


def load_anomaly_feed(path: str | Path = DEFAULT_ANOMALY_FEED) -> pd.DataFrame:
    """Load the saved output produced by the frozen operational detector."""

    frame = _read_csv(path, source_name="Dashboard anomaly feed")
    validate_required_columns(frame, FEED_COLUMNS, source_name="Dashboard anomaly feed")
    frame = _parse_timestamp(frame, source_name="Dashboard anomaly feed")
    frame = _validate_numeric(frame, NUMERIC_FEED_COLUMNS, source_name="Dashboard anomaly feed")
    statuses = set(frame["status"].dropna().astype(str).unique())
    invalid_statuses = statuses - {"NORMAL", "ALERT"}
    if invalid_statuses or frame["status"].isna().any():
        raise DashboardDataError(
            f"Dashboard anomaly feed contains invalid status values: {sorted(invalid_statuses)}"
        )
    expected = np.where(frame["anomaly_score"] > frame["threshold"], "ALERT", "NORMAL")
    if not np.array_equal(expected, frame["status"].to_numpy(dtype=str)):
        raise DashboardDataError(
            "Dashboard anomaly feed statuses do not match the strict frozen threshold rule."
        )
    if not np.allclose(
        frame["anomaly_score"] - frame["threshold"],
        frame["threshold_margin"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise DashboardDataError("Dashboard anomaly feed threshold margins are inconsistent.")
    return frame


@lru_cache(maxsize=1)
def load_cached_frozen_detector() -> FrozenDetector:
    """Cache the verified frozen artifacts across dashboard reruns."""

    return load_frozen_detector()


def score_recorded_operational_data(
    frame: pd.DataFrame, detector: FrozenDetector | None = None
) -> pd.DataFrame:
    """Optionally score dashboard rows by delegating to the frozen inference API."""

    active = detector if detector is not None else load_cached_frozen_detector()
    return score_dataframe(frame, active, include_context=False)


def build_dashboard_data(
    operational_path: str | Path = DEFAULT_OPERATIONAL_DATA,
    feed_path: str | Path = DEFAULT_ANOMALY_FEED,
    *,
    verify_frozen_scores: bool = True,
) -> pd.DataFrame:
    """Join recorded measurements to their saved, verified anomaly results."""

    operational = load_recorded_operational_data(operational_path)
    feed = load_anomaly_feed(feed_path)
    if len(operational) != len(feed):
        raise DashboardDataError(
            "Operational data and anomaly feed have different row counts: "
            f"{len(operational)} versus {len(feed)}."
        )
    if not operational[TIMESTAMP_COLUMN].equals(feed[TIMESTAMP_COLUMN]):
        raise DashboardDataError(
            "Operational data and anomaly feed timestamps are not identical and ordered."
        )
    for column in ("Power_Generated", "Solar_Radiation"):
        if not np.allclose(
            operational[column], feed[column], rtol=0.0, atol=1e-10, equal_nan=False
        ):
            raise DashboardDataError(
                f"Operational data and anomaly feed disagree for {column}."
            )

    result = operational.loc[:, OPERATIONAL_COLUMNS].copy()
    for column in ("anomaly_score", "threshold", "threshold_margin", "status"):
        result[column] = feed[column].to_numpy(copy=True)

    if verify_frozen_scores:
        rescored = score_recorded_operational_data(result)
        if not np.allclose(
            rescored["anomaly_score"], result["anomaly_score"], rtol=0.0, atol=1e-12
        ):
            raise DashboardDataError(
                "Saved anomaly scores do not match the frozen detector output."
            )
        if not np.allclose(
            rescored["threshold"], result["threshold"], rtol=0.0, atol=1e-15
        ) or not rescored["status"].equals(result["status"]):
            raise DashboardDataError(
                "Saved threshold/status values do not match the frozen detector output."
            )
    return result


def clamp_replay_index(index: int, row_count: int) -> int:
    """Clamp a replay position to valid dataframe bounds."""

    if row_count < 1:
        raise DashboardDataError("Recorded replay contains no readings.")
    return min(max(int(index), 0), row_count - 1)


def move_replay_index(index: int, delta: int, row_count: int) -> int:
    """Move sequentially while retaining safe replay bounds."""

    return clamp_replay_index(int(index) + int(delta), row_count)


def select_current_row(frame: pd.DataFrame, index: int) -> pd.Series:
    """Return a copy of the safely selected replay row."""

    return frame.iloc[clamp_replay_index(index, len(frame))].copy()


def anomaly_status(score: float, threshold: float) -> str:
    """Apply the frozen detector's strict threshold mapping."""

    if not np.isfinite(float(score)) or not np.isfinite(float(threshold)):
        raise DashboardDataError("Anomaly score and threshold must be finite.")
    return "ALERT" if float(score) > float(threshold) else "NORMAL"


def build_event_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the real alert-event table through the inference module helper."""

    return group_alert_events(frame)


def dashboard_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate overview values without inventing alert activity."""

    events = build_event_table(frame)
    return {
        "readings": int(len(frame)),
        "alert_rows": int(frame["status"].eq("ALERT").sum()),
        "alert_events": int(len(events)),
        "start_timestamp": frame[TIMESTAMP_COLUMN].iloc[0] if len(frame) else None,
        "end_timestamp": frame[TIMESTAMP_COLUMN].iloc[-1] if len(frame) else None,
    }
