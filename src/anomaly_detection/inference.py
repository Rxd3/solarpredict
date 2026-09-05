"""Read-only inference API for the frozen operational anomaly prototype.

This module deliberately exposes scoring only.  It never fits a scaler or model,
and verifies the recorded artifact hashes before accepting the frozen detector.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.anomaly_detection.calibrate_anomaly_threshold import (
    FEATURE_ORDER,
    load_frozen_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]
FROZEN_DETECTOR_CONFIG = ROOT / "config/frozen_anomaly_detector.yaml"
FROZEN_DETECTOR_METADATA = ROOT / "models/frozen_anomaly_detector_metadata.json"
FROZEN_THRESHOLD_CONFIG = ROOT / "config/frozen_anomaly_threshold.yaml"
FROZEN_THRESHOLD_METADATA = ROOT / "models/frozen_anomaly_threshold_metadata.json"
EXPECTED_THRESHOLD = 0.7135242760182695
TIMESTAMP_COLUMN = "Timestamp"
OUTPUT_COLUMNS = (
    "anomaly_score",
    "threshold",
    "threshold_margin",
    "status",
)
EVENT_COLUMNS = (
    "event_id",
    "start_timestamp",
    "end_timestamp",
    "row_count",
    "duration_minutes",
    "maximum_anomaly_score",
    "mean_anomaly_score",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


@dataclass(frozen=True)
class FrozenDetector:
    """Verified runtime objects and immutable scoring contract."""

    scaler: Any
    model: Any
    threshold: float
    feature_order: tuple[str, ...]
    verification: dict[str, Any]


def load_frozen_detector() -> FrozenDetector:
    """Load and verify the frozen scaler, model, and threshold without fitting."""

    detector_config, _, scaler, model, detector_verification = load_frozen_artifacts(
        FROZEN_DETECTOR_CONFIG,
        FROZEN_DETECTOR_METADATA,
    )
    threshold_config = yaml.safe_load(FROZEN_THRESHOLD_CONFIG.read_text(encoding="utf-8"))
    threshold_metadata = json.loads(FROZEN_THRESHOLD_METADATA.read_text(encoding="utf-8"))

    if threshold_config.get("status") != "prototype_frozen":
        raise ValueError("Frozen threshold configuration is not marked prototype_frozen.")
    if threshold_metadata.get("status") != "prototype_frozen":
        raise ValueError("Frozen threshold metadata is not marked prototype_frozen.")
    threshold = float(threshold_config["threshold"]["value"])
    if not math.isclose(threshold, EXPECTED_THRESHOLD, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Frozen threshold differs from the Day 20 value.")
    if threshold_config["threshold"].get("comparison_operator") != "strictly_greater_than":
        raise ValueError("Frozen alert comparison must be strictly greater than.")
    if not math.isclose(
        float(threshold_metadata["threshold_value"]), threshold, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("Frozen threshold configuration and metadata disagree.")

    referenced_detector = _resolve(threshold_config["frozen_detector_config"])
    if _sha256(referenced_detector) != threshold_config["frozen_detector_config_sha256"]:
        raise ValueError("Frozen threshold references a changed detector configuration.")
    referenced_metadata = _resolve(threshold_metadata["frozen_detector_metadata"]["path"])
    if _sha256(referenced_metadata) != threshold_metadata["frozen_detector_metadata"]["sha256"]:
        raise ValueError("Frozen threshold references changed detector metadata.")

    feature_order = tuple(detector_config["feature_order"])
    if feature_order != FEATURE_ORDER:
        raise ValueError("Frozen feature order no longer matches the exact NO_TIME_7 contract.")
    verification = {
        **detector_verification,
        "threshold_config": {
            "path": "config/frozen_anomaly_threshold.yaml",
            "sha256": _sha256(FROZEN_THRESHOLD_CONFIG),
        },
        "threshold_metadata": {
            "path": "models/frozen_anomaly_threshold_metadata.json",
            "sha256": _sha256(FROZEN_THRESHOLD_METADATA),
        },
        "threshold": threshold,
        "comparison_operator": "strictly_greater_than",
        "fit_called": False,
    }
    return FrozenDetector(scaler, model, threshold, feature_order, verification)


def _validated_features(frame: pd.DataFrame, detector: FrozenDetector) -> np.ndarray:
    missing = [name for name in detector.feature_order if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required frozen model features: {missing}")
    numeric = frame.loc[:, detector.feature_order].apply(pd.to_numeric, errors="coerce")
    failed = [name for name in detector.feature_order if numeric[name].isna().any()]
    if failed:
        raise ValueError(f"Required features contain non-numeric or missing values: {failed}")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [
            name for index, name in enumerate(detector.feature_order)
            if not np.isfinite(values[:, index]).all()
        ]
        raise ValueError(f"Required features must be finite; invalid columns: {bad}")
    return values


def score_dataframe(
    frame: pd.DataFrame,
    detector: FrozenDetector | None = None,
    *,
    include_context: bool = True,
) -> pd.DataFrame:
    """Score rows with the frozen detector and return dashboard-friendly results.

    Model inputs are always selected in ``FEATURE_ORDER``. Extra input columns are
    ignored except for an optional Timestamp and the optional feature context.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("score_dataframe expects a pandas DataFrame.")
    active = detector or load_frozen_detector()
    values = _validated_features(frame, active)

    prefix: list[str] = []
    if TIMESTAMP_COLUMN in frame.columns:
        prefix.append(TIMESTAMP_COLUMN)
    if include_context:
        prefix.extend(name for name in active.feature_order if name not in prefix)
    result = frame.loc[:, prefix].copy().reset_index(drop=True)
    if len(frame) == 0:
        for column in OUTPUT_COLUMNS:
            result[column] = pd.Series(dtype="object" if column == "status" else "float64")
        return result

    scaled = active.scaler.transform(values)
    scores = -active.model.score_samples(scaled)
    if not np.isfinite(scaled).all() or not np.isfinite(scores).all():
        raise RuntimeError("Frozen artifacts produced non-finite transformed values or scores.")
    result["anomaly_score"] = scores
    result["threshold"] = active.threshold
    result["threshold_margin"] = result["anomaly_score"] - active.threshold
    result["status"] = np.where(scores > active.threshold, "ALERT", "NORMAL")
    return result


def score_observation(
    observation: Mapping[str, Any] | pd.Series,
    detector: FrozenDetector | None = None,
    *,
    include_context: bool = True,
) -> dict[str, Any]:
    """Score one mapping/Series and return one plain result dictionary."""

    if not isinstance(observation, (Mapping, pd.Series)):
        raise TypeError("score_observation expects a mapping or pandas Series.")
    result = score_dataframe(
        pd.DataFrame([dict(observation)]), detector, include_context=include_context
    )
    return result.iloc[0].to_dict()


def group_alert_events(
    scored: pd.DataFrame,
    *,
    timestamp_column: str = TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    """Group ALERT rows whose timestamps are separated by exactly two minutes.

    A one-row event has a two-minute inclusive observation duration, matching the
    threshold-calibration convention already used by the project. No smoothing or
    merging across missing/non-alert timestamps is performed.
    """

    required = [timestamp_column, "anomaly_score", "status"]
    missing = [name for name in required if name not in scored.columns]
    if missing:
        raise ValueError(f"Cannot group alert events; missing columns: {missing}")
    alerts = scored.loc[scored["status"].eq("ALERT"), required].copy()
    if alerts.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    alerts[timestamp_column] = pd.to_datetime(alerts[timestamp_column], errors="coerce")
    if alerts[timestamp_column].isna().any():
        raise ValueError("ALERT rows contain invalid timestamps.")
    if alerts[timestamp_column].dt.tz is not None:
        raise ValueError("Alert-event timestamps must remain timezone-naive.")
    alerts["anomaly_score"] = pd.to_numeric(alerts["anomaly_score"], errors="coerce")
    if not np.isfinite(alerts["anomaly_score"].to_numpy(dtype=float)).all():
        raise ValueError("ALERT rows contain invalid anomaly scores.")
    alerts = alerts.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    group_ids = alerts[timestamp_column].diff().ne(pd.Timedelta(minutes=2)).cumsum()

    events: list[dict[str, Any]] = []
    for number, (_, rows) in enumerate(alerts.groupby(group_ids, sort=False), start=1):
        start = rows[timestamp_column].iloc[0]
        end = rows[timestamp_column].iloc[-1]
        events.append({
            "event_id": f"alert-{number:03d}",
            "start_timestamp": start,
            "end_timestamp": end,
            "row_count": int(len(rows)),
            "duration_minutes": int((end - start).total_seconds() / 60) + 2,
            "maximum_anomaly_score": float(rows["anomaly_score"].max()),
            "mean_anomaly_score": float(rows["anomaly_score"].mean()),
        })
    return pd.DataFrame(events, columns=EVENT_COLUMNS)
