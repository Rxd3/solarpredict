"""Tests for calibration-only prototype threshold freezing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.anomaly_detection.calibrate_anomaly_threshold import (
    DEFAULT_CALIBRATION_CONFIG,
    DEFAULT_FROZEN_DETECTOR_CONFIG,
    DEFAULT_FROZEN_DETECTOR_METADATA,
    DEFAULT_FROZEN_THRESHOLD_CONFIG,
    DEFAULT_FROZEN_THRESHOLD_METADATA,
    DEFAULT_REPORT_PATH,
    EXPECTED_CANDIDATES,
    FEATURE_ORDER,
    ROOT,
    group_flagged_events,
    load_frozen_artifacts,
    run_threshold_calibration,
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_predeclared_candidates_and_frozen_feature_order() -> None:
    declared = yaml.safe_load(DEFAULT_CALIBRATION_CONFIG.read_text(encoding="utf-8"))
    actual = tuple(
        (
            candidate["id"],
            candidate["strategy"],
            float(candidate.get("percentile", candidate.get("k"))),
        )
        for candidate in declared["candidate_rules"]
    )
    frozen = yaml.safe_load(DEFAULT_FROZEN_DETECTOR_CONFIG.read_text(encoding="utf-8"))
    assert actual == EXPECTED_CANDIDATES
    assert tuple(frozen["feature_order"]) == FEATURE_ORDER
    assert "threshold" not in frozen


def test_frozen_artifacts_load_and_score_without_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fit() must not be called during threshold calibration")

    monkeypatch.setattr(StandardScaler, "fit", forbidden_fit)
    monkeypatch.setattr(IsolationForest, "fit", forbidden_fit)
    frozen, _, scaler, model, verification = load_frozen_artifacts()
    calibration = pd.read_csv(ROOT / "data/model_ready/core_calibration.csv")
    transformed = scaler.transform(calibration.loc[:, FEATURE_ORDER].to_numpy(float))
    scores = -model.score_samples(transformed)
    assert frozen["feature_order"] == list(FEATURE_ORDER)
    assert verification["fit_called"] is False
    assert transformed.shape == (336, 7)
    assert len(scores) == 336


def test_event_grouping_uses_only_consecutive_two_minute_rows() -> None:
    timestamps = pd.to_datetime(
        ["2022-04-28 10:00", "2022-04-28 10:02", "2022-04-28 10:06", "2022-04-28 10:08"]
    )
    events = group_flagged_events(timestamps, expected_gap_minutes=2)
    assert [event["flagged_observation_count"] for event in events] == [2, 2]
    assert [event["duration_minutes"] for event in events] == [4, 4]


def test_selected_threshold_uses_calibration_only_and_precedes_evaluation() -> None:
    report = _json(DEFAULT_REPORT_PATH)
    selected = report["selected_threshold"]
    order = report["workflow_order"]
    assert selected["id"] == "calibration_percentile_97_5"
    assert selected["threshold_value"] == pytest.approx(0.7135242760182695, abs=1e-15)
    assert selected["selection_scope"] == "calibration only"
    assert selected["evaluation_baseline_used"] is False
    assert order.index("threshold_configuration_and_metadata_frozen") < order.index(
        "post_freeze_evaluation_baseline_sanity_check"
    )
    assert report["post_freeze_evaluation_baseline_sanity_check"]["threshold_modified_after_review"] is False


def test_threshold_config_and_metadata_references_are_consistent() -> None:
    threshold = yaml.safe_load(DEFAULT_FROZEN_THRESHOLD_CONFIG.read_text(encoding="utf-8"))
    metadata = _json(DEFAULT_FROZEN_THRESHOLD_METADATA)
    detector_metadata = _json(DEFAULT_FROZEN_DETECTOR_METADATA)
    assert threshold["status"] == metadata["status"] == "prototype_frozen"
    assert threshold["threshold"]["value"] == metadata["threshold_value"]
    assert threshold["threshold"]["strategy"] == metadata["threshold_strategy"]
    assert threshold["selection"]["evaluation_baseline_used"] is False
    assert metadata["trained_model_stored"] is False
    assert metadata["frozen_detector_metadata"]["sha256"] == _sha256(DEFAULT_FROZEN_DETECTOR_METADATA)
    assert metadata["selected_scaler"]["sha256"] == detector_metadata["selected_scaler"]["sha256"]
    assert metadata["selected_model"]["sha256"] == detector_metadata["selected_model"]["sha256"]
    assert metadata["calibration_score_source"]["partition_sha256"] == _sha256(
        ROOT / "data/model_ready/core_calibration.csv"
    )


def test_report_has_no_synthetic_or_performance_use_and_preserves_artifacts() -> None:
    report = _json(DEFAULT_REPORT_PATH)
    checks = report["validation"]["checks"]
    assert report["validation"]["passed"] is True
    assert checks["no_synthetic_labels_or_data_used"] is True
    assert checks["no_performance_metrics_calculated"] is True
    assert checks["no_new_model_training_or_joblib_artifact"] is True
    assert checks["source_datasets_and_frozen_detector_artifacts_unchanged"] is True
    assert all(item["unchanged"] for item in report["protected_artifact_hashes"].values())
    assert "precision/recall/F1 calculation" in report["explicitly_not_performed"]


def test_integration_cannot_fit_or_overwrite_protected_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fit() must not be called")

    monkeypatch.setattr(StandardScaler, "fit", forbidden_fit)
    monkeypatch.setattr(IsolationForest, "fit", forbidden_fit)
    detector_metadata = _json(DEFAULT_FROZEN_DETECTOR_METADATA)
    protected = [
        ROOT / "data/model_ready/core_calibration.csv",
        ROOT / "data/model_ready/core_evaluation_baseline.csv",
        ROOT / detector_metadata["selected_scaler"]["path"],
        ROOT / detector_metadata["selected_model"]["path"],
        DEFAULT_FROZEN_DETECTOR_CONFIG,
        DEFAULT_FROZEN_DETECTOR_METADATA,
    ]
    before = {str(path): _sha256(path) for path in protected}
    report = run_threshold_calibration(
        frozen_threshold_config_path=tmp_path / "threshold.yaml",
        frozen_threshold_metadata_path=tmp_path / "threshold_metadata.json",
        report_path=tmp_path / "report.json",
        figure_path=tmp_path / "figure.png",
    )
    after = {str(path): _sha256(path) for path in protected}
    assert before == after
    assert report["selected_threshold"]["threshold_value"] == pytest.approx(0.7135242760182695)
    assert report["post_freeze_evaluation_baseline_sanity_check"]["baseline_flagged_observation_count"] == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "figure.png",
        "report.json",
        "threshold.yaml",
        "threshold_metadata.json",
    ]
