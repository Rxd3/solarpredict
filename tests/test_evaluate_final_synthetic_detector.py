"""Tests for the frozen Day 22 synthetic-data evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.anomaly_detection.calibrate_anomaly_threshold import FEATURE_ORDER
from src.anomaly_detection.evaluate_final_synthetic_detector import (
    BASELINE_INPUT,
    DEFAULT_COMPARISON,
    DEFAULT_PREDICTIONS,
    DEFAULT_REPORT,
    EXPECTED_THRESHOLD,
    FROZEN_THRESHOLD_CONFIG,
    FROZEN_THRESHOLD_METADATA,
    PREDICTION_COLUMNS,
    SYNTHETIC_INPUT,
    confusion_and_metrics,
    run_evaluation,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def evaluation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    synthetic = pd.read_csv(SYNTHETIC_INPUT, parse_dates=["Timestamp"])
    predictions = pd.read_csv(DEFAULT_PREDICTIONS, parse_dates=["Timestamp"])
    comparison = pd.read_csv(DEFAULT_COMPARISON, parse_dates=["Timestamp"])
    report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    return synthetic, predictions, comparison, report


def test_frozen_hashes_threshold_and_feature_order(evaluation) -> None:
    synthetic, _, _, report = evaluation
    frozen_threshold = yaml.safe_load(FROZEN_THRESHOLD_CONFIG.read_text(encoding="utf-8"))
    metadata = json.loads(FROZEN_THRESHOLD_METADATA.read_text(encoding="utf-8"))
    verification = report["frozen_configuration"]
    assert tuple(verification["feature_order"]) == FEATURE_ORDER
    assert tuple(synthetic.columns[1:8]) == FEATURE_ORDER
    assert verification["threshold"] == pytest.approx(EXPECTED_THRESHOLD, abs=1e-15)
    assert frozen_threshold["threshold"]["value"] == metadata["threshold_value"]
    assert verification["threshold_verification"]["threshold_config"]["sha256"] == _sha256(
        FROZEN_THRESHOLD_CONFIG
    )
    assert verification["threshold_verification"]["threshold_metadata"]["sha256"] == _sha256(
        FROZEN_THRESHOLD_METADATA
    )


def test_exact_timestamps_and_paired_baseline(evaluation) -> None:
    synthetic, predictions, comparison, report = evaluation
    baseline = pd.read_csv(BASELINE_INPUT, parse_dates=["Timestamp"])
    assert len(synthetic) == len(predictions) == len(comparison) == len(baseline) == 337
    assert synthetic.Timestamp.equals(predictions.Timestamp)
    assert synthetic.Timestamp.equals(comparison.Timestamp)
    assert synthetic.Timestamp.equals(baseline.Timestamp)
    assert report["validation"]["checks"]["paired_baseline_timestamps_match"]


def test_predictions_use_strict_greater_than_rule(evaluation) -> None:
    _, predictions, _, report = evaluation
    expected = (predictions.anomaly_score.to_numpy() > EXPECTED_THRESHOLD).astype(int)
    assert predictions.columns.tolist() == list(PREDICTION_COLUMNS)
    assert np.array_equal(predictions.predicted_anomaly.to_numpy(dtype=int), expected)
    assert predictions.predicted_anomaly.sum() == 0
    assert report["validation"]["checks"]["strict_greater_than_prediction_rule"]


def test_confusion_matrix_and_metrics_are_arithmetically_correct(evaluation) -> None:
    _, predictions, _, report = evaluation
    confusion, metrics = confusion_and_metrics(
        predictions.synthetic_anomaly.to_numpy(dtype=int),
        predictions.predicted_anomaly.to_numpy(dtype=int),
    )
    assert confusion == {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 316,
        "false_negative": 21,
    }
    assert confusion == report["confusion_matrix"]
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0
    assert metrics["false_positive_rate"] == 0
    assert metrics == report["point_metrics"]
    assert sum(confusion.values()) == 337


def test_propagated_context_remains_reference_negative(evaluation) -> None:
    _, predictions, comparison, report = evaluation
    propagated = predictions.synthetic_effect_context.eq("propagated")
    assert propagated.sum() == 64
    assert predictions.loc[propagated, "synthetic_anomaly"].eq(0).all()
    assert predictions.loc[propagated, "predicted_anomaly"].eq(0).all()
    assert np.array_equal(
        comparison.loc[propagated, "synthetic_score"].to_numpy(),
        comparison.loc[propagated, "baseline_score"].to_numpy(),
    )
    assert report["predicted_alert_context"] == {"direct": 0, "propagated": 0, "none": 0}


def test_strict_event_results_and_alert_grouping(evaluation) -> None:
    _, _, _, report = evaluation
    event_metrics = report["event_metrics"]
    assert event_metrics["total_events"] == 3
    assert event_metrics["detected_events"] == 0
    assert event_metrics["missed_events"] == 3
    assert event_metrics["event_level_recall"] == 0
    assert all(not event["strict_event_detected"] for event in report["per_event_results"])
    assert all(event["context_aware_response"]["response_category"] == "not_at_all" for event in report["per_event_results"])
    assert report["detector_alert_events"]["count"] == 0
    assert report["detector_alert_events"]["events"] == []


def test_protected_transition_and_artifacts_remain_unchanged(evaluation) -> None:
    _, _, _, report = evaluation
    protected = report["protected_transition_review"]
    assert protected["no_synthetic_effects_present"]
    assert protected["synthetic_and_baseline_scores_match_exactly"]
    assert protected["flagged_row_count"] == 0
    assert protected["recurring_17_00_transition"] == {
        "anomaly_score": pytest.approx(0.5815159631484387),
        "baseline_score": pytest.approx(0.5815159631484387),
        "score_delta": 0.0,
        "flagged": False,
        "interpretation": "recurring operational transition; not a synthetic event or verified fault",
    }
    checks = report["validation"]["checks"]
    assert checks["threshold_unchanged"]
    assert checks["synthetic_dataset_unchanged"]
    assert checks["protected_sources_unchanged"]
    assert checks["no_retraining_or_new_model_artifact"]
    assert checks["no_synthetic_regeneration"]


def test_run_does_not_fit_or_modify_frozen_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fit() is forbidden during frozen evaluation")

    monkeypatch.setattr(StandardScaler, "fit", forbidden_fit)
    monkeypatch.setattr(IsolationForest, "fit", forbidden_fit)
    protected = [SYNTHETIC_INPUT, FROZEN_THRESHOLD_CONFIG, FROZEN_THRESHOLD_METADATA]
    before = {str(path): _sha256(path) for path in protected}
    report = run_evaluation(
        predictions_path=tmp_path / "predictions.csv",
        comparison_path=tmp_path / "comparison.csv",
        report_path=tmp_path / "report.json",
        figure_path=tmp_path / "figure.png",
    )
    after = {str(path): _sha256(path) for path in protected}
    assert before == after
    assert report["validation"]["passed"]
    assert report["event_metrics"]["detected_events"] == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "comparison.csv", "figure.png", "predictions.csv", "report.json"
    ]
