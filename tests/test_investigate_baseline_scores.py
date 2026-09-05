"""Tests for the read-only Day 17 baseline-score investigation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.anomaly_detection.investigate_baseline_scores import (
    CORE_UNSUPERVISED_FEATURES,
    DEFAULT_CONFIG_PATH,
    DEFAULT_FIGURE_PATH,
    DEFAULT_MODELS_DIRECTORY,
    DEFAULT_OUTPUTS_DIRECTORY,
    DEFAULT_REPORT_PATH,
    CONTEXT_THRESHOLD_W_M2,
    context_score_review,
    feature_distribution_review,
    load_saved_scores,
    reproduce_saved_scores,
    run_investigation,
    sha256_file,
)
from src.anomaly_detection.train_baseline_detector import (
    MODEL_FILENAME,
    PARTITION_ORDER,
    SCALER_FILENAME,
    SYNTHETIC_LABEL_COLUMNS,
    load_core_partitions,
    load_model_config,
)


def _inputs():
    config = load_model_config(DEFAULT_CONFIG_PATH)
    partitions = load_core_partitions(config)
    scores = load_saved_scores(partitions)
    scaler = __import__("joblib").load(DEFAULT_MODELS_DIRECTORY / SCALER_FILENAME)
    model = __import__("joblib").load(DEFAULT_MODELS_DIRECTORY / MODEL_FILENAME)
    transformed, reproduction = reproduce_saved_scores(
        partitions, scaler, model, scores
    )
    return partitions, scores, transformed, reproduction


def test_saved_artifacts_reproduce_day16_scores():
    partitions, scores, _, reproduction = _inputs()
    assert all(result["within_csv_precision"] for result in reproduction.values())
    for name in PARTITION_ORDER:
        assert reproduction[name]["row_count"] == len(partitions[name])
        assert scores[name].Timestamp.equals(partitions[name].Timestamp)
        assert reproduction[name]["maximum_absolute_score_difference"] <= 1e-15


def test_all_core_features_and_required_counts_are_reviewed():
    partitions, _, transformed, _ = _inputs()
    review = feature_distribution_review(partitions, transformed)
    assert set(review) == set(CORE_UNSUPERVISED_FEATURES)
    for feature in CORE_UNSUPERVISED_FEATURES:
        for name in ("calibration", "evaluation_baseline"):
            counts = review[feature]["partitions"][name][
                "counts_relative_to_training"
            ]
            assert set(counts) == {
                "below_training_minimum",
                "above_training_maximum",
                "absolute_scaled_value_above_2",
                "absolute_scaled_value_above_3",
                "absolute_scaled_value_above_5",
            }


def test_light_context_calculation_matches_radiation_rule():
    partitions, scores, _, _ = _inputs()
    review = context_score_review(partitions, scores)
    for name in PARTITION_ORDER:
        expected_low = int(
            partitions[name].Solar_Radiation.le(CONTEXT_THRESHOLD_W_M2).sum()
        )
        assert review[name]["low_light"]["count"] == expected_low
        assert review[name]["daylight_like"]["count"] == len(partitions[name]) - expected_low
    assert (review["training"]["daylight_like"]["count"], review["training"]["low_light"]["count"]) == (100, 236)
    assert (review["calibration"]["daylight_like"]["count"], review["calibration"]["low_light"]["count"]) == (242, 94)
    assert (review["evaluation_baseline"]["daylight_like"]["count"], review["evaluation_baseline"]["low_light"]["count"]) == (149, 188)


def test_report_contains_no_labels_threshold_or_comparison_model():
    report = json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
    assert report["validation"]["passed"]
    assert report["threshold_planning_note"]["threshold_selected"] is False
    assert len(report["top_10_high_score_baseline_observations"]) == 3
    for records in report["top_10_high_score_baseline_observations"].values():
        assert len(records) == 10
        assert all("high-score baseline observation" in row["interpretation"] for row in records)
        assert all(not SYNTHETIC_LABEL_COLUMNS.intersection(row) for row in records)
    assert "model fitting or replacement" in report["explicitly_not_performed"]


def test_recurring_transition_rankings_are_recorded_neutrally():
    report = json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
    transitions = report["recurring_transition_diagnostics"]
    assert transitions["2022-04-27 17:00:00"]["highest_score_rank_in_partition"] == 13
    assert transitions["2022-04-28 17:00:00"]["highest_score_rank_in_partition"] == 94
    assert all("not classified as a fault" in item["interpretation"] for item in transitions.values())


def test_integration_never_calls_fit_and_preserves_every_protected_file(
    monkeypatch, tmp_path
):
    protected = [
        *sorted(path for path in (DEFAULT_CONFIG_PATH.parents[1] / "data").rglob("*") if path.is_file()),
        *sorted(path for path in DEFAULT_MODELS_DIRECTORY.rglob("*") if path.is_file()),
        *sorted(DEFAULT_OUTPUTS_DIRECTORY.glob("anomaly_scores_*.csv")),
    ]
    before = {path: sha256_file(path) for path in protected}

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Day 17 must never fit an artifact")

    monkeypatch.setattr(StandardScaler, "fit", forbidden_fit)
    monkeypatch.setattr(IsolationForest, "fit", forbidden_fit)
    report = run_investigation(
        report_path=tmp_path / "investigation.json",
        figure_path=tmp_path / "context.png",
    )

    assert report["validation"]["passed"]
    assert {path: sha256_file(path) for path in protected} == before
    assert (tmp_path / "investigation.json").is_file()
    assert (tmp_path / "context.png").is_file()
    assert not list(tmp_path.rglob("*.joblib"))
    assert not list(tmp_path.rglob("*.csv"))


def test_partition_timestamps_remain_unchanged():
    config = load_model_config(DEFAULT_CONFIG_PATH)
    partitions = load_core_partitions(config)
    scores = load_saved_scores(partitions)
    expected = {
        "training": ("2022-04-27 15:32:00", "2022-04-28 02:42:00", 336),
        "calibration": ("2022-04-28 02:44:00", "2022-04-28 13:54:00", 336),
        "evaluation_baseline": ("2022-04-28 13:56:00", "2022-04-29 01:08:00", 337),
    }
    for name, (start, end, count) in expected.items():
        assert len(scores[name]) == count
        assert scores[name].Timestamp.iloc[0] == pd.Timestamp(start)
        assert scores[name].Timestamp.iloc[-1] == pd.Timestamp(end)
        assert scores[name].Timestamp.equals(partitions[name].Timestamp)
