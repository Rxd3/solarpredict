"""Regression tests for the final unscored synthetic evaluation dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import IsolationForest

from src.anomaly_detection.generate_final_synthetic_evaluation import (
    BASELINE_EVALUATION,
    DEFAULT_CONFIG,
    DEFAULT_FIGURE,
    DEFAULT_METADATA,
    DEFAULT_NO_TIME_OUTPUT,
    DEFAULT_OUTPUT,
    EFFECT_COLUMN,
    EVALUATION_END,
    EVALUATION_START,
    FROZEN_FILES,
    NO_TIME_COLUMNS,
    PROTECTED_END,
    PROTECTED_START,
    ROOT,
    SCENARIOS,
    SEED,
    run_final_generation,
)
from src.data_processing.engineer_basic_features import ORIGINAL_COLUMNS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    baseline = pd.read_csv(BASELINE_EVALUATION, parse_dates=["Timestamp"])
    synthetic = pd.read_csv(DEFAULT_OUTPUT, parse_dates=["Timestamp"])
    metadata = json.loads(DEFAULT_METADATA.read_text(encoding="utf-8"))
    return baseline, synthetic, metadata


def test_fixed_seed_and_all_scenarios_attempted(artifacts) -> None:
    _, _, metadata = artifacts
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    attempted = {event["type"] for event in metadata["events"]}.union(
        skipped["scenario"] for skipped in metadata["skipped_scenarios"]
    )
    assert config["random_seed"] == metadata["seed"] == SEED == 2026
    assert attempted == set(SCENARIOS)
    assert len(metadata["events"]) == 3
    assert len(metadata["skipped_scenarios"]) == 4


def test_exact_evaluation_timestamps_and_no_outside_direct_modification(artifacts) -> None:
    baseline, synthetic, metadata = artifacts
    assert len(synthetic) == len(baseline) == 337
    assert synthetic.Timestamp.equals(baseline.Timestamp)
    assert synthetic.Timestamp.iloc[0] == EVALUATION_START
    assert synthetic.Timestamp.iloc[-1] == EVALUATION_END
    for event in metadata["events"]:
        assert EVALUATION_START <= pd.Timestamp(event["start_timestamp"])
        assert pd.Timestamp(event["end_timestamp"]) <= EVALUATION_END
    checks = metadata["validation"]["checks"]
    assert checks["no_direct_modification_before_evaluation"]
    assert checks["no_direct_modification_after_evaluation"]
    assert checks["training_and_calibration_all_features_unchanged"]


def test_direct_labels_match_changes_and_events_do_not_overlap(artifacts) -> None:
    baseline, synthetic, metadata = artifacts
    differences = []
    for column in ORIGINAL_COLUMNS[1:]:
        left = baseline[column].to_numpy(dtype=float)
        right = synthetic[column].to_numpy(dtype=float)
        differences.append(~np.isclose(left, right, rtol=1e-12, atol=1e-10, equal_nan=True))
    changed = np.column_stack(differences).any(axis=1).astype(int)
    assert np.array_equal(changed, synthetic.synthetic_anomaly.to_numpy(dtype=int))
    occupied: set[str] = set()
    for event in metadata["events"]:
        timestamps = set(event["direct_timestamps"])
        assert not occupied.intersection(timestamps)
        occupied.update(timestamps)
        selected = synthetic.Timestamp.isin(pd.to_datetime(list(timestamps)))
        assert synthetic.loc[selected, "synthetic_anomaly_type"].eq(event["type"]).all()
        assert synthetic.loc[selected, "synthetic_anomaly_id"].eq(event["id"]).all()


def test_propagated_rows_are_not_direct_labels(artifacts) -> None:
    baseline, synthetic, metadata = artifacts
    propagated = synthetic[EFFECT_COLUMN].eq("propagated")
    assert propagated.sum() == metadata["distribution_review"]["propagated_rows"] == 64
    assert synthetic.loc[propagated, "synthetic_anomaly"].eq(0).all()
    for column in ORIGINAL_COLUMNS[1:]:
        assert np.allclose(
            synthetic.loc[propagated, column], baseline.loc[propagated, column],
            rtol=1e-12, atol=1e-10, equal_nan=True,
        )


def test_protected_transition_and_distribution(artifacts) -> None:
    baseline, synthetic, metadata = artifacts
    protected = synthetic.Timestamp.between(PROTECTED_START, PROTECTED_END)
    assert protected.sum() == 21
    for column in ORIGINAL_COLUMNS[1:]:
        assert np.allclose(
            synthetic.loc[protected, column], baseline.loc[protected, column],
            rtol=1e-12, atol=1e-10, equal_nan=True,
        )
    assert synthetic.loc[protected, "synthetic_anomaly"].eq(0).all()
    assert synthetic.loc[protected, EFFECT_COLUMN].eq("none").all()
    assert metadata["distribution_review"] == {
        "successful_scenarios": [
            "gradual_power_degradation", "sustained_power_reduction", "sensor_excursion"
        ],
        "direct_rows": 21,
        "propagated_rows": 64,
        "none_rows": 252,
        "anomaly_types_represented": [
            "gradual_power_degradation", "sensor_excursion", "sustained_power_reduction"
        ],
        "direct_daylight_like_rows": 20,
        "direct_low_light_rows": 1,
    }


def test_recomputation_ranges_nans_and_unscaled_model_matrix(artifacts) -> None:
    _, synthetic, metadata = artifacts
    checks = metadata["validation"]["checks"]
    assert checks["full_history_used_for_feature_recomputation"]
    assert checks["dependent_features_recomputed_causally"]
    assert checks["electrical_relationship_policy_respected"]
    assert checks["humidity_within_zero_to_one_hundred"]
    assert checks["no_infinities"]
    assert all(metadata["numerical_consistency"]["scenario_ranges_respected"].values())
    assert metadata["numerical_consistency"]["expected_nan_counts_by_column"] == {
        "solar_radiation_relative_change": 187,
        "solar_deviation_ratio_30m": 180,
    }
    assert not np.isinf(synthetic.select_dtypes(include="number").to_numpy()).any()
    no_time = pd.read_csv(DEFAULT_NO_TIME_OUTPUT)
    assert no_time.columns.tolist() == list(NO_TIME_COLUMNS)
    assert "anomaly_score" not in no_time.columns
    assert "prediction" not in no_time.columns


def test_sources_and_frozen_artifacts_are_unchanged_and_no_scoring(artifacts) -> None:
    _, _, metadata = artifacts
    checks = metadata["validation"]["checks"]
    assert checks["protected_sources_unchanged"]
    assert checks["frozen_scaler_model_and_threshold_hashes_unchanged"]
    assert checks["no_detector_scores_or_predictions_created"]
    assert all(item["unchanged"] for item in metadata["protected_source_hashes"].values())
    assert "Isolation Forest fitting or scoring" in metadata["explicitly_not_performed"]


def test_integration_does_not_score_detector_or_change_frozen_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_score(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("The detector must not score the Day 21 dataset")

    monkeypatch.setattr(IsolationForest, "score_samples", forbidden_score)
    before = {str(path): _sha256(path) for path in FROZEN_FILES}
    report = run_final_generation(
        output_path=tmp_path / "final.csv",
        no_time_output_path=tmp_path / "no_time.csv",
        metadata_path=tmp_path / "metadata.json",
        figure_path=tmp_path / "figure.png",
    )
    after = {str(path): _sha256(path) for path in FROZEN_FILES}
    assert before == after
    assert report["validation"]["passed"]
    assert report["event_count"] == 3
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "figure.png", "final.csv", "metadata.json", "no_time.csv"
    ]
