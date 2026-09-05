"""Tests for the controlled Day 18 feature comparison."""

import json

import joblib
import numpy as np

from src.anomaly_detection.compare_detector_features import (
    DEFAULT_COMPARISON_CONFIG,
    DEFAULT_COMPARISON_MODELS,
    DEFAULT_FIGURE_PATH,
    DEFAULT_REPORT_PATH,
    EXPECTED_FEATURE_SETS,
    VARIANT_ORDER,
    load_comparison_config,
    run_comparison,
    select_recommendation,
    sha256_file,
)
from src.anomaly_detection.train_baseline_detector import (
    DEFAULT_CONFIG_PATH as DAY16_CONFIG_PATH,
    DEFAULT_MODELS_DIRECTORY,
    METADATA_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
    load_core_partitions,
    load_model_config,
)


def _report():
    return json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))


def test_exactly_four_predeclared_feature_variants_and_order():
    config = load_comparison_config()
    assert tuple(config["feature_sets"]) == VARIANT_ORDER
    assert len(config["feature_sets"]) == 4
    for variant in VARIANT_ORDER:
        assert tuple(config["feature_sets"][variant]["features"]) == EXPECTED_FEATURE_SETS[variant]
        assert "Timestamp" not in EXPECTED_FEATURE_SETS[variant]


def test_identical_training_rows_and_training_only_scalers():
    day16_config = load_model_config(DAY16_CONFIG_PATH)
    partitions = load_core_partitions(day16_config)
    assert len(partitions["training"]) == 336
    for variant in VARIANT_ORDER[1:]:
        stem = load_comparison_config()["feature_sets"][variant]["artifact_stem"]
        scaler = joblib.load(DEFAULT_COMPARISON_MODELS / f"{stem}_standard_scaler.joblib")
        metadata = json.loads(
            (DEFAULT_COMPARISON_MODELS / f"{stem}_metadata.json").read_text(encoding="utf-8")
        )
        assert int(scaler.n_samples_seen_) == metadata["training_rows"] == 336
        assert scaler.n_features_in_ == len(EXPECTED_FEATURE_SETS[variant])
        assert metadata["training_source_sha256"] == sha256_file(
            DAY16_CONFIG_PATH.parents[1] / day16_config["inputs"]["training"]
        )


def test_identical_hyperparameters_metadata_and_determinism():
    config = load_comparison_config()
    report = _report()
    assert report["validation"]["checks"]["deterministic_results"]
    for variant in VARIANT_ORDER[1:]:
        stem = config["feature_sets"][variant]["artifact_stem"]
        model = joblib.load(DEFAULT_COMPARISON_MODELS / f"{stem}_isolation_forest.joblib")
        metadata = json.loads(
            (DEFAULT_COMPARISON_MODELS / f"{stem}_metadata.json").read_text(encoding="utf-8")
        )
        for parameter, expected in config["isolation_forest_parameters"].items():
            assert model.get_params()[parameter] == expected
            assert metadata["model_parameters"][parameter] == expected
        assert metadata["feature_order"] == list(EXPECTED_FEATURE_SETS[variant])
        assert metadata["timestamp_used_as_feature"] is False
        assert metadata["synthetic_data_used"] is False
        assert metadata["threshold_selected"] is False


def test_recommendation_uses_only_training_and_calibration_evidence():
    config = load_comparison_config()
    report = _report()
    evidence = report["training_calibration_results"]
    recomputed = select_recommendation(evidence, config)
    assert recomputed == report["predeclared_selection"]["recommendation"]
    assert recomputed["recommended_variant"] == "NO_TIME_7"
    assert recomputed["evaluation_baseline_consulted"] is False
    assert report["predeclared_selection"]["locked_before_evaluation_baseline_scoring"]
    assert all(
        item["not_used_for_selection"]
        for item in report["optional_evaluation_baseline_documentation"].values()
    )


def test_no_threshold_labels_or_synthetic_evaluation_created():
    report = _report()
    assert report["validation"]["passed"]
    assert report["evaluation_baseline_excluded_from_selection"]
    assert "threshold selection" in report["explicitly_not_performed"]
    assert "anomaly label creation" in report["explicitly_not_performed"]
    assert "synthetic evaluation generation" in report["explicitly_not_performed"]
    assert all(
        "high-score calibration baseline observation" in row["interpretation"]
        for variant in VARIANT_ORDER
        for row in report["training_calibration_results"][variant]["high_score_calibration_review"]["top_10"]
    )


def test_integration_preserves_day16_and_sources(tmp_path):
    root = DAY16_CONFIG_PATH.parents[1]
    protected = [
        *sorted(path for path in (root / "data").rglob("*") if path.is_file()),
        DEFAULT_MODELS_DIRECTORY / SCALER_FILENAME,
        DEFAULT_MODELS_DIRECTORY / MODEL_FILENAME,
        DEFAULT_MODELS_DIRECTORY / METADATA_FILENAME,
        *sorted((root / "outputs").glob("anomaly_scores_*.csv")),
    ]
    before = {path: sha256_file(path) for path in protected}
    synthetic_before = sorted((root / "data").rglob("*synthetic*"))
    temporary_models = tmp_path / "comparisons"

    report = run_comparison(
        comparison_models_directory=temporary_models,
        report_path=tmp_path / "report.json",
        figure_path=tmp_path / "figure.png",
    )

    assert report["validation"]["passed"]
    assert {path: sha256_file(path) for path in protected} == before
    assert sorted((root / "data").rglob("*synthetic*")) == synthetic_before
    assert len(list(temporary_models.glob("*.joblib"))) == 6
    assert len(list(temporary_models.glob("*_metadata.json"))) == 3
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "figure.png").is_file()


def test_all_variants_keep_nontrivial_training_score_spread():
    report = _report()
    for variant in VARIANT_ORDER:
        result = report["training_calibration_results"][variant]
        assert result["nontrivial_score_spread_guard_passed"]
        stats = result["training_score_summary"]
        assert stats["population_standard_deviation"] >= 0.01
        assert stats["99th_percentile"] - stats["90th_percentile"] >= 0.01
