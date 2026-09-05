"""Tests for Day 19 seed stability and prototype freeze artifacts."""

import json

import joblib
import yaml

from src.anomaly_detection.review_model_stability import (
    COMPACT_MODEL,
    COMPACT_SCALER,
    DECLARED_SEEDS,
    DEFAULT_FROZEN_CONFIG,
    DEFAULT_FROZEN_METADATA,
    DEFAULT_REPORT_PATH,
    DEFAULT_STABILITY_CONFIG,
    FEATURE_ORDER,
    NO_TIME_METADATA,
    NO_TIME_MODEL,
    NO_TIME_SCALER,
    load_stability_config,
    run_stability_review,
    sha256_file,
)


def _report():
    return json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))


def test_exactly_five_declared_seeds_and_identical_feature_order():
    config = load_stability_config()
    report = _report()
    assert tuple(config["review_seeds"]) == DECLARED_SEEDS == (7, 21, 42, 84, 123)
    assert set(report["seed_score_results"]) == {str(seed) for seed in DECLARED_SEEDS}
    for result in report["seed_score_results"].values():
        assert result["feature_order"] == list(FEATURE_ORDER)
        assert result["training_rows"] == 336
        assert result["scaler_fit_scope"] == "training only"


def test_parameters_are_identical_except_declared_random_state():
    config = load_stability_config()
    report = _report()
    base = config["isolation_forest_parameters"]
    for seed in DECLARED_SEEDS:
        parameters = report["seed_score_results"][str(seed)]["model_parameters"]
        assert parameters["random_state"] == seed
        assert {
            key: value for key, value in parameters.items() if key != "random_state"
        } == {key: value for key, value in base.items() if key != "random_state"}


def test_seed_rank_agreement_and_extreme_rtd_accounting():
    report = _report()
    rank = report["calibration_rank_stability"]
    assert rank["minimum_spearman"] >= 0.95
    assert rank["mean_top_10_overlap_fraction"] >= 0.50
    assert rank["mean_top_20_overlap_fraction"] == 1.0
    rtd = report["extreme_rtd_std_review"]
    assert rtd["count"] == 16
    assert len(rtd["records"]) == 16
    assert rtd["no_time_7_count_in_calibration_top_10_percent"] == 16
    assert rtd["no_time_7_count_in_top_10_observations"] == 8
    assert all(abs(row["training_scaled_rtd_std"]) > 5 for row in rtd["records"])


def test_freeze_configuration_matches_selected_artifacts_without_threshold():
    config = yaml.safe_load(DEFAULT_FROZEN_CONFIG.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_FROZEN_METADATA.read_text(encoding="utf-8"))
    assert config["status"] == "prototype_frozen"
    assert config["feature_order"] == list(FEATURE_ORDER)
    assert config["production_prototype_seed"] == 42
    assert "threshold" not in config
    assert metadata["selected_variant"] == "NO_TIME_7"
    assert metadata["new_model_created"] is False
    assert metadata["selected_scaler"]["sha256"] == sha256_file(NO_TIME_SCALER)
    assert metadata["selected_model"]["sha256"] == sha256_file(NO_TIME_MODEL)
    assert metadata["selected_day18_metadata"]["sha256"] == sha256_file(NO_TIME_METADATA)


def test_frozen_references_load_and_have_exact_feature_dimensions():
    metadata = json.loads(DEFAULT_FROZEN_METADATA.read_text(encoding="utf-8"))
    root = DEFAULT_FROZEN_CONFIG.parents[1]
    scaler = joblib.load(root / metadata["selected_scaler"]["path"])
    model = joblib.load(root / metadata["selected_model"]["path"])
    assert scaler.n_features_in_ == model.n_features_in_ == len(FEATURE_ORDER) == 7
    assert int(scaler.n_samples_seen_) == 336


def test_freeze_criteria_pass_without_evaluation_or_threshold():
    report = _report()
    assert report["freeze_decision"]["passed"]
    assert report["freeze_decision"]["decision"] == "freeze_NO_TIME_7_as_prototype"
    assert report["freeze_decision"]["evaluation_baseline_used"] is False
    assert all(report["freeze_decision"]["predeclared_checks"].values())
    assert "threshold selection" in report["explicitly_not_performed"]
    assert "synthetic evaluation generation" in report["explicitly_not_performed"]
    assert "new persisted seed-specific models" in report["explicitly_not_performed"]


def test_integration_preserves_day18_models_and_sources(tmp_path):
    root = DEFAULT_STABILITY_CONFIG.parents[1]
    protected = [
        *sorted(path for path in (root / "data").rglob("*") if path.is_file()),
        NO_TIME_SCALER,
        NO_TIME_MODEL,
        NO_TIME_METADATA,
        COMPACT_SCALER,
        COMPACT_MODEL,
        root / "outputs/model_feature_comparison.json",
        *sorted((root / "outputs").glob("anomaly_scores_*.csv")),
    ]
    before = {path: sha256_file(path) for path in protected}
    synthetic_before = sorted((root / "data").rglob("*synthetic*"))

    report = run_stability_review(
        frozen_config_path=tmp_path / "frozen.yaml",
        frozen_metadata_path=tmp_path / "frozen_metadata.json",
        report_path=tmp_path / "review.json",
        figure_path=tmp_path / "stability.png",
    )

    assert report["validation"]["passed"]
    assert {path: sha256_file(path) for path in protected} == before
    assert sorted((root / "data").rglob("*synthetic*")) == synthetic_before
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "frozen.yaml", "frozen_metadata.json", "review.json", "stability.png"
    ]
    assert not list(tmp_path.rglob("*.joblib"))
