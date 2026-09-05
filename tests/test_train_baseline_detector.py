"""Tests for Day 16 training-only scaling and initial detector training."""

import json

import joblib
import numpy as np
import pandas as pd

from src.anomaly_detection.train_baseline_detector import (
    CORE_UNSUPERVISED_FEATURES,
    DEFAULT_CONFIG_PATH,
    EXPECTED_PARTITIONS,
    MODEL_FILENAME,
    METADATA_FILENAME,
    PARTITION_ORDER,
    SCALER_FILENAME,
    SCORE_FILENAMES,
    SCALED_FILENAMES,
    SYNTHETIC_LABEL_COLUMNS,
    fit_detector,
    fit_training_scaler,
    load_core_partitions,
    load_model_config,
    run_training,
    score_partitions,
    sha256_file,
    transform_partitions,
)


def _source_paths(config):
    root = DEFAULT_CONFIG_PATH.parents[1]
    return [root / config["inputs"][name] for name in PARTITION_ORDER]


def test_scaler_fits_exactly_training_rows_and_feature_order():
    config = load_model_config()
    partitions = load_core_partitions(config)
    scaler = fit_training_scaler(partitions["training"])

    assert int(scaler.n_samples_seen_) == 336
    assert scaler.n_features_in_ == 9
    np.testing.assert_allclose(
        scaler.mean_,
        partitions["training"][list(CORE_UNSUPERVISED_FEATURES)].mean().to_numpy(),
    )
    assert tuple(config["core_features"]) == tuple(CORE_UNSUPERVISED_FEATURES)


def test_one_training_scaler_transforms_all_partitions_without_timestamp():
    config = load_model_config()
    partitions = load_core_partitions(config)
    scaler = fit_training_scaler(partitions["training"])
    transformed, frames = transform_partitions(partitions, scaler)

    assert transformed["training"].shape == (336, 9)
    assert transformed["calibration"].shape == (336, 9)
    assert transformed["evaluation_baseline"].shape == (337, 9)
    for name in PARTITION_ORDER:
        manual = (
            partitions[name][list(CORE_UNSUPERVISED_FEATURES)].to_numpy()
            - scaler.mean_
        ) / scaler.scale_
        np.testing.assert_allclose(transformed[name], manual, rtol=0, atol=1e-14)
        assert np.isfinite(transformed[name]).all()
        assert frames[name].columns.tolist() == ["Timestamp", *CORE_UNSUPERVISED_FEATURES]


def test_fixed_random_state_produces_deterministic_continuous_scores():
    config = load_model_config()
    partitions = load_core_partitions(config)
    scaler = fit_training_scaler(partitions["training"])
    transformed, _ = transform_partitions(partitions, scaler)
    first = fit_detector(transformed["training"], config["model"]["parameters"])
    second = fit_detector(transformed["training"], config["model"]["parameters"])
    first_scores = score_partitions(partitions, transformed, first)
    second_scores = score_partitions(partitions, transformed, second)

    assert first.n_features_in_ == second.n_features_in_ == 9
    for name in PARTITION_ORDER:
        np.testing.assert_array_equal(
            first_scores[name].anomaly_score, second_scores[name].anomaly_score
        )
        assert first_scores[name].columns.tolist() == ["Timestamp", "anomaly_score"]
        assert not SYNTHETIC_LABEL_COLUMNS.intersection(first_scores[name].columns)


def test_existing_score_files_preserve_partition_timestamps():
    config = load_model_config()
    partitions = load_core_partitions(config)
    root = DEFAULT_CONFIG_PATH.parents[1]
    for name in PARTITION_ORDER:
        scores = pd.read_csv(root / "outputs" / SCORE_FILENAMES[name])
        timestamps = pd.to_datetime(scores.Timestamp, format="%Y-%m-%d %H:%M:%S")
        assert scores.columns.tolist() == ["Timestamp", "anomaly_score"]
        assert timestamps.equals(partitions[name].Timestamp)
        assert len(scores) == EXPECTED_PARTITIONS[name][0]
        assert np.isfinite(scores.anomaly_score).all()


def test_metadata_matches_saved_scaler_model_and_training_source():
    root = DEFAULT_CONFIG_PATH.parents[1]
    metadata_path = root / "models" / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scaler = joblib.load(root / "models" / SCALER_FILENAME)
    model = joblib.load(root / "models" / MODEL_FILENAME)

    assert metadata["training_row_count"] == int(scaler.n_samples_seen_) == 336
    assert metadata["feature_order"] == list(CORE_UNSUPERVISED_FEATURES)
    assert scaler.n_features_in_ == model.n_features_in_ == 9
    assert metadata["training_source_sha256"] == sha256_file(
        root / metadata["training_source_path"]
    )
    assert metadata["scaler_sha256"] == sha256_file(root / metadata["scaler_path"])
    assert metadata["model_sha256"] == sha256_file(root / metadata["model_path"])
    assert metadata["final_threshold_calibrated"] is False
    assert metadata["synthetic_labels_used"] is False


def test_integration_preserves_sources_and_creates_only_expected_artifacts(tmp_path):
    config = load_model_config()
    source_paths = _source_paths(config)
    before = {path: sha256_file(path) for path in source_paths}
    root = DEFAULT_CONFIG_PATH.parents[1]
    synthetic_before = sorted(root.glob("data/**/*synthetic*"))
    models = tmp_path / "models"
    models.mkdir()
    (models / ".gitkeep").write_text("", encoding="utf-8")

    report = run_training(
        models_directory=models,
        scaled_directory=tmp_path / "scaled",
        outputs_directory=tmp_path / "outputs",
        summary_path=tmp_path / "summary.json",
        figure_path=tmp_path / "scores.png",
    )

    assert report["validation"]["passed"]
    assert {path: sha256_file(path) for path in source_paths} == before
    assert sorted(root.glob("data/**/*synthetic*")) == synthetic_before
    assert sorted(path.name for path in models.iterdir()) == sorted(
        [".gitkeep", SCALER_FILENAME, MODEL_FILENAME, METADATA_FILENAME]
    )
    assert sorted(path.name for path in (tmp_path / "scaled").iterdir()) == sorted(
        SCALED_FILENAMES.values()
    )
    assert sorted(path.name for path in (tmp_path / "outputs").iterdir()) == sorted(
        SCORE_FILENAMES.values()
    )
    assert not any("synthetic" in path.name for path in tmp_path.rglob("*.csv"))


def test_summary_contains_no_threshold_or_performance_result():
    root = DEFAULT_CONFIG_PATH.parents[1]
    summary = json.loads(
        (root / "outputs/initial_detector_summary.json").read_text(encoding="utf-8")
    )
    assert summary["validation"]["passed"]
    assert summary["isolation_forest"]["final_threshold_selected"] is False
    assert "final threshold calibration" in summary["explicitly_not_performed"]
    assert not {"precision", "recall", "f1"}.intersection(summary)
