"""Tests for Day 15 chronological model-dataset preparation."""

import json

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection.create_chronological_split import (
    BASELINE_FILENAMES,
    CORE_FILENAMES,
    DEFAULT_CONFIG_PATH,
    DEFAULT_SOURCE_PATH,
    PARTITION_ORDER,
    SYNTHETIC_LABEL_COLUMNS,
    contextual_coverage,
    core_feature_statistics,
    create_core_matrices,
    create_partitions,
    load_split_config,
    run_preparation,
    sha256_file,
)
from src.data_processing.review_model_features import (
    CORE_UNSUPERVISED_FEATURES,
    load_feature_dataset,
)


@pytest.fixture(scope="module")
def prepared_in_memory():
    config = load_split_config()
    source = load_feature_dataset(DEFAULT_SOURCE_PATH)
    partitions, coverage = create_partitions(source, config)
    matrices = create_core_matrices(partitions)
    return config, source, partitions, matrices, coverage


def test_adopted_boundaries_and_exact_counts(prepared_in_memory):
    config, _, partitions, _, _ = prepared_in_memory
    expected = {
        "training": ("2022-04-27 15:32:00", "2022-04-28 02:42:00", 336),
        "calibration": ("2022-04-28 02:44:00", "2022-04-28 13:54:00", 336),
        "evaluation_baseline": ("2022-04-28 13:56:00", "2022-04-29 01:08:00", 337),
    }
    for name, (start, end, rows) in expected.items():
        assert config["partitions"][name]["start"] == start
        assert config["partitions"][name]["end"] == end
        assert len(partitions[name]) == rows
        assert partitions[name].Timestamp.iloc[0] == pd.Timestamp(start)
        assert partitions[name].Timestamp.iloc[-1] == pd.Timestamp(end)


def test_complete_chronological_coverage_without_overlap(prepared_in_memory):
    _, source, partitions, _, coverage = prepared_in_memory
    assert sum(len(partitions[name]) for name in PARTITION_ORDER) == 1009 == len(source)
    assert coverage["every_source_row_assigned_exactly_once"]
    assert coverage["no_overlap"]
    assert coverage["no_row_loss"]
    assert coverage["no_duplicate_source_positions"]
    assert coverage["no_duplicate_timestamps"]
    assert coverage["all_81_columns_preserved"]
    reconstructed = pd.concat(
        [partitions[name] for name in PARTITION_ORDER], ignore_index=True
    )
    pd.testing.assert_frame_equal(reconstructed, source, check_exact=True)
    assert all(frame.Timestamp.is_monotonic_increasing for frame in partitions.values())


def test_core_matrix_schema_values_and_variance(prepared_in_memory):
    _, _, _, matrices, _ = prepared_in_memory
    expected_columns = ["Timestamp", *CORE_UNSUPERVISED_FEATURES]
    expected_rows = {"training": 336, "calibration": 336, "evaluation_baseline": 337}
    statistics = core_feature_statistics(matrices)

    for name, matrix in matrices.items():
        assert matrix.shape == (expected_rows[name], 10)
        assert matrix.columns.tolist() == expected_columns
        assert not SYNTHETIC_LABEL_COLUMNS.intersection(matrix.columns)
        numerical = matrix[list(CORE_UNSUPERVISED_FEATURES)].to_numpy(dtype=float)
        assert not np.isnan(numerical).any()
        assert np.isfinite(numerical).all()
        assert statistics[name]["all_core_features_usable"]
        assert all(
            entry["positive_variance"]
            for entry in statistics[name]["features"].values()
        )


def test_light_context_and_recurring_transition_locations(prepared_in_memory):
    config, _, partitions, _, _ = prepared_in_memory
    coverage, transitions = contextual_coverage(partitions, config)
    assert {
        name: (values["daylight_count"], values["low_light_count"])
        for name, values in coverage.items()
    } == {
        "training": (100, 236),
        "calibration": (242, 94),
        "evaluation_baseline": (149, 188),
    }
    assert transitions["2022-04-27 17:00:00"]["partition"] == "training"
    assert transitions["2022-04-28 17:00:00"]["partition"] == "evaluation_baseline"
    assert all(not item["labelled_as_anomaly"] for item in transitions.values())


def test_overlapping_configuration_is_rejected(tmp_path):
    config_text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "start: '2022-04-28 02:44:00'", "start: '2022-04-28 02:42:00'"
    )
    invalid = tmp_path / "overlap.yaml"
    invalid.write_text(config_text, encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_split_config(invalid)


def test_published_files_and_report(prepared_in_memory):
    _, source, _, _, _ = prepared_in_memory
    report_path = DEFAULT_SOURCE_PATH.parents[2] / "outputs/model_dataset_preparation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["validation"]["passed"]
    assert report["coverage_validation"]["partition_row_sum"] == len(source)
    assert report["explicitly_not_performed"] == [
        "scaler fitting",
        "synthetic anomaly injection",
        "detector training",
        "threshold calibration",
        "model-performance evaluation",
    ]


def test_run_preserves_source_and_creates_no_model_or_scaler(tmp_path):
    source_hash_before = sha256_file(DEFAULT_SOURCE_PATH)
    model_directory = tmp_path / "models"
    model_directory.mkdir()
    (model_directory / ".gitkeep").write_text("", encoding="utf-8")
    output_directory = tmp_path / "model_ready"
    report_path = tmp_path / "preparation.json"

    report = run_preparation(
        output_directory=output_directory,
        report_path=report_path,
        models_directory=model_directory,
    )

    assert report["validation"]["passed"]
    assert sha256_file(DEFAULT_SOURCE_PATH) == source_hash_before
    assert sorted(path.name for path in output_directory.iterdir()) == sorted(
        [*BASELINE_FILENAMES.values(), *CORE_FILENAMES.values()]
    )
    assert [path.name for path in model_directory.iterdir()] == [".gitkeep"]
    assert not list(tmp_path.rglob("*.joblib"))
    assert not list(tmp_path.rglob("*.pkl"))
