"""Tests for the read-only Day 11 feature review."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_processing.review_model_features import (
    CORRELATION_THRESHOLD,
    DEFAULT_INPUT_PATH,
    FEATURE_COLUMNS,
    analyze_correlations,
    build_feature_inventory,
    candidate_feature_sets,
    load_feature_dataset,
    review_model_feature_file,
    review_recurring_transitions,
    sha256_file,
    summarize_missingness,
)


@pytest.fixture(scope="module")
def feature_data():
    """Load the real verified Day 10 dataset once for review tests."""
    return load_feature_dataset(DEFAULT_INPUT_PATH)


def test_inventory_contains_every_column_and_actual_missingness(feature_data) -> None:
    inventory = build_feature_inventory(feature_data)

    assert len(inventory) == 81
    assert [item["feature_name"] for item in inventory] == list(FEATURE_COLUMNS)
    assert all(
        item["missing_count"]
        == int(feature_data[str(item["feature_name"])].isna().sum())
        for item in inventory
    )
    assert all(item["unexpected_missing_count"] == 0 for item in inventory)
    assert all(item["documented_missing_values_present"] for item in inventory)

    missingness = summarize_missingness(feature_data, inventory)
    assert missingness["total_missing_cells"] == 1585
    assert missingness["rolling_window_warm_up_nan_cells"] == 546
    assert missingness["first_difference_warm_up_nan_cells"] == 14
    assert missingness["deliberate_denominator_safety_nan_cells"] == 1025
    assert missingness["unexpected_nan_cells"] == 0
    assert missingness["all_missing_values_explained"] is True


def test_candidate_names_exist_and_avoid_obvious_core_redundancy(feature_data) -> None:
    candidates = candidate_feature_sets()
    all_columns = set(feature_data.columns)

    for candidate in candidates.values():
        names = candidate["feature_names"]
        assert candidate["feature_count"] == len(names)
        assert len(names) == len(set(names))
        assert set(names).issubset(all_columns)

    core = candidates["core_unsupervised_features"]["feature_names"]
    assert "Power_Generated" in core
    assert "Array_Voltage" not in core
    assert "electrical_power_product" not in core
    assert "power_consistency_residual" not in core


def test_correlation_review_captures_verified_redundancies(feature_data) -> None:
    analysis = analyze_correlations(feature_data)
    pairs = analysis["highly_correlated_pairs"]

    assert analysis["exploratory_absolute_threshold"] == CORRELATION_THRESHOLD
    assert analysis["highly_correlated_pair_count"] == len(pairs) == 181
    assert all(pair["absolute_correlation"] >= CORRELATION_THRESHOLD for pair in pairs)

    pair_lookup = {
        frozenset((pair["feature_1"], pair["feature_2"])): pair
        for pair in pairs
    }
    power_product = pair_lookup[
        frozenset(("Power_Generated", "electrical_power_product"))
    ]
    assert power_product["pearson_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert all(
        pair["pearson_correlation"] == pytest.approx(1.0, abs=1e-12)
        for pair in analysis["exact_difference_rate_pairs"]
    )


def test_transition_review_uses_complete_neutral_windows(feature_data) -> None:
    review = review_recurring_transitions(feature_data)

    assert len(review["events"]) == 2
    assert all(event["window_row_count"] == 21 for event in review["events"])
    assert all(
        event["neutral_label"] == "recurring power transition"
        for event in review["events"]
    )
    assert review["events"][0]["event_step_changes"]["Power_Generated"] == pytest.approx(
        -125.30496
    )
    assert review["events"][1]["event_step_changes"]["Power_Generated"] == pytest.approx(
        -103.48546
    )
    assert review["aligned_window_comparison"]["Power_Generated"][
        "aligned_pearson_correlation"
    ] > 0.998
    assert all(
        event["previous_observation"]["low_light_context"] == 0
        and event["event_observation"]["low_light_context"] == 0
        for event in review["events"]
    )


def test_end_to_end_review_preserves_data_and_creates_no_model(
    tmp_path: Path,
) -> None:
    source_hash_before = sha256_file(DEFAULT_INPUT_PATH)
    models_directory = tmp_path / "models"
    models_directory.mkdir()
    placeholder = models_directory / ".gitkeep"
    placeholder.write_text("", encoding="utf-8")
    models_before = {path.name: sha256_file(path) for path in models_directory.iterdir()}

    output_directory = tmp_path / "review"
    review_path = output_directory / "model_feature_review.json"
    transition_path = output_directory / "recurring_transition_review.json"
    documentation_path = output_directory / "model_feature_review.md"
    report = review_model_feature_file(
        DEFAULT_INPUT_PATH,
        review_output=review_path,
        transition_output=transition_path,
        documentation_output=documentation_path,
        protected_paths=[DEFAULT_INPUT_PATH],
        models_directory=models_directory,
    )

    assert sha256_file(DEFAULT_INPUT_PATH) == source_hash_before
    assert report["validation"]["passed"] is True
    assert report["validation"]["checks"]["no_source_dataset_modified"] is True
    assert report["validation"]["checks"]["models_directory_unchanged"] is True
    assert report["validation"]["checks"][
        "no_anomaly_label_columns_present_or_introduced"
    ] is True
    assert report["validation"]["checks"]["no_model_was_fitted_or_serialized"] is True
    assert {path.name: sha256_file(path) for path in models_directory.iterdir()} == models_before
    assert {path.suffix for path in output_directory.iterdir()} == {".json", ".md"}
    assert json.loads(review_path.read_text(encoding="utf-8"))["validation"][
        "passed"
    ] is True
    assert len(
        json.loads(transition_path.read_text(encoding="utf-8"))["events"]
    ) == 2
