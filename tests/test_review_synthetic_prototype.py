"""Regression tests for the Day 14 review and split-planning stage."""

import json

import pytest

from src.anomaly_detection.generate_synthetic_anomalies import METADATA, OUTPUT, ROOT
from src.anomaly_detection.review_synthetic_prototype import (
    CORE_UNSUPERVISED_FEATURES,
    analyze_effect_context,
    analyze_split_options,
    build_review,
    load_synthetic_prototype,
    run_review,
    sha256_file,
)
from src.data_processing.review_model_features import (
    DEFAULT_INPUT_PATH,
    load_feature_dataset,
)


@pytest.fixture(scope="module")
def review_inputs():
    baseline = load_feature_dataset(DEFAULT_INPUT_PATH)
    synthetic = load_synthetic_prototype(OUTPUT)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    return baseline, synthetic, metadata


@pytest.fixture(scope="module")
def review(review_inputs):
    return build_review(*review_inputs)


def test_direct_and_propagated_effect_accounting(review_inputs):
    baseline, synthetic, metadata = review_inputs
    effect_review, context = analyze_effect_context(baseline, synthetic, metadata)

    assert effect_review["row_counts"] == {
        "none": 760,
        "direct": 46,
        "propagated": 203,
    }
    assert len(context) == len(baseline) == 1009
    assert effect_review["direct_labels_equal_measurement_changes"]
    assert effect_review["propagated_positions_equal_metadata_tail_union"]
    assert (context.eq("direct") == synthetic["synthetic_anomaly"].eq(1)).all()


def test_all_events_and_special_numeric_reviews(review):
    assert len(review["event_reviews"]) == 7
    assert {event["scenario_type"] for event in review["event_reviews"]} == {
        "sudden_power_drop",
        "sustained_power_reduction",
        "temporary_near_zero_power",
        "sudden_power_spike",
        "gradual_power_degradation",
        "radiation_power_inconsistency",
        "sensor_excursion",
    }
    assert all(event["downstream_features_affected"] for event in review["event_reviews"])
    assert all(event["propagated_row_count"] == 29 for event in review["event_reviews"])

    spike = review["spike_review"]
    assert spike["mathematically_consistent_with_configuration"]
    assert spike["synthetic_value_w"] > spike["historical_distribution_w"]["maximum"]
    assert spike["neutral_result"] == (
        "above the maximum observed in the available baseline dataset."
    )

    near_zero = review["near_zero_nan_review"]
    assert near_zero["additional_nan_count"] == 3
    assert near_zero["infinity_count_in_complete_synthetic_dataset"] == 0
    assert near_zero["follows_existing_safeguard_policy"]
    assert not any(
        "ratio" in feature or "relative_change" in feature
        for feature in CORE_UNSUPERVISED_FEATURES
    )


@pytest.mark.parametrize(
    ("option_id", "expected_counts"),
    [
        ("option_a_larger_training", [600, 90, 319]),
        ("option_b_balanced", [336, 336, 337]),
    ],
)
def test_chronological_split_coverage_and_non_overlap(
    review_inputs, option_id, expected_counts
):
    baseline, _, metadata = review_inputs
    option = analyze_split_options(baseline, metadata)[option_id]
    partitions = list(option["partitions"].values())

    assert [part["row_count"] for part in partitions] == expected_counts
    assert sum(expected_counts) == len(baseline)
    assert partitions[0]["start_row_position"] == 0
    assert partitions[-1]["end_row_position_inclusive"] == len(baseline) - 1
    for left, right in zip(partitions, partitions[1:]):
        assert left["end_row_position_inclusive"] + 1 == right["start_row_position"]
        assert left["end_timestamp"] < right["start_timestamp"]
    assert all(option["coverage_validation"].values())


def test_core_feature_readiness_in_every_proposed_partition(review):
    assert len(CORE_UNSUPERVISED_FEATURES) == 9
    for option in review["split_options"].values():
        for partition in option["partitions"].values():
            assert partition["all_core_features_complete_finite_and_variable"]
            assert set(partition["core_feature_readiness"]) == set(
                CORE_UNSUPERVISED_FEATURES
            )
            for status in partition["core_feature_readiness"].values():
                assert status["missing_count"] == 0
                assert status["finite_count"] == partition["row_count"]
                assert status["population_variance"] > 0
                assert status["variability_status"] == "variable"


def test_run_review_preserves_sources_and_creates_no_model(tmp_path):
    protected = [*sorted((ROOT / "data").rglob("*.csv")), METADATA]
    hashes_before = {path: sha256_file(path) for path in protected}
    temporary_models = tmp_path / "models"
    temporary_models.mkdir()
    (temporary_models / ".gitkeep").write_text("", encoding="utf-8")
    output = tmp_path / "review.json"
    figure = tmp_path / "timeline.png"

    report = run_review(
        baseline_path=DEFAULT_INPUT_PATH,
        synthetic_path=OUTPUT,
        metadata_path=METADATA,
        output_path=output,
        figure_path=figure,
        models_path=temporary_models,
    )

    assert report["validation"]["passed"]
    assert output.is_file() and figure.is_file()
    assert {path: sha256_file(path) for path in protected} == hashes_before
    assert [path.name for path in temporary_models.iterdir()] == [".gitkeep"]
    assert not list(tmp_path.rglob("*.pkl"))
    assert not list(tmp_path.rglob("*.joblib"))


def test_review_rejects_wrong_prototype_shape(review_inputs):
    baseline, synthetic, metadata = review_inputs
    with pytest.raises(ValueError, match="1,009-row"):
        build_review(baseline, synthetic.iloc[:-1], metadata)
