"""Validate the disabled Day 12 design contract without generating data."""

from pathlib import Path

import pytest
import yaml

from src.data_processing.engineer_basic_features import ORIGINAL_COLUMNS
from src.data_processing.review_model_features import CORE_UNSUPERVISED_FEATURES

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/synthetic_anomaly_scenarios.yaml"
EXPECTED_SCENARIOS = (
    "sudden_power_drop",
    "sustained_power_reduction",
    "temporary_near_zero_power",
    "sudden_power_spike",
    "gradual_power_degradation",
    "radiation_power_inconsistency",
    "sensor_excursion",
)


@pytest.fixture(scope="module")
def design():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_design_remains_disabled_and_matches_verified_core(design):
    assert design["schema_version"] == 1
    assert design["stage"] == "design_only"
    assert design["enabled"] is False
    assert isinstance(design["random_seed"], int)
    assert design["sampling_interval_seconds"] == 120
    assert tuple(design["core_features"]) == CORE_UNSUPERVISED_FEATURES
    assert tuple(design["scenarios"]) == EXPECTED_SCENARIOS
    assert all(s["enabled"] is False for s in design["scenarios"].values())


@pytest.mark.parametrize("scenario_id", EXPECTED_SCENARIOS)
def test_scenario_parameters_are_well_formed_and_use_real_measurements(design, scenario_id):
    scenario = design["scenarios"][scenario_id]
    assert set(scenario["affected_variables"]).issubset(ORIGINAL_COLUMNS)
    assert "Timestamp" not in scenario["affected_variables"]
    low, high = scenario["duration_samples"]
    assert type(low) is int and type(high) is int
    assert 1 <= low <= high
    assert scenario["duration_minutes"] == [low * 2, high * 2]
    assert scenario["context_rule"] in {"power_daylight", "finite_sensor_any_light"}
    assert scenario["transformation"]
    severity = scenario["severity"]
    if "range" in severity:
        minimum, maximum = severity["range"]
        assert 0 <= minimum < maximum < 1
    else:
        assert scenario_id == "sensor_excursion"
        assert severity["direction_choices"] == [-1, 1]
        assert set(severity["magnitude_ranges"]) == set(scenario["affected_variables"])
        for specification in severity["magnitude_ranges"].values():
            minimum, maximum = specification["range"]
            assert 0 < minimum <= maximum
            assert specification["unit"]
    if "Power_Generated" in scenario["affected_variables"]:
        assert "Array_Voltage" in scenario["affected_variables"]
        assert scenario["context_rule"] == "power_daylight"


def test_context_rules_disallow_overlap_and_protect_split_boundaries(design):
    context = design["context"]
    assert context["eligible_split"] == "evaluation_only"
    assert context["eligibility_reference"] == "untouched_baseline"
    assert context["overlap_allowed"] is False
    assert context["require_entire_window_eligible"] is True
    assert context["require_finite_inputs"] is True
    assert context["require_contiguous_sampling"] is True
    assert context["minimum_gap_minutes"] >= 60
    assert 0 < context["max_modified_row_fraction_per_copy"] <= 0.05
    assert context["power_event_min_radiation_w_m2"] > context["low_light_threshold_w_m2"]
    assert context["power_event_min_power_w"] > 0
    assert context["insufficient_eligible_windows"] == "skip_and_record_reason"


def test_label_contract_and_evaluation_separation(design):
    labels = design["labels"]
    assert set(labels) == {
        "synthetic_anomaly", "synthetic_anomaly_type", "synthetic_anomaly_id"
    }
    assert labels["synthetic_anomaly"]["baseline_value"] == 0
    assert labels["synthetic_anomaly"]["modified_value"] == 1
    assert labels["synthetic_anomaly_type"]["baseline_value"] == "none"
    assert labels["synthetic_anomaly_id"]["baseline_value"] is None
    assert not set(labels).intersection(design["core_features"])
    evaluation = design["evaluation"]
    assert evaluation["split_boundaries"] is None
    assert evaluation["scaler_fit_scope"] == "baseline_training_only"
    assert evaluation["detector_fit_scope"] == "baseline_training_only"
    assert evaluation["threshold_selection_scope"] == "baseline_calibration_only"
    assert evaluation["final_test_labels_used_for_tuning"] is False
    assert evaluation["paired_untouched_control"] is True


def test_reproducibility_metadata_is_sufficient_to_identify_a_run(design):
    assert design["reproducibility"]["proposed_rng"] == "numpy.random.Generator(PCG64)"
    required = set(design["reproducibility"]["metadata_required"])
    assert {
        "baseline_sha256", "config_sha256", "generator_version",
        "dependency_versions", "seed_and_rng", "split_boundaries", "event_id",
        "sampled_parameters", "original_and_modified_values", "output_sha256",
        "skipped_event_reasons", "source_row_positions_and_timestamps",
    }.issubset(required)


def test_all_scenarios_have_documentation(design):
    document = (ROOT / "docs/synthetic_anomaly_design.md").read_text(encoding="utf-8")
    for scenario_id in design["scenarios"]:
        assert f"| `{scenario_id}`" in document
