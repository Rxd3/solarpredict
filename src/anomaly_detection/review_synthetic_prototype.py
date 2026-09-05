"""Review the Day 13 prototype and plan chronological splits without modeling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.anomaly_detection.generate_synthetic_anomalies import (
    LABELS,
    METADATA as DEFAULT_METADATA_PATH,
    OUTPUT as DEFAULT_SYNTHETIC_PATH,
    ROOT,
)
from src.data_processing.engineer_basic_features import ORIGINAL_COLUMNS
from src.data_processing.review_model_features import (
    CORE_UNSUPERVISED_FEATURES,
    DEFAULT_INPUT_PATH as DEFAULT_BASELINE_PATH,
    NEAR_CONSTANT_CV_THRESHOLD,
    NEAR_CONSTANT_DOMINANT_FRACTION,
    load_feature_dataset,
)


DEFAULT_OUTPUT_PATH = ROOT / "outputs/chronological_split_review.json"
DEFAULT_FIGURE_PATH = ROOT / "outputs/figures/operational/day14_split_timeline.png"
TIMEZONE_NOTE = "Not specified by dataset source; timestamps kept timezone-naive."
EXPECTED_SHAPE = (1009, 84)

# Half-open positional intervals [start, end). The reported timestamp at end-1
# is the inclusive final observation; the next split starts exactly one sample later.
SPLIT_OPTIONS = {
    "option_a_larger_training": {
        "display_name": "Option A: larger training period",
        "partitions": {
            "training": (0, 600),
            "calibration": (600, 690),
            "synthetic_evaluation": (690, 1009),
        },
        "advantages": [
            "Uses 600 normal-like rows for detector and scaler fitting.",
            "Training spans daylight, night, and the following morning.",
            "The evaluation region contains afternoon daylight, the second 17:00 transition, and night.",
        ],
        "disadvantages": [
            "Calibration has only 90 rows (3 hours) and is entirely daylight.",
            "A threshold calibrated only at midday may not transfer well to the low-light portion of evaluation.",
            "The regions still come from one complete daily cycle plus partial boundaries.",
        ],
    },
    "option_b_balanced": {
        "display_name": "Option B: approximately balanced thirds",
        "partitions": {
            "training": (0, 336),
            "calibration": (336, 672),
            "synthetic_evaluation": (672, 1009),
        },
        "advantages": [
            "Provides 336, 336, and 337 rows with daylight and low-light observations in every region.",
            "Calibration covers pre-dawn, sunrise, morning, and midday rather than one narrow regime.",
            "The evaluation region includes afternoon, the second recurring 17:00 transition, and night.",
        ],
        "disadvantages": [
            "Only 336 rows remain for model fitting.",
            "Training daylight is limited to the late-afternoon portion of April 27.",
            "Adjacent regions from this short record are not independent daily samples.",
        ],
    },
}

RECURRING_TRANSITIONS = (
    pd.Timestamp("2022-04-27 17:00:00"),
    pd.Timestamp("2022-04-28 17:00:00"),
)


def sha256_file(path: str | Path) -> str:
    """Return a file hash without changing the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_number(value: Any) -> float | int | None:
    """Convert a scalar to a strict JSON number."""
    if value is None or pd.isna(value):
        return None
    converted = float(value)
    if not math.isfinite(converted):
        return None
    return int(converted) if converted.is_integer() else converted


def _timestamp(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat(sep=" ")


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _atomic_json(report: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".json.tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _snapshot_models(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def load_synthetic_prototype(path: str | Path) -> pd.DataFrame:
    """Load the exact Day 13 schema without sorting or modifying source rows."""
    frame = pd.read_csv(path)
    expected = [*load_feature_dataset(DEFAULT_BASELINE_PATH).columns, *LABELS]
    if frame.columns.tolist() != expected:
        raise ValueError("Synthetic prototype must contain 81 baseline columns plus three labels.")
    timestamps = pd.to_datetime(frame["Timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if timestamps.isna().any() or timestamps.dt.tz is not None:
        raise ValueError("Prototype timestamps must be valid and timezone-naive.")
    frame["Timestamp"] = timestamps
    frame["synthetic_anomaly"] = pd.to_numeric(frame["synthetic_anomaly"], errors="raise").astype("int8")
    return frame


def _different(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Compare values while treating paired NaNs as equal."""
    return ~((left == right) | (left.isna() & right.isna()))


def analyze_effect_context(
    baseline: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], pd.Series]:
    """Separate directly modified measurements from causal derived-feature effects."""
    original_measurements = list(ORIGINAL_COLUMNS[1:])
    engineered = list(baseline.columns[14:])
    direct_measurement_change = _different(
        baseline[original_measurements], synthetic[original_measurements]
    ).any(axis=1)
    direct_label = synthetic["synthetic_anomaly"].eq(1)
    engineered_change = _different(
        baseline[engineered], synthetic[engineered]
    ).any(axis=1)
    propagated = ~direct_label & engineered_change
    context = pd.Series("none", index=baseline.index, dtype="string")
    context.loc[propagated] = "propagated"
    context.loc[direct_label] = "direct"

    metadata_tail_rows = {
        int(row)
        for event in metadata["events"]
        for row in event["derived_only_tail_rows"]
    }
    propagated_positions = set(np.flatnonzero(propagated).tolist())
    counts = context.value_counts().reindex(["none", "direct", "propagated"], fill_value=0)
    return {
        "definitions": {
            "direct": "At least one original operational or sensor measurement was deliberately modified; synthetic_anomaly remains 1.",
            "propagated": "Original measurements are untouched, but one or more causal rolling/change features differ because of an earlier synthetic event.",
            "none": "Neither original measurements nor engineered features differ from the baseline.",
        },
        "proposed_future_field": {
            "name": "synthetic_effect_context",
            "allowed_values": ["none", "direct", "propagated"],
            "precedence": "direct takes precedence if both a measurement and derived context are affected",
            "current_prototype_modified": False,
        },
        "row_counts": {key: int(counts[key]) for key in counts.index},
        "metadata_reported_direct_rows": int(metadata["directly_modified_rows"]),
        "metadata_reported_derived_only_rows": int(metadata["derived_only_rows"]),
        "direct_labels_equal_measurement_changes": bool(direct_label.equals(direct_measurement_change)),
        "propagated_positions_equal_metadata_tail_union": propagated_positions == metadata_tail_rows,
        "propagated_first_timestamp": _timestamp(synthetic.loc[propagated, "Timestamp"].min()),
        "propagated_last_timestamp": _timestamp(synthetic.loc[propagated, "Timestamp"].max()),
        "evaluation_note": (
            "A detector using rolling/change features can alert on propagated rows even though the "
            "strict point label is 0. Keep primary point metrics against direct labels and report "
            "alerts in propagated context separately rather than relabeling them after seeing scores."
        ),
    }, context


def _range(frame: pd.DataFrame, rows: list[int], columns: list[str]) -> dict[str, dict[str, float | int | None]]:
    return {
        column: {
            "minimum": _json_number(frame.loc[rows, column].min()),
            "maximum": _json_number(frame.loc[rows, column].max()),
        }
        for column in columns
    }


def review_events(
    baseline: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Review every event and its actual downstream column changes."""
    reviewed = []
    special = {
        "sudden_power_drop": "Short three-row step reduction; power and voltage use the same factor.",
        "sustained_power_reduction": "Five-row persistent reduction followed by a recovery edge.",
        "temporary_near_zero_power": "Creates three expected extra NaNs in safeguarded power relative change.",
        "sudden_power_spike": "Synthetic value is above the maximum observed in the available baseline dataset; equipment capacity is unknown.",
        "gradual_power_degradation": "Twenty-five-row causal ramp followed by an artificial return to the untouched trajectory.",
        "radiation_power_inconsistency": "Radiation changes while power, voltage, and current retain their baseline values.",
        "sensor_excursion": "The seed-42 event selected Relative_Humidity; other original measurements remain unchanged.",
    }
    for event in metadata["events"]:
        rows = [int(row) for row in event["row_positions"]]
        influence = [int(row) for row in event["derived_feature_influence_rows"]]
        affected = list(event["affected_variables"])
        changed_columns = []
        for column in baseline.columns[14:]:
            left = baseline.loc[influence, column]
            right = synthetic.loc[influence, column]
            if (~((left == right) | (left.isna() & right.isna()))).any():
                changed_columns.append(column)
        reviewed.append(
            {
                "anomaly_id": event["anomaly_id"],
                "scenario_type": event["anomaly_type"],
                "start_timestamp": event["start_timestamp"],
                "end_timestamp": event["end_timestamp"],
                "directly_modified_row_count": len(rows),
                "affected_variables": affected,
                "baseline_ranges": _range(baseline, rows, affected),
                "synthetic_ranges": _range(synthetic, rows, affected),
                "downstream_features_affected": bool(changed_columns),
                "downstream_changed_feature_count": len(changed_columns),
                "downstream_changed_features": changed_columns,
                "propagated_row_count": len(event["derived_only_tail_rows"]),
                "propagated_start_timestamp": _timestamp(
                    synthetic.loc[event["derived_only_tail_rows"][0], "Timestamp"]
                    if event["derived_only_tail_rows"]
                    else None
                ),
                "propagated_end_timestamp": _timestamp(
                    synthetic.loc[event["derived_only_tail_rows"][-1], "Timestamp"]
                    if event["derived_only_tail_rows"]
                    else None
                ),
                "special_observation": special[event["anomaly_type"]],
            }
        )
    return reviewed


def review_spike(
    baseline: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    event = next(event for event in metadata["events"] if event["anomaly_type"] == "sudden_power_spike")
    row = int(event["row_positions"][0])
    baseline_value = float(baseline.loc[row, "Power_Generated"])
    synthetic_value = float(synthetic.loc[row, "Power_Generated"])
    actual_fraction = synthetic_value / baseline_value - 1.0
    event_time = baseline.loc[row, "Timestamp"]
    nearby = baseline.loc[
        baseline["Timestamp"].between(
            event_time - pd.Timedelta(minutes=10),
            event_time + pd.Timedelta(minutes=10),
        )
        & baseline.index.to_series().ne(row),
        "Power_Generated",
    ]
    configured = metadata["configuration"]["scenarios"]["sudden_power_spike"]["severity"]["range"]
    historical_maximum = float(baseline["Power_Generated"].max())
    return {
        "timestamp": _timestamp(event_time),
        "baseline_value_w": baseline_value,
        "synthetic_value_w": synthetic_value,
        "sampled_increase_fraction": float(event["severity"]["value"]),
        "recalculated_increase_fraction": actual_fraction,
        "configured_increase_fraction_range": configured,
        "mathematically_consistent_with_configuration": bool(
            configured[0] <= actual_fraction <= configured[1]
            and math.isclose(actual_fraction, float(event["severity"]["value"]), rel_tol=1e-12)
        ),
        "historical_distribution_w": {
            "mean": float(baseline["Power_Generated"].mean()),
            "population_standard_deviation": float(baseline["Power_Generated"].std(ddof=0)),
            "median": float(baseline["Power_Generated"].median()),
            "90th_percentile": float(baseline["Power_Generated"].quantile(0.90)),
            "95th_percentile": float(baseline["Power_Generated"].quantile(0.95)),
            "99th_percentile": float(baseline["Power_Generated"].quantile(0.99)),
            "maximum": historical_maximum,
        },
        "excess_above_historical_maximum_w": synthetic_value - historical_maximum,
        "percentage_above_historical_maximum": (synthetic_value / historical_maximum - 1.0) * 100.0,
        "population_standard_deviations_above_mean": (
            synthetic_value - baseline["Power_Generated"].mean()
        )
        / baseline["Power_Generated"].std(ddof=0),
        "nearby_baseline_window": {
            "half_window_minutes": 10,
            "excluded_event_row": True,
            "row_count": len(nearby),
            "minimum_w": float(nearby.min()),
            "maximum_w": float(nearby.max()),
            "mean_w": float(nearby.mean()),
        },
        "neutral_result": "above the maximum observed in the available baseline dataset.",
        "capacity_interpretation": "Physical equipment capacity is unknown; no capacity claim is made.",
    }


def review_near_zero(
    baseline: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    event = next(
        event
        for event in metadata["events"]
        if event["anomaly_type"] == "temporary_near_zero_power"
    )
    feature = "power_generated_relative_change"
    extra_mask = synthetic[feature].isna() & baseline[feature].notna()
    previous = synthetic["Power_Generated"].shift(1)
    details = []
    for row in np.flatnonzero(extra_mask):
        details.append(
            {
                "row_position": int(row),
                "timestamp": _timestamp(synthetic.loc[row, "Timestamp"]),
                "previous_synthetic_power_w": float(previous.loc[row]),
                "current_synthetic_power_w": float(synthetic.loc[row, "Power_Generated"]),
                "reason": "absolute previous synthetic power is below the 1 W denominator floor",
                "direct_label": int(synthetic.loc[row, "synthetic_anomaly"]),
            }
        )
    numerical = synthetic.select_dtypes(include=[np.number]).to_numpy()
    return {
        "anomaly_id": event["anomaly_id"],
        "relative_change_feature": feature,
        "denominator_floor_w": 1.0,
        "baseline_nan_count": int(baseline[feature].isna().sum()),
        "synthetic_nan_count": int(synthetic[feature].isna().sum()),
        "additional_nan_count": int(extra_mask.sum()),
        "additional_nan_rows": details,
        "infinity_count_in_complete_synthetic_dataset": int(np.isinf(numerical).sum()),
        "follows_existing_safeguard_policy": bool(
            len(details) == 3
            and all(abs(detail["previous_synthetic_power_w"]) < 1.0 for detail in details)
        ),
        "core_model_recommendation": (
            "Keep power and solar relative-change/ratio features outside the first Core model. "
            "The established Core nine features contain none of these ratios."
        ),
    }


def _variability(series: pd.Series) -> tuple[str, float | None, float]:
    nonmissing = series.dropna()
    if nonmissing.nunique() <= 1:
        return "constant", None, 1.0
    dominant = float(nonmissing.value_counts(normalize=True).iloc[0])
    mean_magnitude = abs(float(nonmissing.mean()))
    coefficient = (
        float(nonmissing.std(ddof=0)) / mean_magnitude
        if mean_magnitude > 1e-12
        else math.inf
    )
    if dominant >= NEAR_CONSTANT_DOMINANT_FRACTION or coefficient <= NEAR_CONSTANT_CV_THRESHOLD:
        return "near_constant", _json_number(coefficient), dominant
    return "variable", _json_number(coefficient), dominant


def _partition_assignment(timestamp: pd.Timestamp, partitions: dict[str, dict[str, Any]]) -> str | None:
    for name, part in partitions.items():
        if pd.Timestamp(part["start_timestamp"]) <= timestamp <= pd.Timestamp(part["end_timestamp"]):
            return name
    return None


def analyze_split_options(
    baseline: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Calculate exact split coverage and Core readiness for both plans."""
    options: dict[str, Any] = {}
    for option_id, specification in SPLIT_OPTIONS.items():
        partitions: dict[str, Any] = {}
        assigned_positions: list[int] = []
        for name, (start, end) in specification["partitions"].items():
            part = baseline.iloc[start:end]
            assigned_positions.extend(range(start, end))
            readiness = {}
            for feature in CORE_UNSUPERVISED_FEATURES:
                values = part[feature]
                status, coefficient, dominant = _variability(values)
                readiness[feature] = {
                    "missing_count": int(values.isna().sum()),
                    "finite_count": int(np.isfinite(values).sum()),
                    "population_variance": float(values.var(ddof=0)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "variability_status": status,
                    "coefficient_of_variation": coefficient,
                    "dominant_value_fraction": dominant,
                }
            daylight = part["low_light_context"].eq(0)
            partitions[name] = {
                "start_row_position": start,
                "end_row_position_inclusive": end - 1,
                "start_timestamp": _timestamp(part["Timestamp"].iloc[0]),
                "end_timestamp": _timestamp(part["Timestamp"].iloc[-1]),
                "next_boundary_timestamp": _timestamp(
                    baseline["Timestamp"].iloc[end] if end < len(baseline) else part["Timestamp"].iloc[-1] + pd.Timedelta(minutes=2)
                ),
                "row_count": len(part),
                "sample_support_hours": len(part) * 2.0 / 60.0,
                "inclusive_timestamp_span_hours": (
                    part["Timestamp"].iloc[-1] - part["Timestamp"].iloc[0]
                ).total_seconds()
                / 3600.0,
                "daylight_row_count": int(daylight.sum()),
                "low_light_row_count": int((~daylight).sum()),
                "radiation_above_100_w_m2_row_count": int(part["Solar_Radiation"].gt(100).sum()),
                "first_daylight_timestamp": _timestamp(part.loc[daylight, "Timestamp"].min()),
                "last_daylight_timestamp": _timestamp(part.loc[daylight, "Timestamp"].max()),
                "first_low_light_timestamp": _timestamp(part.loc[~daylight, "Timestamp"].min()),
                "last_low_light_timestamp": _timestamp(part.loc[~daylight, "Timestamp"].max()),
                "core_feature_readiness": readiness,
                "all_core_features_complete_finite_and_variable": bool(
                    all(
                        item["missing_count"] == 0
                        and item["finite_count"] == len(part)
                        and item["variability_status"] == "variable"
                        for item in readiness.values()
                    )
                ),
            }
        counts = np.bincount(assigned_positions, minlength=len(baseline))
        transition_assignments = {
            _timestamp(timestamp): _partition_assignment(timestamp, partitions)
            for timestamp in RECURRING_TRANSITIONS
        }
        event_assignments = {
            event["anomaly_id"]: _partition_assignment(
                pd.Timestamp(event["start_timestamp"]), partitions
            )
            for event in metadata["events"]
        }
        options[option_id] = {
            "display_name": specification["display_name"],
            "partitions": partitions,
            "advantages": specification["advantages"],
            "disadvantages": specification["disadvantages"],
            "coverage_validation": {
                "covers_all_rows_exactly_once": bool(len(counts) == len(baseline) and np.all(counts == 1)),
                "no_row_duplication": bool(counts.max(initial=0) == 1),
                "no_uncovered_rows": bool(counts.min(initial=1) == 1),
                "chronological_non_overlapping_boundaries": bool(
                    all(
                        specification["partitions"][right][0]
                        == specification["partitions"][left][1]
                        for left, right in (("training", "calibration"), ("calibration", "synthetic_evaluation"))
                    )
                ),
            },
            "recurring_transition_partition": transition_assignments,
            "current_validation_prototype_event_partition": event_assignments,
        }
    return options


def save_timeline(
    baseline: pd.DataFrame,
    recommended: dict[str, Any],
    destination: Path,
) -> None:
    """Save one simple chronological-region diagram without model results."""
    colors = {
        "training": "#2f6690",
        "calibration": "#e6a23c",
        "synthetic_evaluation": "#5b9a62",
    }
    fig, axis = plt.subplots(figsize=(12, 3.1))
    for index, (name, part) in enumerate(recommended["partitions"].items()):
        start = pd.Timestamp(part["start_timestamp"])
        exclusive_end = pd.Timestamp(part["next_boundary_timestamp"])
        axis.barh(
            0,
            mdates.date2num(exclusive_end) - mdates.date2num(start),
            left=mdates.date2num(start),
            height=0.42,
            color=colors[name],
            edgecolor="white",
            label=name.replace("_", " ").title(),
        )
        midpoint = start + (exclusive_end - start) / 2
        axis.text(
            midpoint,
            0,
            f"{name.replace('_', ' ').title()}\n{part['row_count']} rows | {part['sample_support_hours']:.1f} h",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
    for number, event in enumerate(RECURRING_TRANSITIONS, start=1):
        axis.axvline(event, color="#8b1e3f", linestyle="--", linewidth=1.2)
        axis.text(event, 0.34, f"17:00 transition {number}", rotation=25, ha="left", va="bottom", fontsize=8)
    axis.set_xlim(
        baseline["Timestamp"].iloc[0],
        baseline["Timestamp"].iloc[-1] + pd.Timedelta(minutes=2),
    )
    axis.set_ylim(-0.55, 0.65)
    axis.set_yticks([])
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axis.set_xlabel("Source timestamp (timezone not specified)")
    axis.set_title("Proposed chronological split — Option B (planning only)")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            suffix=".png.tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        fig.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
        temporary.replace(destination)
    finally:
        plt.close(fig)
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_review(
    baseline: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build the complete evidence report without writing files."""
    if tuple(synthetic.shape) != EXPECTED_SHAPE or len(baseline) != len(synthetic):
        raise ValueError("Expected the verified 1,009-row Day 13 prototype.")
    if not synthetic["Timestamp"].equals(baseline["Timestamp"]):
        raise ValueError("Baseline and prototype timestamps do not match exactly.")
    events = review_events(baseline, synthetic, metadata)
    effect_review, _ = analyze_effect_context(baseline, synthetic, metadata)
    split_options = analyze_split_options(baseline, metadata)
    spike = review_spike(baseline, synthetic, metadata)
    near_zero = review_near_zero(baseline, synthetic, metadata)
    checks = {
        "prototype_shape_is_1009_by_84": tuple(synthetic.shape) == EXPECTED_SHAPE,
        "metadata_seed_is_42": metadata.get("seed") == 42,
        "seven_events_reviewed": len(events) == 7,
        "direct_row_accounting_matches": effect_review["row_counts"]["direct"] == 46,
        "propagated_row_accounting_matches": effect_review["row_counts"]["propagated"] == 203,
        "effect_context_covers_all_rows": sum(effect_review["row_counts"].values()) == len(baseline),
        "direct_labels_equal_measurement_changes": effect_review["direct_labels_equal_measurement_changes"],
        "propagated_rows_match_generator_metadata": effect_review["propagated_positions_equal_metadata_tail_union"],
        "spike_matches_configured_math": spike["mathematically_consistent_with_configuration"],
        "near_zero_nan_safeguard_verified": near_zero["follows_existing_safeguard_policy"],
        "no_infinities": near_zero["infinity_count_in_complete_synthetic_dataset"] == 0,
        "all_split_options_cover_rows_once": all(
            option["coverage_validation"]["covers_all_rows_exactly_once"]
            for option in split_options.values()
        ),
        "all_core_features_ready_in_every_partition": all(
            partition["all_core_features_complete_finite_and_variable"]
            for option in split_options.values()
            for partition in option["partitions"].values()
        ),
        "prototype_validation_still_passes": all(metadata["saved_csv_validation"].values()),
    }
    return {
        "stage": "Day 14 synthetic prototype review and chronological split planning",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone": TIMEZONE_NOTE,
        "dataset_coverage": {
            "row_count": len(baseline),
            "start_timestamp": _timestamp(baseline["Timestamp"].iloc[0]),
            "end_timestamp": _timestamp(baseline["Timestamp"].iloc[-1]),
            "inclusive_timestamp_span_hours": (
                baseline["Timestamp"].iloc[-1] - baseline["Timestamp"].iloc[0]
            ).total_seconds()
            / 3600.0,
            "sample_support_hours": len(baseline) * 2.0 / 60.0,
            "complete_daily_cycles_available": 1,
            "first_and_last_calendar_days_incomplete": True,
        },
        "prototype": {
            "shape": list(synthetic.shape),
            "seed": int(metadata["seed"]),
            "event_count": int(metadata["event_count"]),
            "generator_validation_only": True,
            "automatic_final_test_reuse_allowed": False,
            "policy": (
                "Choose and freeze chronological boundaries first, then regenerate synthetic "
                "events only inside the untouched evaluation region using predeclared seeds."
            ),
        },
        "event_reviews": events,
        "direct_vs_propagated_effects": effect_review,
        "spike_review": spike,
        "near_zero_nan_review": near_zero,
        "split_options": split_options,
        "recommended_split": {
            "option_id": "option_b_balanced",
            "reason": (
                "All three regions contain daylight and low-light rows, and calibration covers "
                "more operating regimes than Option A. This is preferable for a prototype "
                "threshold review even though training has only 336 rows."
            ),
            "limitation": (
                "The regions are adjacent parts of approximately 33.6 hours with only one complete "
                "daily cycle; they are chronological but not independent days."
            ),
            "status": "planning recommendation only; no split datasets created",
        },
        "future_model_data_policy": [
            "Choose and freeze chronological boundaries.",
            "Use untouched baseline rows from the training region.",
            "Reserve a separate untouched baseline calibration region.",
            "Fit the scaler using training data only.",
            "Train the unsupervised detector using training data only.",
            "Select and freeze the alert threshold using calibration data only.",
            "Regenerate synthetic copies using only the evaluation region and new predeclared seeds.",
            "Evaluate the frozen pipeline against direct synthetic labels and report propagated-context alerts separately.",
        ],
        "interpretation_limit": (
            "Synthetic events are controlled prototype cases, not verified photovoltaic faults, "
            "and no detector-performance result is produced in this stage."
        ),
        "validation": {"passed": all(checks.values()), "checks": checks},
    }


def run_review(
    *,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    synthetic_path: Path = DEFAULT_SYNTHETIC_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
    models_path: Path = ROOT / "models",
) -> dict[str, Any]:
    """Write only the Day 14 JSON and timeline while protecting prior artifacts."""
    protected = [
        *sorted(path for path in (ROOT / "data").rglob("*.csv")),
        Path(metadata_path),
    ]
    before = {path.resolve(): sha256_file(path) for path in protected}
    models_before = _snapshot_models(models_path)
    baseline = load_feature_dataset(baseline_path)
    synthetic = load_synthetic_prototype(synthetic_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    report = build_review(baseline, synthetic, metadata)
    save_timeline(
        baseline,
        report["split_options"][report["recommended_split"]["option_id"]],
        Path(figure_path),
    )
    after = {path.resolve(): sha256_file(path) for path in protected}
    models_after = _snapshot_models(models_path)
    report["artifacts"] = {
        "baseline_file": _relative_path(Path(baseline_path)),
        "synthetic_prototype_file": _relative_path(Path(synthetic_path)),
        "generator_metadata_file": _relative_path(Path(metadata_path)),
        "timeline_figure": _relative_path(Path(figure_path)),
    }
    report["protected_source_files"] = {
        _relative_path(path): {
            "sha256_before": before[path.resolve()],
            "sha256_after": after[path.resolve()],
            "unchanged": before[path.resolve()] == after[path.resolve()],
        }
        for path in protected
    }
    report["models_directory"] = {
        "before": models_before,
        "after": models_after,
        "unchanged": models_before == models_after,
        "contains_only_placeholder": set(models_after).issubset({".gitkeep"}),
    }
    preservation_checks = {
        "all_source_csvs_and_generator_metadata_unchanged": all(
            item["unchanged"] for item in report["protected_source_files"].values()
        ),
        "models_directory_unchanged": models_before == models_after,
        "no_trained_model_files_present": set(models_after).issubset({".gitkeep"}),
    }
    report["validation"]["checks"].update(preservation_checks)
    report["validation"]["passed"] = all(report["validation"]["checks"].values())
    if not report["validation"]["passed"]:
        raise RuntimeError("Day 14 review validation failed.")
    _atomic_json(report, Path(output_path))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_review(
            baseline_path=args.baseline,
            synthetic_path=args.synthetic,
            metadata_path=args.metadata,
            output_path=args.output,
            figure_path=args.figure,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Error: {error}")
        return 1
    counts = report["direct_vs_propagated_effects"]["row_counts"]
    print(f"Direct rows: {counts['direct']}; propagated rows: {counts['propagated']}")
    print(f"Recommended split: {report['recommended_split']['option_id']}")
    print(f"Validation passed: {report['validation']['passed']}")
    print("No split, scaler, detector, threshold, or performance result was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
