"""Generate the final synthetic evaluation copy without scoring the detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.anomaly_detection.generate_synthetic_anomalies import (
    LABELS,
    SCENARIOS,
    VERSION as GENERATOR_VERSION,
    generate_prototype,
    load_config,
)
from src.data_processing.engineer_basic_features import (
    ENGINEERED_FEATURES,
    ORIGINAL_COLUMNS,
    add_basic_features,
)
from src.data_processing.engineer_change_features import CHANGE_FEATURES, add_change_features
from src.data_processing.engineer_rolling_features import ROLLING_FEATURE_SPECS, add_rolling_features
from src.data_processing.review_model_features import FEATURE_COLUMNS, load_feature_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config/final_synthetic_evaluation.yaml"
SCENARIO_SOURCE_CONFIG = ROOT / "config/synthetic_anomaly_scenarios.yaml"
DEFAULT_OUTPUT = ROOT / "data/model_ready/final_synthetic_evaluation.csv"
DEFAULT_NO_TIME_OUTPUT = ROOT / "data/model_ready/final_synthetic_evaluation_no_time_7.csv"
DEFAULT_METADATA = ROOT / "outputs/final_synthetic_evaluation_metadata.json"
DEFAULT_FIGURE = ROOT / "outputs/figures/operational/day21_final_synthetic_evaluation.png"
BASELINE_EVALUATION = ROOT / "data/model_ready/baseline_evaluation.csv"
FROZEN_FILES = (
    ROOT / "config/frozen_anomaly_detector.yaml",
    ROOT / "config/frozen_anomaly_threshold.yaml",
    ROOT / "models/frozen_anomaly_detector_metadata.json",
    ROOT / "models/frozen_anomaly_threshold_metadata.json",
)
EVALUATION_START = pd.Timestamp("2022-04-28 13:56:00")
EVALUATION_END = pd.Timestamp("2022-04-29 01:08:00")
PROTECTED_START = pd.Timestamp("2022-04-28 16:40:00")
PROTECTED_END = pd.Timestamp("2022-04-28 17:20:00")
SEED = 2026
EFFECT_COLUMN = "synthetic_effect_context"
NO_TIME_COLUMNS = (
    "Timestamp",
    "Power_Generated",
    "Solar_Radiation",
    "Air_Temp",
    "Relative_Humidity",
    "Wind_Speed",
    "rtd_mean",
    "rtd_std",
    *LABELS,
    EFFECT_COLUMN,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _timestamp(value: Any) -> str:
    return pd.Timestamp(value).isoformat(sep=" ")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".csv.tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False, lineterminator="\n", na_rep="", float_format="%.17g")
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".json.tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _snapshot(paths: list[Path]) -> dict[str, str]:
    return {
        _relative(path): sha256_file(path)
        for path in sorted(set(path.resolve() for path in paths))
        if path.is_file()
    }


def _series_equal(left: pd.Series, right: pd.Series) -> np.ndarray:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return np.isclose(left_values, right_values, rtol=1e-12, atol=1e-10, equal_nan=True)


def _frame_numeric_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    rtol: float = 1e-12,
    atol: float = 1e-10,
) -> bool:
    return all(
        bool(
            np.isclose(
                pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ).all()
        )
        for column in columns
    )


def _changed_mask(
    baseline: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str] | tuple[str, ...]
) -> pd.DataFrame:
    return pd.DataFrame(
        {column: ~_series_equal(baseline[column], synthetic[column]) for column in columns},
        index=baseline.index,
    )


def load_final_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(path)
    if config.get("configuration_version") != "1.0.0" or config.get("random_seed") != SEED:
        raise ValueError("Final evaluation must use configuration 1.0.0 and seed 2026.")
    if not config.get("enabled") or not all(config["scenarios"][name]["enabled"] for name in SCENARIOS):
        raise ValueError("All seven final-evaluation scenarios must be enabled for an attempted placement.")
    period = config.get("evaluation_period", {})
    placement = config["context"].get("placement_window", {})
    expected = (_timestamp(EVALUATION_START), _timestamp(EVALUATION_END))
    if (str(period.get("start")), str(period.get("end"))) != expected:
        raise ValueError("Final evaluation period differs from the frozen split.")
    if (str(placement.get("start")), str(placement.get("end"))) != expected:
        raise ValueError("Placement window must exactly equal the frozen evaluation period.")
    if int(period.get("expected_rows", 0)) != 337:
        raise ValueError("Final evaluation must contain 337 rows.")
    exclusions = config["context"]["known_transition_exclusions"]
    if exclusions["half_window_minutes"] != 20 or exclusions["timestamps"] != ["2022-04-28 17:00:00"]:
        raise ValueError("The exact 16:40-17:20 recurring-transition window must be protected.")

    source = load_config(SCENARIO_SOURCE_CONFIG)
    comparison_keys = (
        "category", "affected_variables", "severity", "duration_samples",
        "duration_minutes", "transformation", "context_rule",
    )
    for name in SCENARIOS:
        for key in comparison_keys:
            if config["scenarios"][name].get(key) != source["scenarios"][name].get(key):
                raise ValueError(f"Final scenario {name}.{key} differs from the Day 12 definition.")
        if config["scenarios"][name].get("variable_selection") != source["scenarios"][name].get("variable_selection"):
            raise ValueError(f"Final scenario {name}.variable_selection changed.")
    return config


def build_effect_context(
    baseline: pd.DataFrame, synthetic: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    original_changes = _changed_mask(baseline, synthetic, ORIGINAL_COLUMNS[1:])
    derived_changes = _changed_mask(baseline, synthetic, FEATURE_COLUMNS[len(ORIGINAL_COLUMNS):])
    direct = original_changes.any(axis=1)
    propagated = derived_changes.any(axis=1) & ~direct
    context = pd.Series("none", index=synthetic.index, dtype="object")
    context.loc[propagated] = "propagated"
    context.loc[direct] = "direct"
    return context, derived_changes


def recompute_from_measurements(measurements: pd.DataFrame) -> pd.DataFrame:
    basic = add_basic_features(measurements.loc[:, list(ORIGINAL_COLUMNS)].copy())
    rolling = add_rolling_features(basic)
    return add_change_features(rolling)


def _summary(frame: pd.DataFrame, rows: list[int], variables: list[str]) -> dict[str, Any]:
    return {
        variable: {
            "minimum": float(frame.loc[rows, variable].min()),
            "maximum": float(frame.loc[rows, variable].max()),
            "mean": float(frame.loc[rows, variable].mean()),
        }
        for variable in variables
    }


def _scenario_range_check(event: dict[str, Any], config: dict[str, Any]) -> bool:
    spec = config["scenarios"][event["anomaly_type"]]
    samples = int(event["affected_row_count"])
    duration_ok = spec["duration_samples"][0] <= samples <= spec["duration_samples"][1]
    value = float(event["severity"]["value"])
    if event["anomaly_type"] == "sensor_excursion":
        variable = event["affected_variables"][0]
        low, high = spec["severity"]["magnitude_ranges"][variable]["range"]
    else:
        low, high = spec["severity"]["range"]
    return duration_ok and float(low) <= value <= float(high)


def enrich_events(
    events: list[dict[str, Any]],
    baseline: pd.DataFrame,
    synthetic: pd.DataFrame,
    effect_context: pd.Series,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation_start_position = int(baseline.index[baseline.Timestamp.eq(EVALUATION_START)][0])
    enriched: list[dict[str, Any]] = []
    outside_ranges: list[dict[str, Any]] = []
    for event in events:
        rows = [int(row) for row in event["row_positions"]]
        variables = list(event["affected_variables"])
        propagated_positions = [
            int(row) for row in event["derived_only_tail_rows"]
            if EVALUATION_START <= baseline.Timestamp.iloc[row] <= EVALUATION_END
            and effect_context.iloc[row] == "propagated"
        ]
        for variable in variables:
            baseline_min = float(baseline[variable].min())
            baseline_max = float(baseline[variable].max())
            modified_min = float(synthetic.loc[rows, variable].min())
            modified_max = float(synthetic.loc[rows, variable].max())
            if modified_min < baseline_min or modified_max > baseline_max:
                outside_ranges.append({
                    "event_id": event["anomaly_id"],
                    "scenario": event["anomaly_type"],
                    "measurement": variable,
                    "modified_minimum": modified_min,
                    "modified_maximum": modified_max,
                    "baseline_minimum": baseline_min,
                    "baseline_maximum": baseline_max,
                    "description": "outside the range observed in the available baseline dataset",
                })
        baseline_rows = baseline.loc[rows]
        context_conditions = {
            "all_required_measurements_finite": bool(
                np.isfinite(baseline_rows.loc[:, ORIGINAL_COLUMNS[1:]].to_numpy(dtype=float)).all()
            ),
            "continuous_two_minute_sampling": bool(
                len(rows) == 1
                or baseline_rows.Timestamp.diff().dropna().eq(pd.Timedelta(minutes=2)).all()
            ),
            "minimum_baseline_solar_radiation": float(baseline_rows.Solar_Radiation.min()),
            "minimum_baseline_power_generated": float(baseline_rows.Power_Generated.min()),
            "power_context_required": event["anomaly_type"] != "sensor_excursion",
            "protected_transition_excluded": not bool(
                baseline_rows.Timestamp.between(PROTECTED_START, PROTECTED_END).any()
            ),
        }
        enriched.append({
            "id": event["anomaly_id"],
            "type": event["anomaly_type"],
            "seed": int(event["random_seed"]),
            "start_timestamp": event["start_timestamp"],
            "end_timestamp": event["end_timestamp"],
            "duration_minutes": int(event["duration_minutes"]),
            "row_count": int(event["affected_row_count"]),
            "affected_measurements": variables,
            "severity": event["severity"],
            "original_summary": _summary(baseline, rows, variables),
            "modified_summary": _summary(synthetic, rows, variables),
            "context_conditions": context_conditions,
            "direct_full_sequence_row_positions": rows,
            "direct_evaluation_row_indices": [row - evaluation_start_position for row in rows],
            "direct_timestamps": [_timestamp(baseline.Timestamp.iloc[row]) for row in rows],
            "propagated_effect_row_count": len(propagated_positions),
            "propagated_effect_timestamps": [
                _timestamp(baseline.Timestamp.iloc[row]) for row in propagated_positions
            ],
            "configured_ranges_respected": _scenario_range_check(event, config),
        })
    return enriched, outside_ranges


def save_figure(
    baseline: pd.DataFrame,
    synthetic: pd.DataFrame,
    events: list[dict[str, Any]],
    destination: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(13, 5))
    axis.plot(
        baseline.Timestamp, baseline.Power_Generated,
        color="#2f6690", linewidth=1.2, label="Untouched evaluation baseline",
    )
    axis.plot(
        synthetic.Timestamp, synthetic.Power_Generated,
        color="#d97706", linewidth=1.1, linestyle="--", label="Final synthetic evaluation",
    )
    first_region = True
    for event in events:
        if "Power_Generated" not in event["affected_measurements"]:
            continue
        axis.axvspan(
            pd.Timestamp(event["start_timestamp"]),
            pd.Timestamp(event["end_timestamp"]) + pd.Timedelta(minutes=2),
            color="#dc2626", alpha=0.15,
            label="Direct power-event region" if first_region else None,
        )
        first_region = False
    axis.axvspan(
        PROTECTED_START, PROTECTED_END, color="#64748b", alpha=0.08,
        label="Protected recurring-transition window",
    )
    axis.set_title("Day 21 final synthetic evaluation: generated power validation")
    axis.set_xlabel("Source timestamp (timezone not specified)")
    axis.set_ylabel("Power_Generated (W)")
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png.tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        fig.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
        temporary.replace(destination)
    finally:
        plt.close(fig)
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_final_generation(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_path: Path = DEFAULT_OUTPUT,
    no_time_output_path: Path = DEFAULT_NO_TIME_OUTPUT,
    metadata_path: Path = DEFAULT_METADATA,
    figure_path: Path = DEFAULT_FIGURE,
) -> dict[str, Any]:
    config = load_final_config(Path(config_path))
    baseline_path = _resolve(config["baseline_path"])
    output_path = Path(output_path)
    no_time_output_path = Path(no_time_output_path)
    destinations = {output_path.resolve(), no_time_output_path.resolve(), Path(metadata_path).resolve(), Path(figure_path).resolve()}
    if len(destinations) != 4 or baseline_path.resolve() in destinations:
        raise ValueError("Final outputs must be distinct and must not overwrite their source.")

    protected_paths = [
        path for path in (ROOT / "data").rglob("*")
        if path.is_file() and path.resolve() not in {output_path.resolve(), no_time_output_path.resolve()}
    ]
    protected_paths += [path for path in (ROOT / "models").rglob("*") if path.is_file()]
    protected_paths += [
        *FROZEN_FILES,
        Path(config_path),
        SCENARIO_SOURCE_CONFIG,
        ROOT / "src/data_processing/engineer_basic_features.py",
        ROOT / "src/data_processing/engineer_rolling_features.py",
        ROOT / "src/data_processing/engineer_change_features.py",
    ]
    protected_before = _snapshot(protected_paths)
    frozen_before = _snapshot(list(FROZEN_FILES))

    baseline = load_feature_dataset(baseline_path)
    synthetic, generator_report = generate_prototype(
        baseline,
        config,
        seed=SEED,
        prototype=True,
        scenarios=list(SCENARIOS),
    )
    context, derived_changes = build_effect_context(baseline, synthetic)
    synthetic[EFFECT_COLUMN] = context

    # Independently recompute all engineered features from modified original
    # measurements over the complete 1,009-row history.
    fully_recomputed = recompute_from_measurements(synthetic.loc[:, list(ORIGINAL_COLUMNS)])
    full_recomputation_matches = _frame_numeric_equal(
        synthetic, fully_recomputed, FEATURE_COLUMNS[1:], rtol=1e-10, atol=2e-8
    )

    evaluation_mask = baseline.Timestamp.between(EVALUATION_START, EVALUATION_END)
    final = synthetic.loc[evaluation_mask, [*FEATURE_COLUMNS, *LABELS, EFFECT_COLUMN]].reset_index(drop=True)
    baseline_full_slice = baseline.loc[evaluation_mask, FEATURE_COLUMNS].reset_index(drop=True)
    baseline_evaluation = load_feature_dataset(BASELINE_EVALUATION)
    if not baseline_evaluation.Timestamp.equals(baseline_full_slice.Timestamp):
        raise RuntimeError("Saved baseline evaluation timestamps differ from the full source slice.")

    no_time = final.loc[:, NO_TIME_COLUMNS].copy()
    _atomic_csv(final, output_path)
    _atomic_csv(no_time, no_time_output_path)
    saved_final = pd.read_csv(output_path)
    saved_final["Timestamp"] = pd.to_datetime(saved_final["Timestamp"])
    saved_no_time = pd.read_csv(no_time_output_path)
    saved_no_time["Timestamp"] = pd.to_datetime(saved_no_time["Timestamp"])

    events, outside_ranges = enrich_events(
        generator_report["events"], baseline, synthetic, context, config
    )
    direct = final.synthetic_anomaly.eq(1)
    propagated = final[EFFECT_COLUMN].eq("propagated")
    none = final[EFFECT_COLUMN].eq("none")
    low_light = final.Solar_Radiation.le(float(config["context"]["low_light_threshold_w_m2"]))
    protected_full = baseline.Timestamp.between(PROTECTED_START, PROTECTED_END)
    protected_eval = final.Timestamp.between(PROTECTED_START, PROTECTED_END)
    before_evaluation = baseline.Timestamp.lt(EVALUATION_START)
    after_evaluation = baseline.Timestamp.gt(EVALUATION_END)
    original_changes = _changed_mask(baseline, synthetic, ORIGINAL_COLUMNS[1:]).any(axis=1)
    direct_event_counts = np.zeros(len(baseline), dtype=int)
    for event in generator_report["events"]:
        direct_event_counts[event["row_positions"]] += 1

    # Demonstrate why the output was not recomputed from the isolated slice.
    isolated_recomputed = recompute_from_measurements(final.loc[:, list(ORIGINAL_COLUMNS)])
    isolated_context_difference_count = int(sum(
        (~_series_equal(final[column], isolated_recomputed[column])).sum()
        for column in FEATURE_COLUMNS[len(ORIGINAL_COLUMNS):]
    ))

    expected_nan_counts = {
        column: int(final[column].isna().sum())
        for column in FEATURE_COLUMNS
        if final[column].isna().any()
    }
    scenario_ranges = {
        event["type"]: event["configured_ranges_respected"] for event in events
    }
    generated_original_unchanged = bool(
        _frame_numeric_equal(
            final.loc[~direct].reset_index(drop=True),
            baseline_evaluation.loc[~direct].reset_index(drop=True),
            ORIGINAL_COLUMNS[1:],
        )
    )
    protected_original_unchanged = _frame_numeric_equal(
        synthetic.loc[protected_full], baseline.loc[protected_full], ORIGINAL_COLUMNS[1:]
    )
    no_time_exact_columns = saved_no_time.columns.tolist() == list(NO_TIME_COLUMNS)
    saved_values_match = (
        saved_final.Timestamp.equals(final.Timestamp)
        and _frame_numeric_equal(saved_final, final, FEATURE_COLUMNS[1:])
        and np.array_equal(
            saved_final.synthetic_anomaly.to_numpy(dtype=int),
            final.synthetic_anomaly.to_numpy(dtype=int),
        )
        and np.array_equal(
            saved_final.synthetic_anomaly_type.astype(str).to_numpy(),
            final.synthetic_anomaly_type.astype(str).to_numpy(),
        )
        and np.array_equal(
            saved_final.synthetic_anomaly_id.fillna("<null>").astype(str).to_numpy(),
            final.synthetic_anomaly_id.fillna("<null>").astype(str).to_numpy(),
        )
        and np.array_equal(
            saved_final[EFFECT_COLUMN].astype(str).to_numpy(),
            final[EFFECT_COLUMN].astype(str).to_numpy(),
        )
    )

    validations = {
        "fixed_seed_is_2026": generator_report["seed"] == SEED,
        "all_seven_scenarios_attempted": set(event["anomaly_type"] for event in generator_report["events"]).union(
            skipped["scenario"] for skipped in generator_report["skipped_scenarios"]
        ) == set(SCENARIOS),
        "scenario_definitions_and_ranges_unchanged": all(scenario_ranges.values()),
        "full_history_used_for_feature_recomputation": full_recomputation_matches and isolated_context_difference_count > 0,
        "exactly_337_evaluation_rows": len(final) == 337,
        "evaluation_boundaries_exact": final.Timestamp.iloc[0] == EVALUATION_START and final.Timestamp.iloc[-1] == EVALUATION_END,
        "timestamps_identical_to_baseline_evaluation": final.Timestamp.equals(baseline_evaluation.Timestamp),
        "no_direct_modification_before_evaluation": not bool(original_changes.loc[before_evaluation].any()),
        "no_direct_modification_after_evaluation": not bool(original_changes.loc[after_evaluation].any()),
        "training_and_calibration_all_features_unchanged": _frame_numeric_equal(
            synthetic.loc[before_evaluation], baseline.loc[before_evaluation], FEATURE_COLUMNS[1:]
        ),
        "direct_event_windows_do_not_overlap": int(direct_event_counts.max(initial=0)) <= 1,
        "direct_labels_equal_modified_original_measurement_rows": bool(
            np.array_equal(synthetic.synthetic_anomaly.to_numpy(dtype=int), original_changes.to_numpy(dtype=int))
        ),
        "direct_modifications_match_generator_event_metadata": generator_report["validation"]["labels_match_direct_measurement_changes"],
        "propagated_rows_keep_direct_label_zero": bool(final.loc[propagated, "synthetic_anomaly"].eq(0).all()),
        "none_and_propagated_rows_preserve_original_measurements": generated_original_unchanged,
        "protected_transition_original_measurements_unchanged": protected_original_unchanged,
        "protected_transition_has_no_direct_or_propagated_label": bool(
            final.loc[protected_eval, "synthetic_anomaly"].eq(0).all()
            and final.loc[protected_eval, EFFECT_COLUMN].eq("none").all()
        ),
        "dependent_features_recomputed_causally": generator_report["validation"]["dependent_features_recomputed"] and full_recomputation_matches,
        "electrical_relationship_policy_respected": generator_report["validation"]["electrical_scaling_consistent"],
        "humidity_within_zero_to_one_hundred": bool(final.Relative_Humidity.between(0, 100).all()),
        "no_infinities": not bool(np.isinf(final.select_dtypes(include="number").to_numpy()).any()),
        "nan_values_match_feature_pipeline": _frame_numeric_equal(
            final,
            fully_recomputed.loc[evaluation_mask].reset_index(drop=True),
            FEATURE_COLUMNS[1:],
            rtol=1e-10,
            atol=2e-8,
        ),
        "positive_event_durations": all(event["duration_minutes"] > 0 for event in events),
        "saved_full_output_matches_generated_values": saved_values_match,
        "unscaled_no_time_7_matrix_has_exact_columns": no_time_exact_columns,
        "no_detector_scores_or_predictions_created": not any(
            "score" in column.lower() or "prediction" in column.lower() for column in final.columns
        ),
    }
    if not all(validations.values()):
        failed = [name for name, passed in validations.items() if not passed]
        raise RuntimeError(f"Final synthetic evaluation validation failed: {failed}")

    save_figure(baseline_evaluation, final, events, Path(figure_path))
    protected_after = _snapshot(protected_paths)
    frozen_after = _snapshot(list(FROZEN_FILES))
    source_unchanged = protected_before == protected_after
    frozen_unchanged = frozen_before == frozen_after
    if not source_unchanged or not frozen_unchanged:
        raise RuntimeError("A protected source or frozen detector/threshold artifact changed.")

    report = {
        "stage": "Day 21 final synthetic evaluation dataset generation and validation",
        "configuration_version": config["configuration_version"],
        "seed": SEED,
        "rng": "PCG64",
        "generator_version": GENERATOR_VERSION,
        "generation_scope": "synthetic direct modifications restricted to frozen evaluation partition",
        "full_history_policy": {
            "source_rows": len(baseline),
            "method": "inject into complete untouched sequence, recompute causally over full history, then extract evaluation",
            "isolated_slice_recomputation_not_used": True,
            "derived_cell_differences_if_isolated_slice_were_used": isolated_context_difference_count,
        },
        "evaluation_output": {
            "path": _relative(output_path),
            "sha256": sha256_file(output_path),
            "shape": [len(final), len(final.columns)],
            "timestamp_start": _timestamp(final.Timestamp.iloc[0]),
            "timestamp_end": _timestamp(final.Timestamp.iloc[-1]),
        },
        "no_time_7_output": {
            "path": _relative(no_time_output_path),
            "sha256": sha256_file(no_time_output_path),
            "shape": [len(no_time), len(no_time.columns)],
            "scaled": False,
            "columns": list(NO_TIME_COLUMNS),
        },
        "event_count": len(events),
        "events": events,
        "skipped_scenarios": generator_report["skipped_scenarios"],
        "distribution_review": {
            "successful_scenarios": [event["type"] for event in events],
            "direct_rows": int(direct.sum()),
            "propagated_rows": int(propagated.sum()),
            "none_rows": int(none.sum()),
            "anomaly_types_represented": sorted(final.loc[direct, "synthetic_anomaly_type"].unique()),
            "direct_daylight_like_rows": int((direct & ~low_light).sum()),
            "direct_low_light_rows": int((direct & low_light).sum()),
        },
        "effect_context_definitions": {
            "direct": "original operational or sensor measurement deliberately modified; synthetic_anomaly=1",
            "propagated": "original measurements untouched but causal engineered features differ; synthetic_anomaly=0",
            "none": "neither direct nor propagated effect",
        },
        "protected_transition": {
            "start": _timestamp(PROTECTED_START),
            "end": _timestamp(PROTECTED_END),
            "row_count": int(protected_eval.sum()),
            "original_measurements_unchanged": protected_original_unchanged,
            "direct_rows": int(final.loc[protected_eval, "synthetic_anomaly"].sum()),
            "effect_context_counts": final.loc[protected_eval, EFFECT_COLUMN].value_counts().to_dict(),
        },
        "dependent_feature_recomputation": {
            "basic_feature_count": len(ENGINEERED_FEATURES),
            "rolling_feature_count": len(ROLLING_FEATURE_SPECS),
            "change_feature_count": len(CHANGE_FEATURES),
            "full_pipeline_recalculation_matches": full_recomputation_matches,
            "generator_causal_validation_passed": generator_report["validation"]["dependent_features_recomputed"],
            "rtd_summary_features_consistent": all(
                _series_equal(synthetic[column], fully_recomputed[column]).all()
                for column in ("rtd_mean", "rtd_std", "rtd_min", "rtd_max", "rtd_range")
            ),
        },
        "numerical_consistency": {
            "expected_nan_counts_by_column": expected_nan_counts,
            "nan_policy": "NaNs are accepted only where reproduced by the existing causal feature pipeline and denominator safeguards.",
            "infinity_count": int(np.isinf(final.select_dtypes(include="number").to_numpy()).sum()),
            "scenario_ranges_respected": scenario_ranges,
            "values_outside_observed_baseline_range": outside_ranges,
            "outside_range_wording": "outside the range observed in the available baseline dataset",
            "physical_realism_claimed": False,
        },
        "source_and_configuration_hashes": {
            "baseline_source": {"path": _relative(baseline_path), "sha256": sha256_file(baseline_path)},
            "baseline_evaluation": {"path": _relative(BASELINE_EVALUATION), "sha256": sha256_file(BASELINE_EVALUATION)},
            "configuration": {"path": _relative(config_path), "sha256": sha256_file(config_path)},
            "scenario_definition_source": {"path": _relative(SCENARIO_SOURCE_CONFIG), "sha256": sha256_file(SCENARIO_SOURCE_CONFIG)},
            "frozen_artifacts": {
                path: {"before": digest, "after": frozen_after[path], "unchanged": frozen_after[path] == digest}
                for path, digest in frozen_before.items()
            },
        },
        "validation": {
            "passed": True,
            "checks": {
                **validations,
                "protected_sources_unchanged": source_unchanged,
                "frozen_scaler_model_and_threshold_hashes_unchanged": frozen_unchanged,
            },
        },
        "protected_source_hashes": {
            path: {"before": digest, "after": protected_after[path], "unchanged": protected_after[path] == digest}
            for path, digest in protected_before.items()
        },
        "explicitly_not_performed": [
            "StandardScaler fitting",
            "Isolation Forest fitting or scoring",
            "threshold changes",
            "detector application to synthetic evaluation",
            "precision/recall/F1 or detection-performance calculation",
        ],
        "interpretation": (
            "This dataset provides controlled synthetic evaluation cases. It does not establish that the simulated "
            "signatures are equivalent to verified real photovoltaic faults."
        ),
    }
    _atomic_json(report, Path(metadata_path))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-time-output", type=Path, default=DEFAULT_NO_TIME_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_final_generation(
            config_path=args.config,
            output_path=args.output,
            no_time_output_path=args.no_time_output,
            metadata_path=args.metadata,
            figure_path=args.figure,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, yaml.YAMLError) as error:
        print(f"Error: {error}")
        return 1
    distribution = report["distribution_review"]
    print(f"Seed: {report['seed']}")
    print(f"Final evaluation shape: {report['evaluation_output']['shape']}")
    print(f"Successful events: {report['event_count']}; skipped: {len(report['skipped_scenarios'])}")
    print(
        "Effect rows: "
        f"direct={distribution['direct_rows']}, "
        f"propagated={distribution['propagated_rows']}, "
        f"none={distribution['none_rows']}"
    )
    print(f"Validation passed: {report['validation']['passed']}")
    print("The frozen detector was not applied and no performance metrics were calculated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
