"""Create the adopted Day 15 chronological baseline and Core datasets.

This utility performs dataset partitioning and descriptive validation only. It
does not fit a scaler, inject synthetic events, train a detector, select a
threshold, or calculate model performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data_processing.engineer_basic_features import PROCESSED_TIMESTAMP_FORMAT
from src.data_processing.review_model_features import (
    CORE_UNSUPERVISED_FEATURES,
    FEATURE_COLUMNS,
    load_feature_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config/chronological_split.yaml"
DEFAULT_SOURCE_PATH = ROOT / "data/processed/operational_features_change.csv"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "data/model_ready"
DEFAULT_REPORT_PATH = ROOT / "outputs/model_dataset_preparation.json"
DEFAULT_MODELS_DIRECTORY = ROOT / "models"
PARTITION_ORDER = ("training", "calibration", "evaluation_baseline")
BASELINE_FILENAMES = {
    "training": "baseline_train.csv",
    "calibration": "baseline_calibration.csv",
    "evaluation_baseline": "baseline_evaluation.csv",
}
CORE_FILENAMES = {
    "training": "core_train.csv",
    "calibration": "core_calibration.csv",
    "evaluation_baseline": "core_evaluation_baseline.csv",
}
SYNTHETIC_LABEL_COLUMNS = {
    "synthetic_anomaly",
    "synthetic_anomaly_type",
    "synthetic_anomaly_id",
    "synthetic_effect_context",
}
MODEL_OR_SCALER_SUFFIXES = {".joblib", ".pkl", ".pickle", ".pt", ".onnx", ".sav"}


def sha256_file(path: str | Path) -> str:
    """Return a SHA-256 digest without altering the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_files(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _timestamp_text(value: Any) -> str:
    return pd.Timestamp(value).isoformat(sep=" ")


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv.tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(
                handle,
                index=False,
                date_format=PROCESSED_TIMESTAMP_FORMAT,
            )
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


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


def load_split_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the reusable chronological-split configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Split configuration does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Split configuration must be a YAML mapping.")
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported chronological-split schema version.")
    if list(config.get("partitions", {})) != list(PARTITION_ORDER):
        raise ValueError(f"Partitions must be ordered as {PARTITION_ORDER}.")
    configured_core = tuple(config.get("core_features", []))
    if configured_core != tuple(CORE_UNSUPERVISED_FEATURES):
        raise ValueError("Configured Core features differ from the approved Core nine.")

    previous_end: pd.Timestamp | None = None
    for name in PARTITION_ORDER:
        specification = config["partitions"][name]
        start = pd.Timestamp(specification["start"])
        end = pd.Timestamp(specification["end"])
        if start.tzinfo is not None or end.tzinfo is not None:
            raise ValueError("Split boundaries must remain timezone-naive.")
        if start > end:
            raise ValueError(f"Partition {name} starts after it ends.")
        if previous_end is not None and start <= previous_end:
            raise ValueError("Configured chronological partitions overlap or are unordered.")
        if int(specification["expected_rows"]) <= 0:
            raise ValueError(f"Partition {name} must expect at least one row.")
        previous_end = end
    return config


def create_partitions(
    source: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Apply exact timestamp boundaries and prove complete one-time coverage."""
    timestamp_column = config["dataset"]["timestamp_column"]
    if source.columns.tolist() != list(FEATURE_COLUMNS):
        raise ValueError("Source must retain the verified 81-column feature schema.")
    if timestamp_column not in source:
        raise ValueError(f"Timestamp column is missing: {timestamp_column}")
    if len(source) != int(config["dataset"]["expected_rows"]):
        raise ValueError("Source row count differs from the split configuration.")
    if source.shape[1] != int(config["dataset"]["expected_columns"]):
        raise ValueError("Source column count differs from the split configuration.")
    if source[timestamp_column].isna().any():
        raise ValueError("Source contains timestamp parsing failures.")
    if source[timestamp_column].duplicated().any():
        raise ValueError("Source contains duplicate timestamps.")
    if not source[timestamp_column].is_monotonic_increasing:
        raise ValueError("Source timestamps are not chronologically ordered.")

    assignments = np.zeros(len(source), dtype=np.int8)
    partitions: dict[str, pd.DataFrame] = {}
    positions: dict[str, list[int]] = {}
    for name in PARTITION_ORDER:
        specification = config["partitions"][name]
        start = pd.Timestamp(specification["start"])
        end = pd.Timestamp(specification["end"])
        mask = source[timestamp_column].between(start, end, inclusive="both")
        assignments += mask.to_numpy(dtype=np.int8)
        position = np.flatnonzero(mask).tolist()
        partition = source.loc[mask].copy().reset_index(drop=True)
        if len(partition) != int(specification["expected_rows"]):
            raise ValueError(
                f"Partition {name} has {len(partition)} rows; expected "
                f"{specification['expected_rows']}."
            )
        if partition.empty or partition[timestamp_column].iloc[0] != start:
            raise ValueError(f"Partition {name} does not start at its configured boundary.")
        if partition[timestamp_column].iloc[-1] != end:
            raise ValueError(f"Partition {name} does not end at its configured boundary.")
        if not partition[timestamp_column].is_monotonic_increasing:
            raise ValueError(f"Partition {name} is not chronologically ordered.")
        partitions[name] = partition
        positions[name] = position

    if not np.all(assignments == 1):
        raise ValueError("Every source row must be assigned to exactly one partition.")
    all_positions = [row for name in PARTITION_ORDER for row in positions[name]]
    validation = {
        "expected_total_rows": int(config["dataset"]["expected_rows"]),
        "partition_row_sum": sum(len(frame) for frame in partitions.values()),
        "every_source_row_assigned_exactly_once": bool(np.all(assignments == 1)),
        "no_overlap": bool(assignments.max(initial=0) == 1),
        "no_row_loss": bool(assignments.min(initial=1) == 1),
        "no_duplicate_source_positions": len(all_positions) == len(set(all_positions)),
        "no_duplicate_timestamps": not source[timestamp_column].duplicated().any(),
        "chronological_source_order": source[timestamp_column].is_monotonic_increasing,
        "all_81_columns_preserved": all(
            frame.columns.tolist() == source.columns.tolist()
            for frame in partitions.values()
        ),
    }
    if not all(value for key, value in validation.items() if isinstance(value, bool)):
        raise RuntimeError("Chronological split coverage validation failed.")
    return partitions, validation


def create_core_matrices(
    partitions: dict[str, pd.DataFrame],
    timestamp_column: str = "Timestamp",
) -> dict[str, pd.DataFrame]:
    """Extract timestamp plus the approved unscaled Core feature set."""
    columns = [timestamp_column, *CORE_UNSUPERVISED_FEATURES]
    matrices: dict[str, pd.DataFrame] = {}
    for name in PARTITION_ORDER:
        missing = [column for column in columns if column not in partitions[name]]
        if missing:
            raise ValueError(f"Partition {name} lacks Core columns: {missing}")
        matrix = partitions[name].loc[:, columns].copy()
        if SYNTHETIC_LABEL_COLUMNS.intersection(matrix.columns):
            raise ValueError("Synthetic labels must not enter baseline Core matrices.")
        matrices[name] = matrix
    return matrices


def core_feature_statistics(
    matrices: dict[str, pd.DataFrame],
    timestamp_column: str = "Timestamp",
) -> dict[str, Any]:
    """Calculate finite-value and descriptive readiness evidence."""
    result: dict[str, Any] = {}
    for name, matrix in matrices.items():
        feature_stats: dict[str, Any] = {}
        for feature in CORE_UNSUPERVISED_FEATURES:
            values = matrix[feature]
            finite_count = int(np.isfinite(values.to_numpy(dtype=float)).sum())
            variance = float(values.var(ddof=0))
            feature_stats[feature] = {
                "missing_count": int(values.isna().sum()),
                "infinite_count": int(np.isinf(values.to_numpy(dtype=float)).sum()),
                "finite_count": finite_count,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "population_standard_deviation": float(values.std(ddof=0)),
                "population_variance": variance,
                "positive_variance": bool(math.isfinite(variance) and variance > 0),
                "usable": bool(
                    values.isna().sum() == 0
                    and finite_count == len(matrix)
                    and variance > 0
                ),
            }
        result[name] = {
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "timestamp_start": _timestamp_text(matrix[timestamp_column].iloc[0]),
            "timestamp_end": _timestamp_text(matrix[timestamp_column].iloc[-1]),
            "features": feature_stats,
            "all_core_features_usable": all(
                statistics["usable"] for statistics in feature_stats.values()
            ),
        }
    return result


def compare_distributions(core_stats: dict[str, Any]) -> dict[str, Any]:
    """Describe shifts using training-standardized mean differences."""
    comparisons: dict[str, Any] = {}
    notable: list[dict[str, Any]] = []
    training = core_stats["training"]["features"]
    for feature in CORE_UNSUPERVISED_FEATURES:
        comparisons[feature] = {}
        train_mean = training[feature]["mean"]
        train_std = training[feature]["population_standard_deviation"]
        for name in ("calibration", "evaluation_baseline"):
            current = core_stats[name]["features"][feature]
            standardized = (current["mean"] - train_mean) / train_std
            entry = {
                "training_mean": train_mean,
                "partition_mean": current["mean"],
                "mean_difference": current["mean"] - train_mean,
                "training_standardized_mean_difference": float(standardized),
                "absolute_standardized_difference": float(abs(standardized)),
            }
            comparisons[feature][name] = entry
            if abs(standardized) >= 0.5:
                notable.append(
                    {
                        "feature": feature,
                        "partition": name,
                        **entry,
                    }
                )
    notable.sort(key=lambda item: item["absolute_standardized_difference"], reverse=True)
    return {
        "method": (
            "Partition mean minus training mean, divided by training population "
            "standard deviation; descriptive only, not a population hypothesis test."
        ),
        "notable_rule": "absolute training-standardized mean difference >= 0.5",
        "by_feature": comparisons,
        "notable_differences": notable,
    }


def contextual_coverage(
    partitions: dict[str, pd.DataFrame], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify exploratory light context and locate recurring transitions."""
    threshold = float(config["context"]["low_light_threshold_w_m2"])
    light_counts: dict[str, Any] = {}
    for name, frame in partitions.items():
        low_light = frame["Solar_Radiation"].le(threshold)
        stored_context_matches = frame["low_light_context"].eq(low_light.astype(int))
        light_counts[name] = {
            "daylight_count": int((~low_light).sum()),
            "low_light_count": int(low_light.sum()),
            "threshold_w_m2": threshold,
            "stored_context_matches_recalculation": bool(stored_context_matches.all()),
        }

    transitions: dict[str, Any] = {}
    for configured_timestamp in config["context"]["recurring_transition_timestamps"]:
        timestamp = pd.Timestamp(configured_timestamp)
        matches = [
            name
            for name, frame in partitions.items()
            if frame["Timestamp"].eq(timestamp).any()
        ]
        if len(matches) != 1:
            raise ValueError(f"Transition {timestamp} must occur in exactly one partition.")
        transitions[_timestamp_text(timestamp)] = {
            "partition": matches[0],
            "present_exactly_once": True,
            "removed": False,
            "labelled_as_anomaly": False,
            "interpretation": (
                "Known recurring operational transition; retain and interpret carefully "
                "during later false-positive analysis."
            ),
        }
    return light_counts, transitions


def run_preparation(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    source_path: Path | None = None,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    report_path: Path = DEFAULT_REPORT_PATH,
    models_directory: Path = DEFAULT_MODELS_DIRECTORY,
) -> dict[str, Any]:
    """Create split files and a strict JSON report, preserving all source data."""
    config = load_split_config(config_path)
    source = Path(source_path or ROOT / config["dataset"]["feature_file"])
    source_hash_before = sha256_file(source)
    config_hash = sha256_file(config_path)
    models_before = _snapshot_files(models_directory)

    frame = load_feature_dataset(source)
    partitions, coverage = create_partitions(frame, config)
    matrices = create_core_matrices(partitions, config["dataset"]["timestamp_column"])
    statistics = core_feature_statistics(matrices, config["dataset"]["timestamp_column"])
    distribution = compare_distributions(statistics)
    light_counts, transitions = contextual_coverage(partitions, config)

    output_directory = Path(output_directory)
    written: dict[str, Path] = {}
    for name, partition in partitions.items():
        destination = output_directory / BASELINE_FILENAMES[name]
        _atomic_csv(partition, destination)
        written[f"baseline_{name}"] = destination
    for name, matrix in matrices.items():
        destination = output_directory / CORE_FILENAMES[name]
        _atomic_csv(matrix, destination)
        written[f"core_{name}"] = destination

    # Validate the published CSVs, including reconstruction of the source order.
    saved_baseline = {
        name: load_feature_dataset(output_directory / BASELINE_FILENAMES[name])
        for name in PARTITION_ORDER
    }
    reconstructed = pd.concat(
        [saved_baseline[name] for name in PARTITION_ORDER], ignore_index=True
    )
    reconstructed_matches = bool(
        reconstructed.columns.equals(frame.columns)
        and reconstructed["Timestamp"].equals(frame["Timestamp"])
        and np.allclose(
            reconstructed.drop(columns="Timestamp").to_numpy(dtype=float),
            frame.drop(columns="Timestamp").to_numpy(dtype=float),
            equal_nan=True,
            rtol=0.0,
            # CSV parsing/serialization can move a binary float by roughly one
            # machine-precision unit; this does not alter the represented data.
            atol=1e-12,
        )
    )
    coverage["saved_partitions_reconstruct_source_within_csv_precision"] = (
        reconstructed_matches
    )
    source_hash_after = sha256_file(source)
    models_after = _snapshot_files(models_directory)
    forbidden_artifacts = [
        relative
        for relative in models_after
        if Path(relative).suffix.lower() in MODEL_OR_SCALER_SUFFIXES
        or "scaler" in Path(relative).name.lower()
    ]

    policy = [
        "Load the Core training matrix.",
        "Fit a scaler only on training features.",
        "Transform training with that fitted scaler.",
        "Apply the same already-fitted scaler to calibration.",
        "Train an unsupervised detector using training data only.",
        "Use untouched calibration data to study normal scores and freeze a threshold.",
        "Generate a new synthetic copy only within the evaluation period.",
        "Recompute dependent features for the synthetic evaluation copy.",
        "Extract the same Core features.",
        "Apply the training-fitted scaler.",
        "Evaluate using the frozen detector and threshold.",
    ]
    validation = {
        **{key: value for key, value in coverage.items() if isinstance(value, bool)},
        "all_core_features_usable": all(
            entry["all_core_features_usable"] for entry in statistics.values()
        ),
        "all_context_counts_match_stored_flag": all(
            entry["stored_context_matches_recalculation"]
            for entry in light_counts.values()
        ),
        "source_file_unchanged": source_hash_before == source_hash_after,
        "models_directory_unchanged": models_before == models_after,
        "no_model_or_scaler_artifacts": not forbidden_artifacts,
        "no_synthetic_labels_in_baseline_or_core_files": all(
            not SYNTHETIC_LABEL_COLUMNS.intersection(dataframe.columns)
            for dataframe in [*partitions.values(), *matrices.values()]
        ),
    }
    if not all(validation.values()):
        raise RuntimeError("Day 15 model-dataset preparation validation failed.")

    report: dict[str, Any] = {
        "stage": "Day 15 final chronological split creation and model dataset preparation",
        "adopted_split": {
            "name": config["split_name"],
            "status": config["status"],
            "boundaries_are_inclusive": True,
            "partitions": {
                name: {
                    "start": config["partitions"][name]["start"],
                    "end": config["partitions"][name]["end"],
                    "row_count": len(partitions[name]),
                    "baseline_file": _relative(output_directory / BASELINE_FILENAMES[name]),
                    "core_file": _relative(output_directory / CORE_FILENAMES[name]),
                }
                for name in PARTITION_ORDER
            },
            "reason": config["split_policy"]["reason"],
            "temporal_limitation": config["split_policy"]["temporal_limitation"],
        },
        "coverage_validation": coverage,
        "daylight_low_light_coverage": light_counts,
        "core_feature_statistics": statistics,
        "distribution_comparison": distribution,
        "recurring_transition_locations": transitions,
        "future_model_preprocessing_policy": policy,
        "leakage_policy": {
            "scaler_fit_scope": "training features only",
            "detector_fit_scope": "training features only",
            "threshold_selection_scope": "untouched calibration only",
            "synthetic_labels_used_for_training": False,
        },
        "source_hashes": {
            "source_file": _relative(source),
            "source_sha256_before": source_hash_before,
            "source_sha256_after": source_hash_after,
            "source_unchanged": source_hash_before == source_hash_after,
            "configuration_file": _relative(config_path),
            "configuration_sha256": config_hash,
            "created_files": {
                label: {"path": _relative(path), "sha256": sha256_file(path)}
                for label, path in written.items()
            },
        },
        "models_directory": {
            "before": models_before,
            "after": models_after,
            "unchanged": models_before == models_after,
            "forbidden_artifacts": forbidden_artifacts,
        },
        "validation": {"passed": True, "checks": validation},
        "explicitly_not_performed": [
            "scaler fitting",
            "synthetic anomaly injection",
            "detector training",
            "threshold calibration",
            "model-performance evaluation",
        ],
    }
    _atomic_json(report, Path(report_path))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_preparation(
            config_path=args.config,
            source_path=args.source,
            output_directory=args.output_directory,
            report_path=args.report,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, yaml.YAMLError) as error:
        print(f"Error: {error}")
        return 1
    partitions = report["adopted_split"]["partitions"]
    print(
        "Created baseline/Core partitions: "
        + ", ".join(f"{name}={entry['row_count']}" for name, entry in partitions.items())
    )
    print(f"Validation passed: {report['validation']['passed']}")
    print("No scaler, detector, threshold, synthetic evaluation, or model result was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
