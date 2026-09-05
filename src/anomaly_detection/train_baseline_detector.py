"""Fit the Day 16 training-only scaler and initial Isolation Forest.

The utility creates continuous baseline scores only. It does not select a final
threshold, generate synthetic evaluation data, create anomaly labels, or report
classification performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import tempfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.data_processing.engineer_basic_features import PROCESSED_TIMESTAMP_FORMAT
from src.data_processing.review_model_features import CORE_UNSUPERVISED_FEATURES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config/anomaly_model.yaml"
DEFAULT_MODELS_DIRECTORY = ROOT / "models"
DEFAULT_SCALED_DIRECTORY = ROOT / "data/model_ready/scaled"
DEFAULT_OUTPUTS_DIRECTORY = ROOT / "outputs"
DEFAULT_FIGURE_PATH = ROOT / "outputs/figures/operational/day16_baseline_anomaly_scores.png"
DEFAULT_SUMMARY_PATH = ROOT / "outputs/initial_detector_summary.json"
PARTITION_ORDER = ("training", "calibration", "evaluation_baseline")
EXPECTED_PARTITIONS = {
    "training": (336, "2022-04-27 15:32:00", "2022-04-28 02:42:00"),
    "calibration": (336, "2022-04-28 02:44:00", "2022-04-28 13:54:00"),
    "evaluation_baseline": (337, "2022-04-28 13:56:00", "2022-04-29 01:08:00"),
}
SCALED_FILENAMES = {
    "training": "core_train_scaled.csv",
    "calibration": "core_calibration_scaled.csv",
    "evaluation_baseline": "core_evaluation_baseline_scaled.csv",
}
SCORE_FILENAMES = {
    "training": "anomaly_scores_train.csv",
    "calibration": "anomaly_scores_calibration.csv",
    "evaluation_baseline": "anomaly_scores_evaluation_baseline.csv",
}
SCALER_FILENAME = "core_standard_scaler.joblib"
MODEL_FILENAME = "isolation_forest_core.joblib"
METADATA_FILENAME = "isolation_forest_core_metadata.json"
SYNTHETIC_LABEL_COLUMNS = {
    "synthetic_anomaly",
    "synthetic_anomaly_type",
    "synthetic_anomaly_id",
    "synthetic_effect_context",
}
TRANSITION_TIMESTAMPS = (
    pd.Timestamp("2022-04-27 17:00:00"),
    pd.Timestamp("2022-04-28 17:00:00"),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _snapshot(paths: list[Path]) -> dict[str, str]:
    return {_relative(path): sha256_file(path) for path in paths if path.is_file()}


def _timestamp(value: Any) -> str:
    return pd.Timestamp(value).isoformat(sep=" ")


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


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".csv.tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False, date_format=PROCESSED_TIMESTAMP_FORMAT)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_joblib(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".joblib.tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        joblib.dump(value, temporary, compress=3)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_model_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a strict configuration for the fixed Day 16 experiment."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Model configuration does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported anomaly-model configuration.")
    if tuple(config.get("core_features", [])) != tuple(CORE_UNSUPERVISED_FEATURES):
        raise ValueError("Configuration must preserve the approved Core feature order.")
    if config.get("scaler", {}).get("type") != "StandardScaler":
        raise ValueError("Day 16 requires StandardScaler.")
    if config.get("scaler", {}).get("fit_scope") != "training_only":
        raise ValueError("Scaler fit scope must be training_only.")
    if config.get("model", {}).get("type") != "IsolationForest":
        raise ValueError("Day 16 requires IsolationForest.")
    parameters = config["model"].get("parameters", {})
    required = {
        "n_estimators", "max_samples", "max_features", "bootstrap",
        "contamination", "random_state", "n_jobs",
    }
    if set(parameters) != required:
        raise ValueError(f"Isolation Forest parameters must be exactly {sorted(required)}.")
    if parameters["contamination"] != "auto":
        raise ValueError("Contamination must not encode a presumed anomaly fraction.")
    if parameters["random_state"] != config.get("random_state"):
        raise ValueError("Model and experiment random states must match.")
    if config.get("scoring", {}).get("create_final_labels") is not False:
        raise ValueError("Day 16 must not create final anomaly labels.")
    return config


def load_core_partitions(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load timestamp plus exactly nine finite Core features."""
    expected_columns = ["Timestamp", *CORE_UNSUPERVISED_FEATURES]
    partitions: dict[str, pd.DataFrame] = {}
    for name in PARTITION_ORDER:
        path = ROOT / config["inputs"][name]
        if not path.is_file():
            raise FileNotFoundError(f"Core partition does not exist: {path}")
        frame = pd.read_csv(path)
        if frame.columns.tolist() != expected_columns:
            raise ValueError(f"{name} must contain Timestamp plus the Core nine in order.")
        if SYNTHETIC_LABEL_COLUMNS.intersection(frame.columns):
            raise ValueError(f"Synthetic labels found in baseline partition {name}.")
        frame["Timestamp"] = pd.to_datetime(
            frame["Timestamp"], format=PROCESSED_TIMESTAMP_FORMAT, errors="coerce"
        )
        rows, start, end = EXPECTED_PARTITIONS[name]
        if len(frame) != rows or frame["Timestamp"].isna().any():
            raise ValueError(f"{name} has an invalid row count or timestamp.")
        if frame.Timestamp.iloc[0] != pd.Timestamp(start) or frame.Timestamp.iloc[-1] != pd.Timestamp(end):
            raise ValueError(f"{name} does not match adopted split boundaries.")
        if frame.Timestamp.duplicated().any() or not frame.Timestamp.is_monotonic_increasing:
            raise ValueError(f"{name} timestamps must be unique and chronological.")
        for feature in CORE_UNSUPERVISED_FEATURES:
            original_missing = frame[feature].isna()
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
            if (frame[feature].isna() & ~original_missing).any():
                raise ValueError(f"{name}.{feature} contains non-numeric values.")
        values = frame[list(CORE_UNSUPERVISED_FEATURES)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} Core features must be complete and finite.")
        partitions[name] = frame
    combined = pd.concat([partitions[name] for name in PARTITION_ORDER], ignore_index=True)
    if combined.Timestamp.duplicated().any() or not combined.Timestamp.is_monotonic_increasing:
        raise ValueError("Core partitions overlap or violate chronological order.")
    return partitions


def fit_training_scaler(
    training: pd.DataFrame, features: tuple[str, ...] = CORE_UNSUPERVISED_FEATURES
) -> StandardScaler:
    """Fit exactly one scaler on training features, excluding Timestamp."""
    values = training.loc[:, features].to_numpy(dtype=float)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(values)
    return scaler


def transform_partitions(
    partitions: dict[str, pd.DataFrame], scaler: StandardScaler,
    features: tuple[str, ...] = CORE_UNSUPERVISED_FEATURES,
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """Apply the already-fitted training scaler to every partition."""
    arrays: dict[str, np.ndarray] = {}
    frames: dict[str, pd.DataFrame] = {}
    for name in PARTITION_ORDER:
        transformed = scaler.transform(partitions[name].loc[:, features].to_numpy(dtype=float))
        if not np.isfinite(transformed).all():
            raise RuntimeError(f"Scaling introduced a non-finite value in {name}.")
        arrays[name] = transformed
        scaled = pd.DataFrame(transformed, columns=features)
        scaled.insert(0, "Timestamp", partitions[name]["Timestamp"].to_numpy())
        frames[name] = scaled
    return arrays, frames


def fit_detector(training_scaled: np.ndarray, parameters: dict[str, Any]) -> IsolationForest:
    """Fit the initial detector using the scaled training matrix only."""
    detector = IsolationForest(**parameters)
    detector.fit(training_scaled)
    return detector


def score_partitions(
    partitions: dict[str, pd.DataFrame], transformed: dict[str, np.ndarray],
    detector: IsolationForest,
) -> dict[str, pd.DataFrame]:
    """Return continuous scores where higher means more isolated."""
    scores: dict[str, pd.DataFrame] = {}
    for name in PARTITION_ORDER:
        score = -detector.score_samples(transformed[name])
        if not np.isfinite(score).all():
            raise RuntimeError(f"Non-finite anomaly score produced for {name}.")
        scores[name] = pd.DataFrame(
            {"Timestamp": partitions[name]["Timestamp"].to_numpy(), "anomaly_score": score}
        )
    return scores


def scaler_review(
    partitions: dict[str, pd.DataFrame], transformed: dict[str, np.ndarray],
    scaler: StandardScaler,
) -> dict[str, Any]:
    features = list(CORE_UNSUPERVISED_FEATURES)
    training = partitions["training"][features]
    review: dict[str, Any] = {
        "fit_partition": "training",
        "fit_row_count": int(scaler.n_samples_seen_),
        "timestamp_excluded": True,
        "feature_order": features,
        "training_means_before_scaling": dict(zip(features, training.mean().astype(float))),
        "training_population_std_before_scaling": dict(zip(features, training.std(ddof=0).astype(float))),
        "fitted_mean": dict(zip(features, scaler.mean_.astype(float))),
        "fitted_scale": dict(zip(features, scaler.scale_.astype(float))),
        "scaled_training_means": dict(zip(features, transformed["training"].mean(axis=0).astype(float))),
        "scaled_training_population_std": dict(zip(features, transformed["training"].std(axis=0).astype(float))),
        "transformed_partition_ranges": {},
    }
    for name in PARTITION_ORDER:
        values = transformed[name]
        review["transformed_partition_ranges"][name] = {
            feature: {"minimum": float(values[:, index].min()), "maximum": float(values[:, index].max())}
            for index, feature in enumerate(features)
        }
    review["no_missing_or_infinite_values"] = all(
        np.isfinite(values).all() for values in transformed.values()
    )
    review["calibration_or_evaluation_used_for_fit"] = False
    return review


def score_statistics(scores: dict[str, pd.DataFrame]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, frame in scores.items():
        values = frame["anomaly_score"]
        result[name] = {
            "row_count": len(frame),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "population_standard_deviation": float(values.std(ddof=0)),
            "percentiles": {
                "90": float(values.quantile(0.90)),
                "95": float(values.quantile(0.95)),
                "97.5": float(values.quantile(0.975)),
                "99": float(values.quantile(0.99)),
            },
        }
    training_mean = result["training"]["mean"]
    training_std = result["training"]["population_standard_deviation"]
    for name in PARTITION_ORDER:
        result[name]["mean_difference_from_training"] = result[name]["mean"] - training_mean
        result[name]["training_standardized_mean_difference"] = (
            result[name]["mean"] - training_mean
        ) / training_std
    return result


def transition_review(scores: dict[str, pd.DataFrame]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for transition in TRANSITION_TIMESTAMPS:
        located = [name for name, frame in scores.items() if frame.Timestamp.eq(transition).any()]
        if len(located) != 1:
            raise ValueError(f"Transition {transition} must occur in exactly one score partition.")
        name = located[0]
        frame = scores[name]
        row = frame.loc[frame.Timestamp.eq(transition)].iloc[0]
        nearby = frame.loc[
            frame.Timestamp.between(
                transition - pd.Timedelta(minutes=10),
                transition + pd.Timedelta(minutes=10),
            )
            & frame.Timestamp.ne(transition),
            "anomaly_score",
        ]
        result[_timestamp(transition)] = {
            "partition": name,
            "anomaly_score": float(row.anomaly_score),
            "nearby_window_minutes": 10,
            "nearby_row_count_excluding_transition": len(nearby),
            "nearby_minimum": float(nearby.min()),
            "nearby_maximum": float(nearby.max()),
            "nearby_mean": float(nearby.mean()),
            "nearby_median": float(nearby.median()),
            "classified_as_fault": False,
            "interpretation": "Known recurring operational transition; review neutrally during later false-positive analysis.",
        }
    return result


def save_score_figure(
    scores: dict[str, pd.DataFrame], destination: Path,
) -> None:
    colors = {"training": "#2f6690", "calibration": "#e6a23c", "evaluation_baseline": "#5b9a62"}
    labels = {"training": "Training", "calibration": "Calibration", "evaluation_baseline": "Evaluation baseline"}
    fig, axis = plt.subplots(figsize=(13, 5))
    for name in PARTITION_ORDER:
        frame = scores[name]
        axis.plot(frame.Timestamp, frame.anomaly_score, color=colors[name], linewidth=1.1, label=labels[name])
    for name in PARTITION_ORDER[1:]:
        axis.axvline(scores[name].Timestamp.iloc[0], color="#555555", linestyle="--", linewidth=1)
    for number, transition in enumerate(TRANSITION_TIMESTAMPS, start=1):
        axis.axvline(transition, color="#8b1e3f", linestyle=":", linewidth=1.2)
        axis.text(transition, axis.get_ylim()[1], f"17:00 transition {number}", rotation=90, ha="right", va="top", fontsize=8, color="#8b1e3f")
    axis.set_title("Initial Isolation Forest baseline scores (no final threshold)")
    axis.set_xlabel("Source timestamp (timezone not specified)")
    axis.set_ylabel("Anomaly score = -score_samples (higher is more isolated)")
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png.tmp", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
        fig.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
        temporary.replace(destination)
    finally:
        plt.close(fig)
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_training(
    *, config_path: Path = DEFAULT_CONFIG_PATH,
    models_directory: Path = DEFAULT_MODELS_DIRECTORY,
    scaled_directory: Path = DEFAULT_SCALED_DIRECTORY,
    outputs_directory: Path = DEFAULT_OUTPUTS_DIRECTORY,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
) -> dict[str, Any]:
    """Run the training-only Day 16 workflow and publish validated artifacts."""
    config = load_model_config(config_path)
    input_paths = [ROOT / config["inputs"][name] for name in PARTITION_ORDER]
    protected_paths = sorted(path for path in (ROOT / "data").rglob("*.csv") if "scaled" not in path.parts)
    protected_before = _snapshot(protected_paths)
    inputs_before = _snapshot(input_paths)
    synthetic_paths_before = sorted(_relative(path) for path in (ROOT / "data").rglob("*synthetic*"))

    partitions = load_core_partitions(config)
    scaler = fit_training_scaler(partitions["training"])
    transformed, scaled_frames = transform_partitions(partitions, scaler)
    parameters = config["model"]["parameters"]
    detector = fit_detector(transformed["training"], parameters)
    repeated = fit_detector(transformed["training"], parameters)
    scores = score_partitions(partitions, transformed, detector)
    repeated_scores = score_partitions(partitions, transformed, repeated)
    deterministic = all(
        np.array_equal(scores[name].anomaly_score.to_numpy(), repeated_scores[name].anomaly_score.to_numpy())
        for name in PARTITION_ORDER
    )

    scaled_directory = Path(scaled_directory)
    outputs_directory = Path(outputs_directory)
    models_directory = Path(models_directory)
    for name in PARTITION_ORDER:
        _atomic_csv(scaled_frames[name], scaled_directory / SCALED_FILENAMES[name])
        _atomic_csv(scores[name], outputs_directory / SCORE_FILENAMES[name])
    scaler_path = models_directory / SCALER_FILENAME
    model_path = models_directory / MODEL_FILENAME
    metadata_path = models_directory / METADATA_FILENAME
    _atomic_joblib(scaler, scaler_path)
    _atomic_joblib(detector, model_path)

    model_metadata = {
        "stage": "Day 16 initial baseline detector",
        "model_type": "IsolationForest",
        "parameters": detector.get_params(deep=False),
        "random_state": parameters["random_state"],
        "feature_order": list(CORE_UNSUPERVISED_FEATURES),
        "timestamp_used_as_model_feature": False,
        "training_row_count": len(partitions["training"]),
        "training_timestamp_start": _timestamp(partitions["training"].Timestamp.iloc[0]),
        "training_timestamp_end": _timestamp(partitions["training"].Timestamp.iloc[-1]),
        "training_source_path": _relative(input_paths[0]),
        "training_source_sha256": sha256_file(input_paths[0]),
        "scaler_path": _relative(scaler_path),
        "scaler_sha256": sha256_file(scaler_path),
        "model_path": _relative(model_path),
        "model_sha256": sha256_file(model_path),
        "score_convention": config["scoring"],
        "final_threshold_calibrated": False,
        "synthetic_labels_used": False,
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    _atomic_json(model_metadata, metadata_path)

    # Reload the saved artifacts and prove their outputs and metadata agree.
    saved_scaler = joblib.load(scaler_path)
    saved_detector = joblib.load(model_path)
    saved_training = saved_scaler.transform(
        partitions["training"].loc[:, CORE_UNSUPERVISED_FEATURES].to_numpy(dtype=float)
    )
    saved_scores = -saved_detector.score_samples(saved_training)
    serialized_artifacts_match = bool(
        np.array_equal(saved_training, transformed["training"])
        and np.array_equal(saved_scores, scores["training"].anomaly_score.to_numpy())
    )
    review = scaler_review(partitions, transformed, scaler)
    statistics = score_statistics(scores)
    transitions = transition_review(scores)
    save_score_figure(scores, Path(figure_path))

    protected_after = _snapshot(protected_paths)
    synthetic_paths_after = sorted(_relative(path) for path in (ROOT / "data").rglob("*synthetic*"))
    model_files = sorted(path.name for path in models_directory.iterdir() if path.is_file())
    expected_model_files = sorted([".gitkeep", SCALER_FILENAME, MODEL_FILENAME, METADATA_FILENAME])
    validation = {
        "scaler_fit_count_is_336": int(scaler.n_samples_seen_) == 336,
        "scaler_fitted_on_training_only": review["calibration_or_evaluation_used_for_fit"] is False,
        "exact_core_feature_order": list(CORE_UNSUPERVISED_FEATURES) == model_metadata["feature_order"],
        "timestamp_excluded_from_scaler_and_model": scaler.n_features_in_ == detector.n_features_in_ == 9,
        "all_transformed_values_finite": review["no_missing_or_infinite_values"],
        "model_fit_input_is_336_by_9": transformed["training"].shape == (336, 9),
        "fixed_seed_outputs_are_deterministic": deterministic,
        "score_files_have_only_timestamp_and_continuous_score": all(
            frame.columns.tolist() == ["Timestamp", "anomaly_score"] for frame in scores.values()
        ),
        "score_timestamps_match_source_partitions": all(
            scores[name].Timestamp.equals(partitions[name].Timestamp) for name in PARTITION_ORDER
        ),
        "no_final_anomaly_labels_created": all(
            not SYNTHETIC_LABEL_COLUMNS.intersection(frame.columns) for frame in scores.values()
        ),
        "no_new_synthetic_evaluation_dataset": synthetic_paths_before == synthetic_paths_after,
        "source_datasets_unchanged": protected_before == protected_after,
        "serialized_scaler_and_model_reproduce_scores": serialized_artifacts_match,
        "model_metadata_consistent": (
            model_metadata["training_row_count"] == 336
            and model_metadata["training_source_sha256"] == inputs_before[_relative(input_paths[0])]
            and model_metadata["scaler_sha256"] == sha256_file(scaler_path)
            and model_metadata["model_sha256"] == sha256_file(model_path)
        ),
        "only_expected_model_artifacts_created": model_files == expected_model_files,
    }
    if not all(validation.values()):
        failed = [name for name, passed in validation.items() if not passed]
        raise RuntimeError(f"Day 16 validation failed: {failed}")

    summary = {
        "stage": "Day 16 training-only scaling and initial unsupervised anomaly detector",
        "feature_order": list(CORE_UNSUPERVISED_FEATURES),
        "scaler": review,
        "isolation_forest": {
            "parameters": detector.get_params(deep=False),
            "training_partition": "training",
            "training_shape": list(transformed["training"].shape),
            "score_convention": config["scoring"],
            "final_threshold_selected": False,
        },
        "partition_score_statistics": statistics,
        "distribution_shift_review": {
            "calibration_mean_score_higher_than_training": statistics["calibration"]["mean"] > statistics["training"]["mean"],
            "evaluation_mean_score_higher_than_training": statistics["evaluation_baseline"]["mean"] > statistics["training"]["mean"],
            "interpretation": (
                "Differences are descriptive and can reflect time-of-day and operating-regime shift in the short chronological record. "
                "Calibration/evaluation are not added to model or scaler fitting."
            ),
        },
        "recurring_transition_scores": transitions,
        "artifacts": {
            "scaler": {"path": _relative(scaler_path), "sha256": sha256_file(scaler_path)},
            "model": {"path": _relative(model_path), "sha256": sha256_file(model_path)},
            "metadata": {"path": _relative(metadata_path), "sha256": sha256_file(metadata_path)},
            "scaled_matrices": {
                name: {"path": _relative(scaled_directory / SCALED_FILENAMES[name]), "sha256": sha256_file(scaled_directory / SCALED_FILENAMES[name])}
                for name in PARTITION_ORDER
            },
            "score_files": {
                name: {"path": _relative(outputs_directory / SCORE_FILENAMES[name]), "sha256": sha256_file(outputs_directory / SCORE_FILENAMES[name])}
                for name in PARTITION_ORDER
            },
            "figure": _relative(Path(figure_path)),
        },
        "source_hashes_before_and_after": {
            path: {"before": digest, "after": protected_after[path], "unchanged": protected_after[path] == digest}
            for path, digest in protected_before.items()
        },
        "validation": {"passed": True, "checks": validation},
        "explicitly_not_performed": [
            "final threshold calibration", "synthetic evaluation generation",
            "precision/recall/F1 calculation", "model comparison or tuning",
        ],
    }
    _atomic_json(summary, Path(summary_path))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--models-directory", type=Path, default=DEFAULT_MODELS_DIRECTORY)
    parser.add_argument("--scaled-directory", type=Path, default=DEFAULT_SCALED_DIRECTORY)
    parser.add_argument("--outputs-directory", type=Path, default=DEFAULT_OUTPUTS_DIRECTORY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_training(
            config_path=args.config, models_directory=args.models_directory,
            scaled_directory=args.scaled_directory, outputs_directory=args.outputs_directory,
            summary_path=args.summary, figure_path=args.figure,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, yaml.YAMLError) as error:
        print(f"Error: {error}")
        return 1
    print(f"Scaler fit rows: {report['scaler']['fit_row_count']}")
    print("Score means: " + ", ".join(f"{name}={item['mean']:.6f}" for name, item in report["partition_score_statistics"].items()))
    print(f"Validation passed: {report['validation']['passed']}")
    print("No final threshold, synthetic evaluation, labels, or performance metrics were created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
