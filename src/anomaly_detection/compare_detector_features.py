"""Compare four predeclared Isolation Forest feature sets without thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml

from src.anomaly_detection.investigate_baseline_scores import (
    CONTEXT_THRESHOLD_W_M2,
    load_saved_scores,
)
from src.anomaly_detection.train_baseline_detector import (
    DEFAULT_CONFIG_PATH as DAY16_CONFIG_PATH,
    DEFAULT_MODELS_DIRECTORY,
    DEFAULT_OUTPUTS_DIRECTORY,
    METADATA_FILENAME as DAY16_METADATA_FILENAME,
    MODEL_FILENAME as DAY16_MODEL_FILENAME,
    PARTITION_ORDER,
    SCALER_FILENAME as DAY16_SCALER_FILENAME,
    SYNTHETIC_LABEL_COLUMNS,
    TRANSITION_TIMESTAMPS,
    fit_detector,
    fit_training_scaler,
    load_core_partitions,
    load_model_config,
    score_partitions,
    transform_partitions,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_CONFIG = ROOT / "config/model_feature_comparison.yaml"
DEFAULT_COMPARISON_MODELS = ROOT / "models/comparisons"
DEFAULT_REPORT_PATH = ROOT / "outputs/model_feature_comparison.json"
DEFAULT_FIGURE_PATH = ROOT / "outputs/figures/operational/day18_model_feature_comparison.png"
VARIANT_ORDER = ("CORE_9", "NO_TIME_7", "NO_RTD_STD_8", "COMPACT_6")
EXPECTED_FEATURE_SETS = {
    "CORE_9": (
        "Power_Generated", "Solar_Radiation", "Air_Temp", "Relative_Humidity",
        "Wind_Speed", "rtd_mean", "rtd_std", "minute_of_day_sin", "minute_of_day_cos",
    ),
    "NO_TIME_7": (
        "Power_Generated", "Solar_Radiation", "Air_Temp", "Relative_Humidity",
        "Wind_Speed", "rtd_mean", "rtd_std",
    ),
    "NO_RTD_STD_8": (
        "Power_Generated", "Solar_Radiation", "Air_Temp", "Relative_Humidity",
        "Wind_Speed", "rtd_mean", "minute_of_day_sin", "minute_of_day_cos",
    ),
    "COMPACT_6": (
        "Power_Generated", "Solar_Radiation", "Air_Temp", "Relative_Humidity",
        "Wind_Speed", "rtd_mean",
    ),
}


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


def load_comparison_config(path: str | Path = DEFAULT_COMPARISON_CONFIG) -> dict[str, Any]:
    """Load the four predeclared variants and fixed selection rule."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Comparison configuration does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported feature-comparison configuration.")
    if tuple(config.get("feature_sets", {})) != VARIANT_ORDER:
        raise ValueError("Exactly four feature variants must be declared in the fixed order.")
    for variant in VARIANT_ORDER:
        if tuple(config["feature_sets"][variant].get("features", [])) != EXPECTED_FEATURE_SETS[variant]:
            raise ValueError(f"Feature order differs for {variant}.")
    day16 = load_model_config(DAY16_CONFIG_PATH)
    if config.get("isolation_forest_parameters") != day16["model"]["parameters"]:
        raise ValueError("Comparison model parameters must exactly match Day 16.")
    rule = config.get("selection_rule", {})
    weights = rule.get("components", {})
    if not np.isclose(sum(float(value) for value in weights.values()), 1.0):
        raise ValueError("Predeclared selection component weights must sum to one.")
    return config


def _score_summary(values: pd.Series) -> dict[str, float | int]:
    return {
        "count": len(values), "mean": float(values.mean()), "median": float(values.median()),
        "population_standard_deviation": float(values.std(ddof=0)),
        "90th_percentile": float(values.quantile(0.90)),
        "95th_percentile": float(values.quantile(0.95)),
        "99th_percentile": float(values.quantile(0.99)),
    }


def _context_summary(
    partition: pd.DataFrame, score: pd.Series
) -> dict[str, dict[str, float | int]]:
    low = partition.Solar_Radiation.le(CONTEXT_THRESHOLD_W_M2).to_numpy()
    return {
        "daylight_like": _score_summary(score.loc[~low]),
        "low_light": _score_summary(score.loc[low]),
    }


def _maximum_window_count(timestamps: pd.Series, minutes: int = 60) -> int:
    ordered = timestamps.sort_values().reset_index(drop=True)
    maximum = 0
    right = 0
    for left, start in enumerate(ordered):
        while right < len(ordered) and ordered.iloc[right] <= start + pd.Timedelta(minutes=minutes):
            right += 1
        maximum = max(maximum, right - left)
    return maximum


def high_score_review(
    calibration: pd.DataFrame, calibration_scores: pd.DataFrame
) -> dict[str, Any]:
    joined = calibration.copy()
    joined["anomaly_score"] = calibration_scores.anomaly_score.to_numpy()
    joined["light_context"] = np.where(
        joined.Solar_Radiation.le(CONTEXT_THRESHOLD_W_M2), "low_light", "daylight_like"
    )
    highest = joined.nlargest(10, "anomaly_score").sort_values("anomaly_score", ascending=False)
    records = [
        {
            "timestamp": _timestamp(row.Timestamp), "anomaly_score": float(row.anomaly_score),
            "light_context": row.light_context, "Power_Generated": float(row.Power_Generated),
            "Solar_Radiation": float(row.Solar_Radiation), "Air_Temp": float(row.Air_Temp),
            "Relative_Humidity": float(row.Relative_Humidity), "Wind_Speed": float(row.Wind_Speed),
            "rtd_mean": float(row.rtd_mean), "rtd_std": float(row.rtd_std),
            "interpretation": "high-score calibration baseline observation; not an anomaly or fault label",
        }
        for _, row in highest.iterrows()
    ]
    chronological = highest.Timestamp.sort_values()
    return {
        "top_10": records,
        "earliest_timestamp": _timestamp(chronological.iloc[0]),
        "latest_timestamp": _timestamp(chronological.iloc[-1]),
        "overall_span_hours": float((chronological.iloc[-1] - chronological.iloc[0]).total_seconds() / 3600),
        "maximum_top_10_observations_in_any_60_minute_window": _maximum_window_count(chronological),
        "daylight_like_count": int(highest.light_context.eq("daylight_like").sum()),
        "low_light_count": int(highest.light_context.eq("low_light").sum()),
    }


def transition_result(training_scores: pd.DataFrame) -> dict[str, Any]:
    transition = TRANSITION_TIMESTAMPS[0]
    match = training_scores.loc[training_scores.Timestamp.eq(transition)]
    if len(match) != 1:
        raise ValueError("The training 17:00 transition must occur exactly once.")
    index = match.index[0]
    values = training_scores.anomaly_score
    return {
        "timestamp": _timestamp(transition),
        "anomaly_score": float(match.iloc[0].anomaly_score),
        "highest_score_rank_in_training": int(values.rank(method="min", ascending=False).loc[index]),
        "score_percentile_in_training": float(values.le(match.iloc[0].anomaly_score).mean() * 100),
        "classified_as_fault": False,
    }


def comparison_evidence(
    variant: str, features: tuple[str, ...], partitions: dict[str, pd.DataFrame],
    scaled: dict[str, np.ndarray], scores: dict[str, pd.DataFrame],
    rule: dict[str, Any],
) -> dict[str, Any]:
    training_stats = _score_summary(scores["training"].anomaly_score)
    calibration_stats = _score_summary(scores["calibration"].anomaly_score)
    training_std = float(training_stats["population_standard_deviation"])
    shift = float(calibration_stats["mean"] - training_stats["mean"])
    contexts = {
        name: _context_summary(partitions[name], scores[name].anomaly_score)
        for name in ("training", "calibration")
    }
    context_shifts = {
        context: float(
            (contexts["calibration"][context]["mean"] - contexts["training"][context]["mean"])
            / training_std
        )
        for context in ("daylight_like", "low_light")
    }
    limited = [feature for feature in rule["unsupported_or_coverage_limited_features"] if feature in features]
    limited_indices = [features.index(feature) for feature in limited]
    extreme_fraction = (
        float((np.abs(scaled["calibration"][:, limited_indices]) > float(rule["unsupported_extreme_boundary"])).any(axis=1).mean())
        if limited_indices else 0.0
    )
    components = {
        "absolute_calibration_shift_z": abs(shift / training_std),
        "mean_absolute_context_shift_z": float(np.mean(np.abs(list(context_shifts.values())))),
        "unsupported_extreme_fraction": extreme_fraction,
        "compactness_penalty": float(
            (len(features) - int(rule["compactness_minimum_features"]))
            / (int(rule["compactness_reference_features"]) - int(rule["compactness_minimum_features"]))
        ),
    }
    weights = rule["components"]
    objective = (
        components["absolute_calibration_shift_z"] * float(weights["absolute_calibration_shift_z_weight"])
        + components["mean_absolute_context_shift_z"] * float(weights["mean_context_shift_z_weight"])
        + components["unsupported_extreme_fraction"] * float(weights["unsupported_extreme_fraction_weight"])
        + components["compactness_penalty"] * float(weights["compactness_penalty_weight"])
    )
    eligible = bool(
        training_std >= float(rule["eligible_score_std_minimum"])
        and float(training_stats["99th_percentile"] - training_stats["90th_percentile"])
        >= float(rule["eligible_p99_minus_p90_minimum"])
    )
    return {
        "variant": variant, "feature_order": list(features),
        "training_score_summary": training_stats,
        "calibration_score_summary": calibration_stats,
        "calibration_minus_training_mean": shift,
        "calibration_shift_in_training_score_std": shift / training_std,
        "context_score_summaries": contexts,
        "context_shift_in_training_score_std": context_shifts,
        "selection_components": components,
        "predeclared_objective": float(objective),
        "nontrivial_score_spread_guard_passed": eligible,
        "high_score_calibration_review": high_score_review(partitions["calibration"], scores["calibration"]),
        "training_17_00_transition": transition_result(scores["training"]),
    }


def select_recommendation(
    training_calibration_evidence: dict[str, dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Select using only predeclared training/calibration evidence."""
    eligible = [variant for variant in VARIANT_ORDER if training_calibration_evidence[variant]["nontrivial_score_spread_guard_passed"]]
    if not eligible:
        raise RuntimeError("No variant passes the non-trivial score-spread guard.")
    declaration_index = {variant: index for index, variant in enumerate(VARIANT_ORDER)}
    selected = min(
        eligible,
        key=lambda variant: (
            training_calibration_evidence[variant]["predeclared_objective"],
            len(training_calibration_evidence[variant]["feature_order"]),
            declaration_index[variant],
        ),
    )
    return {
        "recommended_variant": selected,
        "feature_order": training_calibration_evidence[selected]["feature_order"],
        "selection_scope": "training and calibration only",
        "evaluation_baseline_consulted": False,
        "criterion": "minimum predeclared weighted objective among variants passing non-trivial score-spread guards",
        "selected_objective": training_calibration_evidence[selected]["predeclared_objective"],
        "rationale": (
            "Selected from predeclared shift, context-stability, unsupported-extremeness, and compactness components; "
            "not from absolute calibration score or evaluation-baseline behavior."
        ),
    }


def top_timestamp_overlap(evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sets = {
        variant: {row["timestamp"] for row in evidence[variant]["high_score_calibration_review"]["top_10"]}
        for variant in VARIANT_ORDER
    }
    pairwise: dict[str, Any] = {}
    for left, right in combinations(VARIANT_ORDER, 2):
        intersection = sets[left] & sets[right]
        union = sets[left] | sets[right]
        pairwise[f"{left}__{right}"] = {
            "overlap_count": len(intersection), "jaccard": len(intersection) / len(union),
            "shared_timestamps": sorted(intersection),
        }
    common = set.intersection(*(sets[variant] for variant in VARIANT_ORDER))
    return {"pairwise": pairwise, "common_to_all_four_count": len(common), "common_to_all_four": sorted(common)}


def save_figure(evidence: dict[str, dict[str, Any]], destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = ["#2f6690", "#e6a23c", "#7a5195", "#5b9a62"]
    for axis, partition, title in zip(axes, ("training", "calibration"), ("Training", "Calibration baseline")):
        values = [evidence[variant][f"{partition}_score_summary"] for variant in VARIANT_ORDER]
        # Reconstruct box statistics from stored score series attached for plotting.
        raw = [evidence[variant]["_plot_scores"][partition] for variant in VARIANT_ORDER]
        boxes = axis.boxplot(
            raw, tick_labels=VARIANT_ORDER, patch_artist=True, showfliers=False
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        axis.set_title(title)
        axis.set_xlabel("Predeclared feature variant")
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Anomaly score = -score_samples (no threshold)")
    fig.suptitle("Controlled feature comparison on training and calibration only")
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


def run_comparison(
    *, comparison_config_path: Path = DEFAULT_COMPARISON_CONFIG,
    comparison_models_directory: Path = DEFAULT_COMPARISON_MODELS,
    report_path: Path = DEFAULT_REPORT_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
) -> dict[str, Any]:
    config = load_comparison_config(comparison_config_path)
    day16_config = load_model_config(DAY16_CONFIG_PATH)
    partitions = load_core_partitions(day16_config)
    saved_day16_scores = load_saved_scores(partitions, DEFAULT_OUTPUTS_DIRECTORY)
    protected_paths = [
        *sorted(path for path in (ROOT / "data").rglob("*") if path.is_file()),
        *sorted(path for path in DEFAULT_MODELS_DIRECTORY.glob("*") if path.is_file()),
        *sorted(DEFAULT_OUTPUTS_DIRECTORY.glob("anomaly_scores_*.csv")),
    ]
    protected_before = _snapshot(protected_paths)
    synthetic_before = sorted(_relative(path) for path in (ROOT / "data").rglob("*synthetic*"))
    parameters = config["isolation_forest_parameters"]
    runtime: dict[str, Any] = {}
    artifact_metadata: dict[str, Any] = {}
    deterministic: dict[str, bool] = {}
    comparison_models_directory = Path(comparison_models_directory)

    # Variant A is the frozen Day 16 reference and is never retrained.
    day16_scaler_path = DEFAULT_MODELS_DIRECTORY / DAY16_SCALER_FILENAME
    day16_model_path = DEFAULT_MODELS_DIRECTORY / DAY16_MODEL_FILENAME
    day16_metadata_path = DEFAULT_MODELS_DIRECTORY / DAY16_METADATA_FILENAME
    day16_scaler = joblib.load(day16_scaler_path)
    day16_model = joblib.load(day16_model_path)
    core_features = EXPECTED_FEATURE_SETS["CORE_9"]
    core_scaled, _ = transform_partitions(partitions, day16_scaler, core_features)
    core_scored = score_partitions(partitions, core_scaled, day16_model)
    if not all(np.allclose(core_scored[name].anomaly_score, saved_day16_scores[name].anomaly_score, rtol=0, atol=1e-15) for name in PARTITION_ORDER):
        raise RuntimeError("Frozen CORE_9 artifacts do not reproduce Day 16 scores.")
    runtime["CORE_9"] = {"scaler": day16_scaler, "model": day16_model, "scaled": core_scaled, "scores": saved_day16_scores}
    deterministic["CORE_9"] = True
    artifact_metadata["CORE_9"] = {
        "reused_day16_artifacts": True,
        "scaler_path": _relative(day16_scaler_path), "scaler_sha256": sha256_file(day16_scaler_path),
        "model_path": _relative(day16_model_path), "model_sha256": sha256_file(day16_model_path),
        "metadata_path": _relative(day16_metadata_path), "metadata_sha256": sha256_file(day16_metadata_path),
    }

    for variant in VARIANT_ORDER[1:]:
        specification = config["feature_sets"][variant]
        features = tuple(specification["features"])
        scaler = fit_training_scaler(partitions["training"], features)
        scaled, _ = transform_partitions(partitions, scaler, features)
        model = fit_detector(scaled["training"], parameters)
        scores = score_partitions(partitions, scaled, model)
        repeated = fit_detector(scaled["training"], parameters)
        repeated_scores = score_partitions(partitions, scaled, repeated)
        deterministic[variant] = all(
            np.array_equal(scores[name].anomaly_score, repeated_scores[name].anomaly_score)
            for name in ("training", "calibration")
        )
        stem = specification["artifact_stem"]
        scaler_path = comparison_models_directory / f"{stem}_standard_scaler.joblib"
        model_path = comparison_models_directory / f"{stem}_isolation_forest.joblib"
        metadata_path = comparison_models_directory / f"{stem}_metadata.json"
        _atomic_joblib(scaler, scaler_path)
        _atomic_joblib(model, model_path)
        metadata = {
            "variant": variant, "feature_order": list(features),
            "training_rows": len(partitions["training"]),
            "training_timestamp_start": _timestamp(partitions["training"].Timestamp.iloc[0]),
            "training_timestamp_end": _timestamp(partitions["training"].Timestamp.iloc[-1]),
            "timestamp_used_as_feature": False, "model_type": "IsolationForest",
            "model_parameters": model.get_params(deep=False), "random_state": parameters["random_state"],
            "training_source_sha256": sha256_file(ROOT / day16_config["inputs"]["training"]),
            "comparison_config_sha256": sha256_file(comparison_config_path),
            "scaler_path": _relative(scaler_path), "scaler_sha256": sha256_file(scaler_path),
            "model_path": _relative(model_path), "model_sha256": sha256_file(model_path),
            "synthetic_data_used": False, "threshold_selected": False,
            "package_versions": {
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
            },
        }
        _atomic_json(metadata, metadata_path)
        artifact_metadata[variant] = {
            "reused_day16_artifacts": False,
            "scaler_path": _relative(scaler_path), "scaler_sha256": sha256_file(scaler_path),
            "model_path": _relative(model_path), "model_sha256": sha256_file(model_path),
            "metadata_path": _relative(metadata_path), "metadata_sha256": sha256_file(metadata_path),
        }
        runtime[variant] = {"scaler": scaler, "model": model, "scaled": scaled, "scores": scores}

    evidence: dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        features = EXPECTED_FEATURE_SETS[variant]
        limited_scores = {name: runtime[variant]["scores"][name] for name in ("training", "calibration")}
        evidence[variant] = comparison_evidence(
            variant, features, partitions, runtime[variant]["scaled"], limited_scores,
            config["selection_rule"],
        )
        evidence[variant]["_plot_scores"] = {
            name: limited_scores[name].anomaly_score.tolist() for name in ("training", "calibration")
        }

    # This recommendation is completed and hashed before evaluation scoring below.
    recommendation = select_recommendation(evidence, config)
    selection_record = {
        "criteria": config["selection_rule"],
        "training_calibration_metrics": {
            variant: {
                "predeclared_objective": evidence[variant]["predeclared_objective"],
                "components": evidence[variant]["selection_components"],
                "eligible": evidence[variant]["nontrivial_score_spread_guard_passed"],
            }
            for variant in VARIANT_ORDER
        },
        "recommendation": recommendation,
    }
    selection_digest = hashlib.sha256(
        json.dumps(selection_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    # Evaluation is descriptive and deliberately occurs after selection is frozen.
    evaluation_documentation = {
        variant: {
            "score_summary": _score_summary(runtime[variant]["scores"]["evaluation_baseline"].anomaly_score),
            "not_used_for_selection": True,
        }
        for variant in VARIANT_ORDER
    }
    overlap = top_timestamp_overlap(evidence)
    save_figure(evidence, Path(figure_path))
    for variant in VARIANT_ORDER:
        evidence[variant].pop("_plot_scores")

    protected_after = _snapshot(protected_paths)
    synthetic_after = sorted(_relative(path) for path in (ROOT / "data").rglob("*synthetic*"))
    expected_comparison_files = sorted(
        f"{config['feature_sets'][variant]['artifact_stem']}_{suffix}"
        for variant in VARIANT_ORDER[1:]
        for suffix in ("standard_scaler.joblib", "isolation_forest.joblib", "metadata.json")
    )
    actual_comparison_files = sorted(path.name for path in comparison_models_directory.iterdir() if path.is_file())
    validation = {
        "exactly_four_declared_variants": tuple(config["feature_sets"]) == VARIANT_ORDER,
        "identical_336_training_rows_for_all_variants": all(runtime[v]["scaled"]["training"].shape[0] == 336 for v in VARIANT_ORDER),
        "all_new_scalers_fit_training_only": all(int(runtime[v]["scaler"].n_samples_seen_) == 336 for v in VARIANT_ORDER[1:]),
        "identical_isolation_forest_parameters": all(
            all(runtime[v]["model"].get_params()[key] == value for key, value in parameters.items()) for v in VARIANT_ORDER
        ),
        "deterministic_results": all(deterministic.values()),
        "correct_feature_order_and_no_timestamp": all(
            runtime[v]["scaled"]["training"].shape[1] == len(EXPECTED_FEATURE_SETS[v]) and "Timestamp" not in EXPECTED_FEATURE_SETS[v]
            for v in VARIANT_ORDER
        ),
        "existing_day16_artifacts_unchanged": all(
            protected_before[_relative(path)] == protected_after[_relative(path)]
            for path in (day16_scaler_path, day16_model_path, day16_metadata_path)
        ),
        "no_synthetic_data_used_or_created": synthetic_before == synthetic_after,
        "no_threshold_or_anomaly_labels_created": all(
            not SYNTHETIC_LABEL_COLUMNS.intersection(frame.columns)
            for variant in VARIANT_ORDER for frame in runtime[variant]["scores"].values()
        ),
        "recommendation_uses_training_calibration_only": recommendation["evaluation_baseline_consulted"] is False,
        "recommendation_recorded_before_evaluation_scoring": bool(selection_digest),
        "all_protected_source_files_unchanged": protected_before == protected_after,
        "only_predeclared_comparison_artifacts_created": actual_comparison_files == expected_comparison_files,
    }
    if not all(validation.values()):
        failed = [name for name, value in validation.items() if not value]
        raise RuntimeError(f"Day 18 validation failed: {failed}")

    report = {
        "stage": "Day 18 controlled anomaly-detector feature comparison",
        "comparison_scope": "prototype model comparison; not final model evaluation",
        "feature_sets": {variant: list(EXPECTED_FEATURE_SETS[variant]) for variant in VARIANT_ORDER},
        "controlled_training_procedure": {
            "training_rows": 336, "scaler_fit_scope": "training only",
            "model_fit_scope": "training only", "timestamp_excluded": True,
            "isolation_forest_parameters": parameters, "score_convention": "higher is more isolated",
        },
        "artifact_metadata": artifact_metadata,
        "training_calibration_results": evidence,
        "calibration_top_10_overlap": overlap,
        "predeclared_selection": {
            "rule": config["selection_rule"], "selection_record_sha256": selection_digest,
            "recommendation": recommendation,
            "locked_before_evaluation_baseline_scoring": True,
        },
        "optional_evaluation_baseline_documentation": evaluation_documentation,
        "evaluation_baseline_excluded_from_selection": True,
        "protected_artifact_hashes": {
            path: {"before": digest, "after": protected_after[path], "unchanged": protected_after[path] == digest}
            for path, digest in protected_before.items()
        },
        "validation": {"passed": True, "checks": validation},
        "explicitly_not_performed": [
            "threshold selection", "anomaly label creation", "synthetic evaluation generation",
            "synthetic precision/recall/F1", "hyperparameter tuning", "chronological split changes",
        ],
        "limitation": "Approximately 33.6 hours and one complete calendar day support only a prototype configuration comparison, not general validation.",
    }
    _atomic_json(report, Path(report_path))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-config", type=Path, default=DEFAULT_COMPARISON_CONFIG)
    parser.add_argument("--comparison-models", type=Path, default=DEFAULT_COMPARISON_MODELS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_comparison(
            comparison_config_path=args.comparison_config,
            comparison_models_directory=args.comparison_models,
            report_path=args.report, figure_path=args.figure,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, yaml.YAMLError) as error:
        print(f"Error: {error}")
        return 1
    print(f"Recommended feature set: {report['predeclared_selection']['recommendation']['recommended_variant']}")
    print(f"Validation passed: {report['validation']['passed']}")
    print("No threshold, anomaly labels, synthetic evaluation, or performance metrics were created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
