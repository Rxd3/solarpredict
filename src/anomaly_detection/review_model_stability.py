"""Review NO_TIME_7 seed stability and freeze it only if fixed gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
import yaml

from src.anomaly_detection.compare_detector_features import (
    DEFAULT_COMPARISON_MODELS,
    DEFAULT_REPORT_PATH as DAY18_REPORT_PATH,
    EXPECTED_FEATURE_SETS,
)
from src.anomaly_detection.train_baseline_detector import (
    DEFAULT_CONFIG_PATH as DAY16_CONFIG_PATH,
    DEFAULT_MODELS_DIRECTORY,
    MODEL_FILENAME as DAY16_MODEL_FILENAME,
    SCALER_FILENAME as DAY16_SCALER_FILENAME,
    SYNTHETIC_LABEL_COLUMNS,
    fit_detector,
    fit_training_scaler,
    load_core_partitions,
    load_model_config,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STABILITY_CONFIG = ROOT / "config/model_stability_review.yaml"
DEFAULT_FROZEN_CONFIG = ROOT / "config/frozen_anomaly_detector.yaml"
DEFAULT_FROZEN_METADATA = ROOT / "models/frozen_anomaly_detector_metadata.json"
DEFAULT_REPORT_PATH = ROOT / "outputs/model_stability_review.json"
DEFAULT_FIGURE_PATH = ROOT / "outputs/figures/operational/day19_seed_stability.png"
DECLARED_SEEDS = (7, 21, 42, 84, 123)
FEATURE_ORDER = EXPECTED_FEATURE_SETS["NO_TIME_7"]
BASE_OPERATIONAL_FEATURES = EXPECTED_FEATURE_SETS["COMPACT_6"]
NO_TIME_SCALER = DEFAULT_COMPARISON_MODELS / "no_time_standard_scaler.joblib"
NO_TIME_MODEL = DEFAULT_COMPARISON_MODELS / "no_time_isolation_forest.joblib"
NO_TIME_METADATA = DEFAULT_COMPARISON_MODELS / "no_time_metadata.json"
COMPACT_SCALER = DEFAULT_COMPARISON_MODELS / "compact_standard_scaler.joblib"
COMPACT_MODEL = DEFAULT_COMPARISON_MODELS / "compact_isolation_forest.joblib"


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


def _atomic_text(text: str, destination: Path, suffix: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=suffix,
            dir=destination.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(value: dict[str, Any], destination: Path) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    _atomic_text(text, destination, ".json.tmp")


def load_stability_config(path: str | Path = DEFAULT_STABILITY_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Stability configuration does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported stability-review configuration.")
    if tuple(config.get("review_seeds", [])) != DECLARED_SEEDS:
        raise ValueError("Exactly the five declared seeds must be used in order.")
    if tuple(config.get("feature_order", [])) != FEATURE_ORDER:
        raise ValueError("Stability feature order must equal NO_TIME_7.")
    day16 = load_model_config(DAY16_CONFIG_PATH)
    if config.get("isolation_forest_parameters") != day16["model"]["parameters"]:
        raise ValueError("Stability parameters must equal Day 16 before seed substitution.")
    if config.get("production_prototype_seed") != 42:
        raise ValueError("The reproducible prototype seed must remain 42.")
    return config


def _score_frame(partition: pd.DataFrame, scaled: np.ndarray, model: Any) -> pd.DataFrame:
    scores = -model.score_samples(scaled)
    if not np.isfinite(scores).all():
        raise RuntimeError("A seed-specific model produced non-finite scores.")
    return pd.DataFrame({"Timestamp": partition.Timestamp.to_numpy(), "anomaly_score": scores})


def _score_summary(values: pd.Series) -> dict[str, float | int]:
    return {
        "count": len(values), "mean": float(values.mean()),
        "population_standard_deviation": float(values.std(ddof=0)),
        "95th_percentile": float(values.quantile(0.95)),
        "calculation_scope": "continuous baseline scores; no threshold or labels",
    }


def reproduce_day18_no_time(
    partitions: dict[str, pd.DataFrame], day18: dict[str, Any]
) -> tuple[Any, Any, dict[str, np.ndarray], dict[str, pd.DataFrame], dict[str, Any]]:
    """Load rather than refit the selected seed-42 Day 18 artifacts."""
    scaler = joblib.load(NO_TIME_SCALER)
    model = joblib.load(NO_TIME_MODEL)
    scaled = {
        name: scaler.transform(partitions[name].loc[:, FEATURE_ORDER].to_numpy(dtype=float))
        for name in ("training", "calibration")
    }
    scores = {
        name: _score_frame(partitions[name], scaled[name], model)
        for name in ("training", "calibration")
    }
    recorded = day18["training_calibration_results"]["NO_TIME_7"]
    checks: dict[str, Any] = {}
    for name in ("training", "calibration"):
        current = _score_summary(scores[name].anomaly_score)
        expected = recorded[f"{name}_score_summary"]
        checks[name] = {
            "mean_difference": current["mean"] - expected["mean"],
            "std_difference": current["population_standard_deviation"] - expected["population_standard_deviation"],
            "p95_difference": current["95th_percentile"] - expected["95th_percentile"],
            "summary_matches": all(
                math.isclose(float(current[key]), float(expected[key]), abs_tol=1e-15)
                for key in ("mean", "population_standard_deviation", "95th_percentile")
            ),
        }
    current_top = scores["calibration"].nlargest(10, "anomaly_score").Timestamp.map(_timestamp).tolist()
    recorded_top = [
        row["timestamp"] for row in recorded["high_score_calibration_review"]["top_10"]
    ]
    checks["top_10_timestamps_match"] = current_top == recorded_top
    checks["artifact_hashes_match_day18"] = bool(
        sha256_file(NO_TIME_SCALER) == day18["artifact_metadata"]["NO_TIME_7"]["scaler_sha256"]
        and sha256_file(NO_TIME_MODEL) == day18["artifact_metadata"]["NO_TIME_7"]["model_sha256"]
    )
    return scaler, model, scaled, scores, checks


def run_seed_models(
    partitions: dict[str, pd.DataFrame], config: dict[str, Any]
) -> tuple[dict[int, Any], dict[int, dict[str, np.ndarray]], dict[int, dict[str, pd.DataFrame]], dict[int, Any]]:
    """Fit only in-memory stability models on identical training rows."""
    models: dict[int, Any] = {}
    transformed: dict[int, dict[str, np.ndarray]] = {}
    scores: dict[int, dict[str, pd.DataFrame]] = {}
    results: dict[int, Any] = {}
    base_parameters = config["isolation_forest_parameters"]
    for seed in DECLARED_SEEDS:
        scaler = fit_training_scaler(partitions["training"], FEATURE_ORDER)
        current_scaled = {
            name: scaler.transform(partitions[name].loc[:, FEATURE_ORDER].to_numpy(dtype=float))
            for name in ("training", "calibration")
        }
        parameters = {**base_parameters, "random_state": seed}
        model = fit_detector(current_scaled["training"], parameters)
        current_scores = {
            name: _score_frame(partitions[name], current_scaled[name], model)
            for name in ("training", "calibration")
        }
        training_summary = _score_summary(current_scores["training"].anomaly_score)
        calibration_summary = _score_summary(current_scores["calibration"].anomaly_score)
        shift = float(calibration_summary["mean"] - training_summary["mean"])
        transition = pd.Timestamp("2022-04-27 17:00:00")
        transition_index = current_scores["training"].index[
            current_scores["training"].Timestamp.eq(transition)
        ][0]
        transition_score = float(current_scores["training"].loc[transition_index, "anomaly_score"])
        results[seed] = {
            "feature_order": list(FEATURE_ORDER),
            "training_rows": int(scaler.n_samples_seen_),
            "scaler_fit_scope": "training only",
            "model_parameters": parameters,
            "training_score_summary": training_summary,
            "calibration_score_summary": calibration_summary,
            "calibration_minus_training_mean": shift,
            "standardized_calibration_shift": shift / float(training_summary["population_standard_deviation"]),
            "training_17_00_transition": {
                "anomaly_score": transition_score,
                "highest_score_rank": int(
                    current_scores["training"].anomaly_score.rank(method="min", ascending=False).loc[transition_index]
                ),
                "score_percentile": float(
                    current_scores["training"].anomaly_score.le(transition_score).mean() * 100
                ),
                "interpretation": "recurring operational transition; not a fault label",
            },
        }
        models[seed] = model
        transformed[seed] = current_scaled
        scores[seed] = current_scores
    return models, transformed, scores, results


def rank_stability(
    seed_scores: dict[int, dict[str, pd.DataFrame]]
) -> dict[str, Any]:
    pairwise: dict[str, Any] = {}
    spearman_values: list[float] = []
    top_10_fractions: list[float] = []
    top_20_fractions: list[float] = []
    for left, right in combinations(DECLARED_SEEDS, 2):
        left_scores = seed_scores[left]["calibration"]
        right_scores = seed_scores[right]["calibration"]
        correlation = float(left_scores.anomaly_score.corr(right_scores.anomaly_score, method="spearman"))
        left_10 = set(left_scores.nlargest(10, "anomaly_score").Timestamp)
        right_10 = set(right_scores.nlargest(10, "anomaly_score").Timestamp)
        left_20 = set(left_scores.nlargest(20, "anomaly_score").Timestamp)
        right_20 = set(right_scores.nlargest(20, "anomaly_score").Timestamp)
        top_10 = len(left_10 & right_10)
        top_20 = len(left_20 & right_20)
        pairwise[f"{left}__{right}"] = {
            "spearman_rank_correlation": correlation,
            "top_10_overlap_count": top_10,
            "top_10_overlap_fraction": top_10 / 10,
            "top_20_overlap_count": top_20,
            "top_20_overlap_fraction": top_20 / 20,
        }
        spearman_values.append(correlation)
        top_10_fractions.append(top_10 / 10)
        top_20_fractions.append(top_20 / 20)
    return {
        "pairwise": pairwise,
        "minimum_spearman": min(spearman_values),
        "median_spearman": float(np.median(spearman_values)),
        "mean_spearman": float(np.mean(spearman_values)),
        "minimum_top_10_overlap_fraction": min(top_10_fractions),
        "mean_top_10_overlap_fraction": float(np.mean(top_10_fractions)),
        "minimum_top_20_overlap_fraction": min(top_20_fractions),
        "mean_top_20_overlap_fraction": float(np.mean(top_20_fractions)),
    }


def extreme_rtd_review(
    partitions: dict[str, pd.DataFrame], no_time_scaler: Any,
    no_time_scores: pd.DataFrame, compact_scores: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration = partitions["calibration"]
    scaled = no_time_scaler.transform(calibration.loc[:, FEATURE_ORDER].to_numpy(dtype=float))
    rtd_index = FEATURE_ORDER.index("rtd_std")
    mask = np.abs(scaled[:, rtd_index]) > 5
    positions = np.flatnonzero(mask)
    if len(positions) != 16:
        raise RuntimeError(f"Expected 16 extreme-rtd rows, found {len(positions)}.")
    no_time_rank = no_time_scores.anomaly_score.rank(method="min", ascending=False)
    compact_rank = compact_scores.anomaly_score.rank(method="min", ascending=False)
    records: list[dict[str, Any]] = []
    for position in positions:
        row = calibration.loc[position]
        score = float(no_time_scores.loc[position, "anomaly_score"])
        records.append({
            "timestamp": _timestamp(row.Timestamp), "raw_rtd_std": float(row.rtd_std),
            "training_scaled_rtd_std": float(scaled[position, rtd_index]),
            "no_time_7_anomaly_score": score,
            "no_time_7_calibration_score_percentile": float(no_time_scores.anomaly_score.le(score).mean() * 100),
            "no_time_7_rank": int(no_time_rank.loc[position]),
            "compact_6_anomaly_score": float(compact_scores.loc[position, "anomaly_score"]),
            "compact_6_rank": int(compact_rank.loc[position]),
            "absolute_rank_change_without_rtd_std": int(abs(compact_rank.loc[position] - no_time_rank.loc[position])),
            "Solar_Radiation": float(row.Solar_Radiation), "Power_Generated": float(row.Power_Generated),
            "rtd_mean": float(row.rtd_mean),
            "interpretation": "extreme training-scaled RTD spread; not a physical or anomaly label",
        })
    top_10_cutoff = math.ceil(len(calibration) * 0.10)
    top_5_cutoff = math.ceil(len(calibration) * 0.05)
    rtd_summary = {
        "definition": "absolute training-scaled rtd_std > 5",
        "count": len(records),
        "records": records,
        "no_time_7_count_in_calibration_top_10_percent": sum(row["no_time_7_rank"] <= top_10_cutoff for row in records),
        "no_time_7_count_in_calibration_top_5_percent": sum(row["no_time_7_rank"] <= top_5_cutoff for row in records),
        "no_time_7_count_in_top_10_observations": sum(row["no_time_7_rank"] <= 10 for row in records),
        "high_rtd_std_automatically_implies_high_score": all(row["no_time_7_rank"] <= top_10_cutoff for row in records),
        "physical_interpretation_assigned": False,
    }
    all_rank_correlation = float(no_time_scores.anomaly_score.corr(compact_scores.anomaly_score, method="spearman"))
    sensitivity = {
        "calibration_score_spearman_no_time_7_vs_compact_6": all_rank_correlation,
        "median_absolute_rank_change_for_16_rows": float(np.median([row["absolute_rank_change_without_rtd_std"] for row in records])),
        "maximum_absolute_rank_change_for_16_rows": max(row["absolute_rank_change_without_rtd_std"] for row in records),
        "compact_6_count_in_calibration_top_10_percent": sum(row["compact_6_rank"] <= top_10_cutoff for row in records),
        "compact_6_count_in_calibration_top_5_percent": sum(row["compact_6_rank"] <= top_5_cutoff for row in records),
        "compact_6_count_in_top_10_observations": sum(row["compact_6_rank"] <= 10 for row in records),
    }
    return rtd_summary, sensitivity


def apply_freeze_criteria(
    config: dict[str, Any], rank: dict[str, Any], seed_results: dict[int, Any],
    rtd: dict[str, Any], day18: dict[str, Any],
) -> dict[str, Any]:
    criteria = config["freeze_criteria"]
    core_shift = float(day18["training_calibration_results"]["CORE_9"]["calibration_shift_in_training_score_std"])
    no_time_shift = float(day18["training_calibration_results"]["NO_TIME_7"]["calibration_shift_in_training_score_std"])
    compact_shift = float(day18["training_calibration_results"]["COMPACT_6"]["calibration_shift_in_training_score_std"])
    checks = {
        "rankings_broadly_stable": bool(
            rank["minimum_spearman"] >= float(criteria["minimum_pairwise_calibration_spearman"])
            and rank["mean_top_10_overlap_fraction"] >= float(criteria["minimum_mean_top_10_overlap_fraction"])
            and rank["mean_top_20_overlap_fraction"] >= float(criteria["minimum_mean_top_20_overlap_fraction"])
        ),
        "every_seed_shift_improves_on_core_9": all(
            abs(result["standardized_calibration_shift"]) < abs(core_shift)
            for result in seed_results.values()
        ),
        "rtd_std_does_not_fill_seed_42_top_10": rtd["no_time_7_count_in_top_10_observations"] <= int(criteria["maximum_extreme_rtd_rows_in_seed_42_top_10"]),
        "compact_and_interpretable": len(FEATURE_ORDER) == 7 and set(BASE_OPERATIONAL_FEATURES).issubset(FEATURE_ORDER),
        "compact_6_does_not_improve_standardized_shift": no_time_shift < compact_shift,
    }
    return {
        "predeclared_checks": checks,
        "passed": all(checks.values()),
        "decision": "freeze_NO_TIME_7_as_prototype" if all(checks.values()) else "defer_freeze",
        "core_9_reference_shift": core_shift,
        "day18_no_time_7_shift": no_time_shift,
        "day18_compact_6_shift": compact_shift,
        "selection_scope": "training and calibration only",
        "evaluation_baseline_used": False,
    }


def write_freeze_artifacts(
    config: dict[str, Any], decision: dict[str, Any], frozen_config_path: Path,
    frozen_metadata_path: Path,
) -> dict[str, Any]:
    if not decision["passed"]:
        return {"created": False, "reason": "predeclared freeze criteria did not all pass"}
    frozen = {
        "schema_version": 1,
        "status": "prototype_frozen",
        "configuration_name": "NO_TIME_7_standard_scaler_isolation_forest",
        "feature_order": list(FEATURE_ORDER),
        "scaler": {
            "type": "StandardScaler", "fit_scope": "training_only",
            "artifact": _relative(NO_TIME_SCALER),
        },
        "model": {
            "type": "IsolationForest",
            "parameters": config["isolation_forest_parameters"],
            "artifact": _relative(NO_TIME_MODEL),
        },
        "training_partition": {
            "rows": 336, "start": "2022-04-27 15:32:00", "end": "2022-04-28 02:42:00",
        },
        "calibration_partition": {
            "rows": 336, "start": "2022-04-28 02:44:00", "end": "2022-04-28 13:54:00",
        },
        "stability_review_seeds": list(DECLARED_SEEDS),
        "production_prototype_seed": 42,
        "known_limitations": config["limitations"],
        "rtd_uncertainty": "RTD placement and units are unconfirmed; calibration contains 16 rows with absolute training-scaled rtd_std above 5.",
    }
    _atomic_text(yaml.safe_dump(frozen, sort_keys=False), frozen_config_path, ".yaml.tmp")
    selected_metadata = {
        "status": "prototype_frozen",
        "selected_variant": "NO_TIME_7",
        "feature_order": list(FEATURE_ORDER),
        "production_prototype_seed": 42,
        "new_model_created": False,
        "selected_scaler": {"path": _relative(NO_TIME_SCALER), "sha256": sha256_file(NO_TIME_SCALER)},
        "selected_model": {"path": _relative(NO_TIME_MODEL), "sha256": sha256_file(NO_TIME_MODEL)},
        "selected_day18_metadata": {"path": _relative(NO_TIME_METADATA), "sha256": sha256_file(NO_TIME_METADATA)},
        "stability_config": {"path": _relative(DEFAULT_STABILITY_CONFIG), "sha256": sha256_file(DEFAULT_STABILITY_CONFIG)},
        "freeze_decision": decision,
    }
    _atomic_json(selected_metadata, frozen_metadata_path)
    return {
        "created": True,
        "frozen_config": {"path": _relative(frozen_config_path), "sha256": sha256_file(frozen_config_path)},
        "frozen_metadata": {"path": _relative(frozen_metadata_path), "sha256": sha256_file(frozen_metadata_path)},
        "selected_scaler": selected_metadata["selected_scaler"],
        "selected_model": selected_metadata["selected_model"],
        "new_model_created": False,
    }


def save_figure(seed_scores: dict[int, dict[str, pd.DataFrame]], destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    values = [seed_scores[seed]["calibration"].anomaly_score for seed in DECLARED_SEEDS]
    boxes = axis.boxplot(values, tick_labels=[str(seed) for seed in DECLARED_SEEDS], patch_artist=True, showfliers=False)
    colors = ["#5b8ff9", "#61d9a8", "#f6bd16", "#7262fd", "#e8684a"]
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.78)
    axis.set_title("NO_TIME_7 calibration score stability across declared seeds")
    axis.set_xlabel("Isolation Forest random state")
    axis.set_ylabel("Anomaly score = -score_samples (no threshold)")
    axis.grid(axis="y", alpha=0.2)
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


def run_stability_review(
    *, stability_config_path: Path = DEFAULT_STABILITY_CONFIG,
    frozen_config_path: Path = DEFAULT_FROZEN_CONFIG,
    frozen_metadata_path: Path = DEFAULT_FROZEN_METADATA,
    report_path: Path = DEFAULT_REPORT_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
) -> dict[str, Any]:
    config = load_stability_config(stability_config_path)
    day16_config = load_model_config(DAY16_CONFIG_PATH)
    partitions = load_core_partitions(day16_config)
    day18 = json.loads(DAY18_REPORT_PATH.read_text(encoding="utf-8"))
    protected_paths = [
        *sorted(path for path in (ROOT / "data").rglob("*") if path.is_file()),
        *sorted(path for path in DEFAULT_MODELS_DIRECTORY.rglob("*") if path.is_file() and path.resolve() != Path(frozen_metadata_path).resolve()),
        DAY18_REPORT_PATH,
        *sorted((ROOT / "outputs").glob("anomaly_scores_*.csv")),
    ]
    before = _snapshot(protected_paths)
    synthetic_before = sorted(_relative(path) for path in (ROOT / "data").rglob("*synthetic*"))

    selected_scaler, _, _, selected_scores, reproduction = reproduce_day18_no_time(partitions, day18)
    _, _, seed_scores, seed_results = run_seed_models(partitions, config)
    rank = rank_stability(seed_scores)
    compact_scaler = joblib.load(COMPACT_SCALER)
    compact_model = joblib.load(COMPACT_MODEL)
    compact_scaled = compact_scaler.transform(
        partitions["calibration"].loc[:, BASE_OPERATIONAL_FEATURES].to_numpy(dtype=float)
    )
    compact_scores = _score_frame(partitions["calibration"], compact_scaled, compact_model)
    rtd, sensitivity = extreme_rtd_review(
        partitions, selected_scaler, selected_scores["calibration"], compact_scores
    )
    decision = apply_freeze_criteria(config, rank, seed_results, rtd, day18)
    freeze_artifacts = write_freeze_artifacts(
        config, decision, Path(frozen_config_path), Path(frozen_metadata_path)
    )
    save_figure(seed_scores, Path(figure_path))

    after = _snapshot(protected_paths)
    synthetic_after = sorted(_relative(path) for path in (ROOT / "data").rglob("*synthetic*"))
    frozen_yaml = (
        yaml.safe_load(Path(frozen_config_path).read_text(encoding="utf-8"))
        if decision["passed"] else None
    )
    validation = {
        "exactly_five_declared_seeds": tuple(config["review_seeds"]) == DECLARED_SEEDS,
        "identical_training_rows_and_feature_order": all(
            result["training_rows"] == 336 and result["feature_order"] == list(FEATURE_ORDER)
            for result in seed_results.values()
        ),
        "parameters_identical_except_random_state": all(
            all(
                value == config["isolation_forest_parameters"][key]
                for key, value in result["model_parameters"].items() if key != "random_state"
            ) and result["model_parameters"]["random_state"] == seed
            for seed, result in seed_results.items()
        ),
        "all_scalers_fit_training_only": all(
            result["training_rows"] == 336 and result["scaler_fit_scope"] == "training only"
            for result in seed_results.values()
        ),
        "day18_selected_artifacts_reproduced": all(
            reproduction[name]["summary_matches"] for name in ("training", "calibration")
        ) and reproduction["top_10_timestamps_match"] and reproduction["artifact_hashes_match_day18"],
        "no_threshold_or_anomaly_labels_created": all(
            not SYNTHETIC_LABEL_COLUMNS.intersection(frame.columns)
            for seed in DECLARED_SEEDS for frame in seed_scores[seed].values()
        ) and (frozen_yaml is None or "threshold" not in frozen_yaml),
        "no_synthetic_evaluation_generated": synthetic_before == synthetic_after,
        "existing_day18_artifacts_unchanged": all(
            before[_relative(path)] == after[_relative(path)]
            for path in (NO_TIME_SCALER, NO_TIME_MODEL, NO_TIME_METADATA, COMPACT_SCALER, COMPACT_MODEL, DAY18_REPORT_PATH)
        ),
        "frozen_configuration_matches_no_time_7": bool(
            not decision["passed"] or (
                frozen_yaml["status"] == "prototype_frozen"
                and frozen_yaml["feature_order"] == list(FEATURE_ORDER)
                and frozen_yaml["model"]["parameters"] == config["isolation_forest_parameters"]
            )
        ),
        "frozen_artifact_hashes_correct": bool(
            not decision["passed"] or (
                freeze_artifacts["selected_scaler"]["sha256"] == sha256_file(NO_TIME_SCALER)
                and freeze_artifacts["selected_model"]["sha256"] == sha256_file(NO_TIME_MODEL)
            )
        ),
        "no_new_seed_model_files_created": freeze_artifacts.get("new_model_created") is False,
        "protected_source_datasets_and_models_unchanged": before == after,
        "freeze_decision_uses_training_calibration_only": decision["evaluation_baseline_used"] is False,
    }
    if not all(validation.values()):
        failed = [name for name, value in validation.items() if not value]
        raise RuntimeError(f"Day 19 validation failed: {failed}")

    report = {
        "stage": "Day 19 model stability review and prototype configuration freeze",
        "selected_variant_under_review": "NO_TIME_7",
        "declared_seeds": list(DECLARED_SEEDS),
        "day18_artifact_reproduction": reproduction,
        "seed_score_results": {str(seed): result for seed, result in seed_results.items()},
        "calibration_rank_stability": rank,
        "extreme_rtd_std_review": rtd,
        "no_time_7_vs_compact_6_extreme_rtd_sensitivity": {
            **sensitivity,
            "day18_no_time_7_standardized_shift": decision["day18_no_time_7_shift"],
            "day18_compact_6_standardized_shift": decision["day18_compact_6_shift"],
            "overall_stability_improved_by_removing_rtd_std": decision["day18_compact_6_shift"] < decision["day18_no_time_7_shift"],
            "interpretation": "Removing rtd_std changes rankings and shifts sensitivity toward correlated operating variables; it did not improve the Day 18 standardized calibration shift.",
        },
        "recurring_17_00_seed_stability": {
            str(seed): result["training_17_00_transition"] for seed, result in seed_results.items()
        },
        "freeze_criteria": {
            "configuration": config["freeze_criteria"],
            "application": decision,
        },
        "freeze_decision": decision,
        "frozen_artifact_references": freeze_artifacts,
        "protected_artifact_hashes": {
            path: {"before": digest, "after": after[path], "unchanged": after[path] == digest}
            for path, digest in before.items()
        },
        "validation": {"passed": True, "checks": validation},
        "explicitly_not_performed": [
            "threshold selection", "synthetic evaluation generation", "precision/recall/F1",
            "chronological split changes", "Isolation Forest hyperparameter tuning",
            "new persisted seed-specific models",
        ],
    }
    _atomic_json(report, Path(report_path))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability-config", type=Path, default=DEFAULT_STABILITY_CONFIG)
    parser.add_argument("--frozen-config", type=Path, default=DEFAULT_FROZEN_CONFIG)
    parser.add_argument("--frozen-metadata", type=Path, default=DEFAULT_FROZEN_METADATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_stability_review(
            stability_config_path=args.stability_config,
            frozen_config_path=args.frozen_config,
            frozen_metadata_path=args.frozen_metadata,
            report_path=args.report, figure_path=args.figure,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, yaml.YAMLError) as error:
        print(f"Error: {error}")
        return 1
    print(f"Freeze criteria passed: {report['freeze_decision']['passed']}")
    print(f"Decision: {report['freeze_decision']['decision']}")
    print(f"Validation passed: {report['validation']['passed']}")
    print("No threshold, synthetic evaluation, performance metric, or new persisted seed model was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
