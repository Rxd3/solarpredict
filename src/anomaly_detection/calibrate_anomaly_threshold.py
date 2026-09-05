"""Calibrate and freeze a prototype threshold without fitting model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
import yaml

from src.anomaly_detection.train_baseline_detector import SYNTHETIC_LABEL_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_CONFIG = ROOT / "config/anomaly_threshold_calibration.yaml"
DEFAULT_FROZEN_DETECTOR_CONFIG = ROOT / "config/frozen_anomaly_detector.yaml"
DEFAULT_FROZEN_DETECTOR_METADATA = ROOT / "models/frozen_anomaly_detector_metadata.json"
DEFAULT_FROZEN_THRESHOLD_CONFIG = ROOT / "config/frozen_anomaly_threshold.yaml"
DEFAULT_FROZEN_THRESHOLD_METADATA = ROOT / "models/frozen_anomaly_threshold_metadata.json"
DEFAULT_REPORT_PATH = ROOT / "outputs/anomaly_threshold_calibration.json"
DEFAULT_FIGURE_PATH = ROOT / "outputs/figures/operational/day20_threshold_calibration.png"
DAY18_REPORT_PATH = ROOT / "outputs/model_feature_comparison.json"
SPLIT_CONFIG_PATH = ROOT / "config/chronological_split.yaml"

FEATURE_ORDER = (
    "Power_Generated",
    "Solar_Radiation",
    "Air_Temp",
    "Relative_Humidity",
    "Wind_Speed",
    "rtd_mean",
    "rtd_std",
)
EXPECTED_CANDIDATES = (
    ("calibration_percentile_95", "calibration_percentile", 95.0),
    ("calibration_percentile_97_5", "calibration_percentile", 97.5),
    ("calibration_percentile_99", "calibration_percentile", 99.0),
    ("calibration_median_plus_1_5_mad", "calibration_median_plus_k_mad", 1.5),
    ("calibration_median_plus_2_mad", "calibration_median_plus_k_mad", 2.0),
    ("calibration_median_plus_2_5_mad", "calibration_median_plus_k_mad", 2.5),
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


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _timestamp(value: Any) -> str:
    return pd.Timestamp(value).isoformat(sep=" ")


def _atomic_text(text: str, destination: Path, suffix: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=suffix,
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(value: dict[str, Any], destination: Path) -> None:
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    _atomic_text(serialized, destination, ".json.tmp")


def _snapshot(paths: list[Path]) -> dict[str, str]:
    return {
        _relative(path): sha256_file(path)
        for path in sorted(set(path.resolve() for path in paths))
        if path.is_file()
    }


def load_calibration_config(path: str | Path = DEFAULT_CALIBRATION_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Threshold calibration configuration not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported threshold calibration configuration.")
    declared = []
    for candidate in config.get("candidate_rules", []):
        parameter = candidate.get("percentile", candidate.get("k"))
        declared.append((candidate.get("id"), candidate.get("strategy"), float(parameter)))
    if tuple(declared) != EXPECTED_CANDIDATES:
        raise ValueError("Threshold candidates must exactly match the predeclared candidate set.")
    if config.get("comparison_operator") != "strictly_greater_than":
        raise ValueError("The frozen comparison operator must be strictly greater than.")
    grouping = config.get("event_grouping", {})
    if grouping.get("same_event_gap_minutes") != 2 or grouping.get("smoothing") is not False:
        raise ValueError("Event grouping must use exact consecutive two-minute rows without smoothing.")
    return config


def load_frozen_artifacts(
    detector_config_path: str | Path = DEFAULT_FROZEN_DETECTOR_CONFIG,
    detector_metadata_path: str | Path = DEFAULT_FROZEN_DETECTOR_METADATA,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any, dict[str, Any]]:
    """Load and verify the frozen scaler/model; never call ``fit`` here."""
    config_path = Path(detector_config_path)
    metadata_path = Path(detector_metadata_path)
    detector_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    detector_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if detector_config.get("status") != "prototype_frozen":
        raise ValueError("Detector configuration is not frozen.")
    if tuple(detector_config.get("feature_order", [])) != FEATURE_ORDER:
        raise ValueError("Frozen detector feature order does not match NO_TIME_7.")
    if detector_metadata.get("feature_order") != list(FEATURE_ORDER):
        raise ValueError("Frozen detector metadata feature order does not match NO_TIME_7.")

    scaler_path = _resolve_project_path(detector_metadata["selected_scaler"]["path"])
    model_path = _resolve_project_path(detector_metadata["selected_model"]["path"])
    if _relative(scaler_path) != detector_config["scaler"]["artifact"]:
        raise ValueError("Scaler reference differs between frozen configuration and metadata.")
    if _relative(model_path) != detector_config["model"]["artifact"]:
        raise ValueError("Model reference differs between frozen configuration and metadata.")
    if sha256_file(scaler_path) != detector_metadata["selected_scaler"]["sha256"]:
        raise ValueError("Frozen scaler hash verification failed.")
    if sha256_file(model_path) != detector_metadata["selected_model"]["sha256"]:
        raise ValueError("Frozen model hash verification failed.")

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    if int(scaler.n_features_in_) != len(FEATURE_ORDER):
        raise ValueError("Frozen scaler was not fitted on seven features.")
    if int(scaler.n_samples_seen_) != detector_config["training_partition"]["rows"]:
        raise ValueError("Frozen scaler training-row count disagrees with the configuration.")
    if int(model.n_features_in_) != len(FEATURE_ORDER):
        raise ValueError("Frozen model was not fitted on seven features.")
    model_parameters = model.get_params(deep=False)
    for name, expected in detector_config["model"]["parameters"].items():
        if model_parameters[name] != expected:
            raise ValueError(f"Frozen model parameter mismatch: {name}.")
    verification = {
        "detector_config": {"path": _relative(config_path), "sha256": sha256_file(config_path)},
        "detector_metadata": {"path": _relative(metadata_path), "sha256": sha256_file(metadata_path)},
        "scaler": {"path": _relative(scaler_path), "sha256": sha256_file(scaler_path)},
        "model": {"path": _relative(model_path), "sha256": sha256_file(model_path)},
        "scaler_training_rows": int(scaler.n_samples_seen_),
        "feature_order": list(FEATURE_ORDER),
        "fit_called": False,
    }
    return detector_config, detector_metadata, scaler, model, verification


def load_partition(
    path: str | Path,
    *,
    expected_rows: int,
    expected_start: str,
    expected_end: str,
) -> pd.DataFrame:
    source = Path(path)
    frame = pd.read_csv(source)
    if "Timestamp" not in frame.columns:
        raise ValueError(f"Timestamp is missing from {source}.")
    parsed = pd.to_datetime(frame["Timestamp"], errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Timestamp parsing failed for {source}.")
    if parsed.dt.tz is not None:
        raise ValueError("Operational timestamps must remain timezone-naive.")
    frame["Timestamp"] = parsed
    missing_features = [feature for feature in FEATURE_ORDER if feature not in frame.columns]
    if missing_features:
        raise ValueError(f"Required frozen features missing from {source}: {missing_features}")
    if SYNTHETIC_LABEL_COLUMNS.intersection(frame.columns):
        raise ValueError(f"Synthetic label columns are forbidden in {source}.")
    for feature in FEATURE_ORDER:
        original_missing = frame[feature].isna()
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        if (frame[feature].isna() & ~original_missing).any():
            raise ValueError(f"Non-numeric values found in {source}:{feature}.")
    values = frame.loc[:, FEATURE_ORDER].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Frozen features in {source} must be complete and finite.")
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {source}, found {len(frame)}.")
    if frame.Timestamp.duplicated().any() or not frame.Timestamp.is_monotonic_increasing:
        raise ValueError(f"Timestamps in {source} must be unique and chronological.")
    if _timestamp(frame.Timestamp.iloc[0]) != expected_start or _timestamp(frame.Timestamp.iloc[-1]) != expected_end:
        raise ValueError(f"Partition boundaries do not match for {source}.")
    return frame


def score_partition(frame: pd.DataFrame, scaler: Any, model: Any) -> tuple[pd.DataFrame, np.ndarray]:
    scaled = scaler.transform(frame.loc[:, FEATURE_ORDER].to_numpy(dtype=float))
    scores = -model.score_samples(scaled)
    if not np.isfinite(scaled).all() or not np.isfinite(scores).all():
        raise RuntimeError("Frozen artifacts produced non-finite transformed values or scores.")
    result = pd.DataFrame({"Timestamp": frame.Timestamp.to_numpy(), "anomaly_score": scores})
    return result, scaled


def score_vector_sha256(scores: pd.DataFrame) -> str:
    lines = (
        f"{_timestamp(row.Timestamp)}|{float(row.anomaly_score):.17g}"
        for row in scores.itertuples(index=False)
    )
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def verify_day18_calibration_scores(scores: pd.DataFrame) -> dict[str, Any]:
    day18 = json.loads(DAY18_REPORT_PATH.read_text(encoding="utf-8"))
    recorded = day18["training_calibration_results"]["NO_TIME_7"]
    values = scores.anomaly_score
    reproduced = {
        "count": len(values),
        "mean": float(values.mean()),
        "population_standard_deviation": float(values.std(ddof=0)),
        "95th_percentile": float(values.quantile(0.95)),
    }
    expected = recorded["calibration_score_summary"]
    summary_matches = reproduced["count"] == expected["count"] and all(
        math.isclose(reproduced[name], float(expected[name]), abs_tol=1e-15)
        for name in ("mean", "population_standard_deviation", "95th_percentile")
    )
    top = scores.nlargest(10, "anomaly_score")
    reproduced_top = [
        {"timestamp": _timestamp(row.Timestamp), "anomaly_score": float(row.anomaly_score)}
        for row in top.itertuples(index=False)
    ]
    expected_top = recorded["high_score_calibration_review"]["top_10"]
    top_matches = all(
        current["timestamp"] == saved["timestamp"]
        and math.isclose(current["anomaly_score"], float(saved["anomaly_score"]), abs_tol=1e-15)
        for current, saved in zip(reproduced_top, expected_top, strict=True)
    )
    return {
        "day18_report": {"path": _relative(DAY18_REPORT_PATH), "sha256": sha256_file(DAY18_REPORT_PATH)},
        "reproduced_summary": reproduced,
        "summary_matches": summary_matches,
        "top_10_timestamps_and_scores_match": top_matches,
        "passed": summary_matches and top_matches,
    }


def group_flagged_events(
    timestamps: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    *,
    expected_gap_minutes: int = 2,
) -> list[dict[str, Any]]:
    ordered = pd.DatetimeIndex(pd.to_datetime(timestamps)).sort_values()
    if ordered.has_duplicates:
        raise ValueError("Flagged timestamps must be unique.")
    if len(ordered) == 0:
        return []
    groups: list[list[pd.Timestamp]] = [[pd.Timestamp(ordered[0])]]
    exact_gap = pd.Timedelta(minutes=expected_gap_minutes)
    for value in ordered[1:]:
        timestamp = pd.Timestamp(value)
        if timestamp - groups[-1][-1] == exact_gap:
            groups[-1].append(timestamp)
        else:
            groups.append([timestamp])
    return [
        {
            "start": _timestamp(group[0]),
            "end": _timestamp(group[-1]),
            "flagged_observation_count": len(group),
            "duration_minutes": len(group) * expected_gap_minutes,
        }
        for group in groups
    ]


def summarize_flags(
    partition: pd.DataFrame,
    scores: pd.DataFrame,
    threshold: float,
    extreme_rtd_mask: np.ndarray,
    *,
    low_light_maximum: float,
    event_gap_minutes: int,
) -> dict[str, Any]:
    flagged = scores.anomaly_score.to_numpy() > threshold
    flagged_count = int(flagged.sum())
    events = group_flagged_events(
        scores.loc[flagged, "Timestamp"], expected_gap_minutes=event_gap_minutes
    )
    event_durations = [event["duration_minutes"] for event in events]
    low_light = partition.Solar_Radiation.to_numpy(dtype=float) <= low_light_maximum
    extreme_count = int(np.logical_and(flagged, extreme_rtd_mask).sum())
    isolated_count = sum(event["flagged_observation_count"] == 1 for event in events)
    return {
        "threshold_value": float(threshold),
        "comparison_operator": "strictly_greater_than",
        "baseline_flagged_observation_count": flagged_count,
        "baseline_flagged_percentage": flagged_count / len(partition) * 100,
        "event_count": len(events),
        "typical_event_duration_minutes": float(np.median(event_durations)) if event_durations else 0.0,
        "maximum_event_duration_minutes": max(event_durations, default=0),
        "isolated_event_count": isolated_count,
        "isolated_event_fraction": isolated_count / len(events) if events else 0.0,
        "context_breakdown": {
            "daylight_like_flagged_count": int(np.logical_and(flagged, ~low_light).sum()),
            "low_light_flagged_count": int(np.logical_and(flagged, low_light).sum()),
        },
        "extreme_rtd_std_sensitivity": {
            "extreme_rtd_std_flagged_count": extreme_count,
            "other_flagged_count": flagged_count - extreme_count,
            "extreme_rtd_fraction_of_flagged_rows": extreme_count / flagged_count if flagged_count else None,
        },
        "events": events,
        "flagged_mask": flagged,
    }


def evaluate_candidates(
    calibration: pd.DataFrame,
    scores: pd.DataFrame,
    scaled: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = scores.anomaly_score.to_numpy(dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    rtd_index = FEATURE_ORDER.index(config["extreme_rtd_rule"]["feature"])
    rtd_limit = float(config["extreme_rtd_rule"]["absolute_training_scaled_value_strictly_greater_than"])
    extreme = np.abs(scaled[:, rtd_index]) > rtd_limit
    expected_extreme = int(config["extreme_rtd_rule"]["expected_calibration_count"])
    if int(extreme.sum()) != expected_extreme:
        raise RuntimeError(f"Expected {expected_extreme} extreme-rtd rows, found {int(extreme.sum())}.")

    criteria = config["selection_criteria"]
    candidates: list[dict[str, Any]] = []
    for declaration_index, rule in enumerate(config["candidate_rules"]):
        if rule["strategy"] == "calibration_percentile":
            threshold = float(np.quantile(values, float(rule["percentile"]) / 100))
            statistic = {"percentile": float(rule["percentile"])}
        elif rule["strategy"] == "calibration_median_plus_k_mad":
            threshold = median + float(rule["k"]) * mad
            statistic = {"median": median, "mad": mad, "k": float(rule["k"])}
        else:
            raise ValueError(f"Unsupported threshold strategy: {rule['strategy']}")
        summary = summarize_flags(
            calibration,
            scores,
            threshold,
            extreme,
            low_light_maximum=float(config["context_rule"]["low_light_maximum_inclusive"]),
            event_gap_minutes=int(config["event_grouping"]["same_event_gap_minutes"]),
        )
        rtd_fraction = summary["extreme_rtd_std_sensitivity"]["extreme_rtd_fraction_of_flagged_rows"]
        gates = {
            "minimum_flagged_rows": summary["baseline_flagged_observation_count"] >= int(criteria["minimum_flagged_rows"]),
            "maximum_flagged_rows": summary["baseline_flagged_observation_count"] <= int(criteria["maximum_flagged_rows"]),
            "minimum_alert_percentage": summary["baseline_flagged_percentage"] >= float(criteria["minimum_baseline_alert_percentage"]),
            "maximum_alert_percentage": summary["baseline_flagged_percentage"] <= float(criteria["maximum_baseline_alert_percentage"]),
            "maximum_isolated_event_fraction": summary["isolated_event_fraction"] <= float(criteria["maximum_isolated_event_fraction"]),
            "maximum_extreme_rtd_fraction": rtd_fraction is not None and rtd_fraction <= float(criteria["maximum_extreme_rtd_fraction_of_flagged_rows"]),
        }
        public_summary = {key: value for key, value in summary.items() if key != "flagged_mask"}
        candidates.append({
            "id": rule["id"],
            "strategy": rule["strategy"],
            "calibration_statistic": statistic,
            "declaration_index": declaration_index,
            **public_summary,
            "selection_gate_results": gates,
            "eligible": all(gates.values()),
        })
    return candidates, {
        "definition": f"absolute training-scaled rtd_std > {rtd_limit:g}",
        "calibration_count": int(extreme.sum()),
        "mask": extreme,
        "calibration_score_median": median,
        "calibration_score_mad": mad,
    }


def select_threshold(candidates: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Select from calibration summaries only, using the predeclared ordering."""
    target = float(config["selection_criteria"]["target_baseline_alert_percentage"])
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise RuntimeError("No threshold candidate passed every predeclared selection gate.")

    def key(candidate: dict[str, Any]) -> tuple[float, float, float, int]:
        rtd_fraction = candidate["extreme_rtd_std_sensitivity"]["extreme_rtd_fraction_of_flagged_rows"]
        return (
            abs(candidate["baseline_flagged_percentage"] - target),
            candidate["isolated_event_fraction"],
            float(rtd_fraction),
            candidate["declaration_index"],
        )

    chosen = min(eligible, key=key)
    return {
        **chosen,
        "selection_scope": "calibration only",
        "evaluation_baseline_used": False,
        "selection_target_baseline_alert_percentage": target,
        "rationale": (
            "Passed every predeclared alert-rate, fragmentation, and RTD-concentration gate and was closest "
            "to the predeclared 2.5% calibration baseline alert-rate target; the highest threshold was not automatically preferred."
        ),
    }


def write_frozen_threshold(
    selected: dict[str, Any],
    config: dict[str, Any],
    verification: dict[str, Any],
    calibration_path: Path,
    calibration: pd.DataFrame,
    score_hash: str,
    calibration_config_path: Path,
    threshold_config_path: Path,
    threshold_metadata_path: Path,
) -> dict[str, Any]:
    context = selected["context_breakdown"]
    rtd = selected["extreme_rtd_std_sensitivity"]
    threshold_config = {
        "schema_version": 1,
        "status": "prototype_frozen",
        "configuration_name": "NO_TIME_7_calibration_threshold",
        "frozen_detector_config": verification["detector_config"]["path"],
        "frozen_detector_config_sha256": verification["detector_config"]["sha256"],
        "threshold": {
            "strategy": selected["strategy"],
            "value": selected["threshold_value"],
            "comparison_operator": selected["comparison_operator"],
            "calibration_statistic": selected["calibration_statistic"],
        },
        "calibration_partition": {
            "path": _relative(calibration_path),
            "rows": len(calibration),
            "start": _timestamp(calibration.Timestamp.iloc[0]),
            "end": _timestamp(calibration.Timestamp.iloc[-1]),
            "baseline_flagged_observations": selected["baseline_flagged_observation_count"],
            "baseline_flagged_percentage": selected["baseline_flagged_percentage"],
            "alert_event_count": selected["event_count"],
            "daylight_like_flagged_count": context["daylight_like_flagged_count"],
            "low_light_flagged_count": context["low_light_flagged_count"],
            "extreme_rtd_std_flagged_count": rtd["extreme_rtd_std_flagged_count"],
            "other_flagged_count": rtd["other_flagged_count"],
        },
        "event_grouping": config["event_grouping"],
        "selection": {
            "scope": "calibration only",
            "evaluation_baseline_used": False,
            "candidate_config": _relative(calibration_config_path),
            "rationale": selected["rationale"],
        },
        "known_limitations": config["known_limitations"],
    }
    _atomic_text(yaml.safe_dump(threshold_config, sort_keys=False), threshold_config_path, ".yaml.tmp")
    threshold_metadata = {
        "schema_version": 1,
        "status": "prototype_frozen",
        "artifact_type": "anomaly_threshold_metadata",
        "trained_model_stored": False,
        "threshold_strategy": selected["strategy"],
        "threshold_value": selected["threshold_value"],
        "comparison_operator": selected["comparison_operator"],
        "calibration_statistic": selected["calibration_statistic"],
        "calibration_score_source": {
            "partition_path": _relative(calibration_path),
            "partition_sha256": sha256_file(calibration_path),
            "score_vector_sha256": score_hash,
            "score_convention": "higher score = more anomalous; score = -model.score_samples",
        },
        "frozen_detector_metadata": verification["detector_metadata"],
        "frozen_detector_config": verification["detector_config"],
        "selected_scaler": verification["scaler"],
        "selected_model": verification["model"],
        "calibration_partition_boundaries": {
            "rows": len(calibration),
            "start": _timestamp(calibration.Timestamp.iloc[0]),
            "end": _timestamp(calibration.Timestamp.iloc[-1]),
        },
        "selection_scope": "calibration only",
        "evaluation_baseline_used_for_selection": False,
        "selection_completed_before_evaluation_sanity_check": True,
        "known_limitations": config["known_limitations"],
    }
    _atomic_json(threshold_metadata, threshold_metadata_path)
    return {
        "threshold_config": {"path": _relative(threshold_config_path), "sha256": sha256_file(threshold_config_path)},
        "threshold_metadata": {"path": _relative(threshold_metadata_path), "sha256": sha256_file(threshold_metadata_path)},
        "new_model_created": False,
    }


def evaluation_sanity_check(
    evaluation: pd.DataFrame,
    scores: pd.DataFrame,
    scaled: np.ndarray,
    threshold: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    rtd_index = FEATURE_ORDER.index(config["extreme_rtd_rule"]["feature"])
    limit = float(config["extreme_rtd_rule"]["absolute_training_scaled_value_strictly_greater_than"])
    summary = summarize_flags(
        evaluation,
        scores,
        threshold,
        np.abs(scaled[:, rtd_index]) > limit,
        low_light_maximum=float(config["context_rule"]["low_light_maximum_inclusive"]),
        event_gap_minutes=int(config["event_grouping"]["same_event_gap_minutes"]),
    )
    public_summary = {key: value for key, value in summary.items() if key != "flagged_mask"}
    transition = pd.Timestamp("2022-04-28 17:00:00")
    location = evaluation.index[evaluation.Timestamp.eq(transition)]
    if len(location) != 1:
        raise RuntimeError("Expected the recurring 2022-04-28 17:00 transition exactly once.")
    position = int(location[0])
    public_summary.update({
        "partition_rows": len(evaluation),
        "partition_start": _timestamp(evaluation.Timestamp.iloc[0]),
        "partition_end": _timestamp(evaluation.Timestamp.iloc[-1]),
        "score_vector_sha256": score_vector_sha256(scores),
        "recurring_2022_04_28_17_00_transition": {
            "anomaly_score": float(scores.loc[position, "anomaly_score"]),
            "flagged": bool(summary["flagged_mask"][position]),
            "context": (
                "low_light"
                if float(evaluation.loc[position, "Solar_Radiation"])
                <= float(config["context_rule"]["low_light_maximum_inclusive"])
                else "daylight_like"
            ),
            "interpretation": "post-freeze sanity check for a recurring operational transition; not a performance label",
        },
        "threshold_modified_after_review": False,
        "interpretation": "Post-freeze descriptive baseline sanity check; not model-performance evaluation.",
    })
    return public_summary


def save_figure(
    calibration: pd.DataFrame,
    scores: pd.DataFrame,
    threshold: float,
    destination: Path,
) -> None:
    flagged = scores.anomaly_score.to_numpy() > threshold
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(scores.Timestamp, scores.anomaly_score, color="#2f6690", linewidth=1.1, label="Calibration baseline score")
    axis.axhline(threshold, color="#b23a48", linestyle="--", linewidth=1.3, label=f"Frozen threshold: {threshold:.6f}")
    axis.scatter(
        scores.loc[flagged, "Timestamp"],
        scores.loc[flagged, "anomaly_score"],
        color="#d1495b",
        s=28,
        zorder=3,
        label="Flagged calibration observation",
    )
    axis.set_title("Day 20 calibration baseline scores and frozen prototype threshold")
    axis.set_xlabel("Source timestamp (timezone not specified)")
    axis.set_ylabel("Anomaly score = -score_samples (higher is more isolated)")
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
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


def run_threshold_calibration(
    *,
    calibration_config_path: Path = DEFAULT_CALIBRATION_CONFIG,
    frozen_detector_config_path: Path = DEFAULT_FROZEN_DETECTOR_CONFIG,
    frozen_detector_metadata_path: Path = DEFAULT_FROZEN_DETECTOR_METADATA,
    frozen_threshold_config_path: Path = DEFAULT_FROZEN_THRESHOLD_CONFIG,
    frozen_threshold_metadata_path: Path = DEFAULT_FROZEN_THRESHOLD_METADATA,
    report_path: Path = DEFAULT_REPORT_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
) -> dict[str, Any]:
    workflow_order: list[str] = []
    excluded_model_path = Path(frozen_threshold_metadata_path).resolve()
    protected_paths = [path for path in (ROOT / "data").rglob("*") if path.is_file()]
    protected_paths += [
        path for path in (ROOT / "models").rglob("*")
        if path.is_file() and path.resolve() != excluded_model_path
    ]
    protected_paths += [
        Path(frozen_detector_config_path),
        ROOT / "config/model_stability_review.yaml",
        DAY18_REPORT_PATH,
        ROOT / "outputs/model_stability_review.json",
    ]
    protected_before = _snapshot(protected_paths)
    joblib_before = _snapshot([path for path in (ROOT / "models").rglob("*.joblib")])
    synthetic_before = _snapshot([path for path in (ROOT / "data").rglob("*synthetic*") if path.is_file()])

    config = load_calibration_config(calibration_config_path)
    detector_config, _, scaler, model, verification = load_frozen_artifacts(
        frozen_detector_config_path, frozen_detector_metadata_path
    )
    workflow_order.append("frozen_artifacts_loaded_and_hash_verified_without_fit")

    calibration_path = _resolve_project_path(config["calibration_partition"])
    calibration_boundary = detector_config["calibration_partition"]
    calibration = load_partition(
        calibration_path,
        expected_rows=int(calibration_boundary["rows"]),
        expected_start=str(calibration_boundary["start"]),
        expected_end=str(calibration_boundary["end"]),
    )
    calibration_scores, calibration_scaled = score_partition(calibration, scaler, model)
    reproduction = verify_day18_calibration_scores(calibration_scores)
    if not reproduction["passed"]:
        raise RuntimeError("Frozen artifacts did not reproduce Day 18 NO_TIME_7 calibration scores.")
    calibration_score_hash = score_vector_sha256(calibration_scores)
    workflow_order.append("calibration_scores_reproduced")

    candidates, rtd_review = evaluate_candidates(calibration, calibration_scores, calibration_scaled, config)
    selected = select_threshold(candidates, config)
    workflow_order.append("threshold_selected_using_calibration_only")
    selected_snapshot = json.dumps(selected, sort_keys=True, allow_nan=False)

    frozen_artifacts = write_frozen_threshold(
        selected,
        config,
        verification,
        calibration_path,
        calibration,
        calibration_score_hash,
        Path(calibration_config_path),
        Path(frozen_threshold_config_path),
        Path(frozen_threshold_metadata_path),
    )
    workflow_order.append("threshold_configuration_and_metadata_frozen")
    frozen_hashes_before_evaluation = {
        "config": sha256_file(frozen_threshold_config_path),
        "metadata": sha256_file(frozen_threshold_metadata_path),
    }

    # Evaluation is deliberately first touched only after the threshold files exist.
    split_config = yaml.safe_load(SPLIT_CONFIG_PATH.read_text(encoding="utf-8"))
    evaluation_boundary = split_config["partitions"]["evaluation_baseline"]
    evaluation_path = _resolve_project_path(config["evaluation_baseline_partition"])
    evaluation = load_partition(
        evaluation_path,
        expected_rows=int(evaluation_boundary["expected_rows"]),
        expected_start=str(evaluation_boundary["start"]),
        expected_end=str(evaluation_boundary["end"]),
    )
    evaluation_scores, evaluation_scaled = score_partition(evaluation, scaler, model)
    evaluation_review = evaluation_sanity_check(
        evaluation, evaluation_scores, evaluation_scaled, selected["threshold_value"], config
    )
    workflow_order.append("post_freeze_evaluation_baseline_sanity_check")
    selected_unchanged = selected_snapshot == json.dumps(selected, sort_keys=True, allow_nan=False)
    frozen_unchanged_after_evaluation = frozen_hashes_before_evaluation == {
        "config": sha256_file(frozen_threshold_config_path),
        "metadata": sha256_file(frozen_threshold_metadata_path),
    }

    save_figure(calibration, calibration_scores, selected["threshold_value"], Path(figure_path))
    protected_after = _snapshot(protected_paths)
    joblib_after = _snapshot([path for path in (ROOT / "models").rglob("*.joblib")])
    synthetic_after = _snapshot([path for path in (ROOT / "data").rglob("*synthetic*") if path.is_file()])

    frozen_threshold = yaml.safe_load(Path(frozen_threshold_config_path).read_text(encoding="utf-8"))
    validation = {
        "frozen_artifact_hashes_verified": all(
            verification[name]["sha256"] == sha256_file(_resolve_project_path(verification[name]["path"]))
            for name in ("detector_config", "detector_metadata", "scaler", "model")
        ),
        "frozen_scaler_and_model_loaded_without_fit": verification["fit_called"] is False,
        "exact_seven_feature_order": verification["feature_order"] == list(FEATURE_ORDER),
        "scaler_was_fitted_on_336_training_rows": verification["scaler_training_rows"] == 336,
        "day18_calibration_scores_reproduced": reproduction["passed"],
        "candidate_set_matches_predeclaration": tuple(
            (candidate["id"], candidate["strategy"], float(candidate["calibration_statistic"].get("percentile", candidate["calibration_statistic"].get("k"))))
            for candidate in candidates
        ) == EXPECTED_CANDIDATES,
        "threshold_selection_used_calibration_only": selected["evaluation_baseline_used"] is False,
        "selection_preceded_evaluation_sanity_check": workflow_order.index("threshold_configuration_and_metadata_frozen") < workflow_order.index("post_freeze_evaluation_baseline_sanity_check"),
        "evaluation_could_not_modify_selection": selected_unchanged and frozen_unchanged_after_evaluation,
        "event_grouping_is_exact_two_minutes": config["event_grouping"]["same_event_gap_minutes"] == 2 and config["event_grouping"]["smoothing"] is False,
        "threshold_config_references_unchanged_detector": frozen_threshold["frozen_detector_config"] == _relative(frozen_detector_config_path),
        "no_new_model_training_or_joblib_artifact": joblib_before == joblib_after and frozen_artifacts["new_model_created"] is False,
        "source_datasets_and_frozen_detector_artifacts_unchanged": protected_before == protected_after,
        "no_synthetic_labels_or_data_used": synthetic_before == synthetic_after and not SYNTHETIC_LABEL_COLUMNS.intersection(calibration.columns) and not SYNTHETIC_LABEL_COLUMNS.intersection(evaluation.columns),
        "no_performance_metrics_calculated": True,
    }
    if not all(validation.values()):
        failed = [name for name, passed in validation.items() if not passed]
        raise RuntimeError(f"Day 20 validation failed: {failed}")

    report = {
        "stage": "Day 20 anomaly threshold calibration",
        "scope": {
            "selection_partition": "calibration baseline only",
            "evaluation_baseline_role": "post-freeze descriptive sanity check only",
            "score_convention": "higher score = more anomalous; score = -model.score_samples",
            "threshold_comparison": "score > threshold",
        },
        "frozen_detector_verification": verification,
        "calibration_score_reproduction": {
            **reproduction,
            "score_vector_sha256": calibration_score_hash,
            "partition_path": _relative(calibration_path),
            "partition_sha256": sha256_file(calibration_path),
        },
        "predeclared_candidate_configuration": {
            "path": _relative(calibration_config_path),
            "sha256": sha256_file(calibration_config_path),
            "selection_criteria": config["selection_criteria"],
        },
        "candidate_thresholds": candidates,
        "extreme_rtd_std_definition": {key: value for key, value in rtd_review.items() if key != "mask"},
        "selected_threshold": selected,
        "freeze_rationale": selected["rationale"],
        "frozen_threshold_artifacts": frozen_artifacts,
        "event_grouping_rule": {
            **config["event_grouping"],
            "duration_definition": "flagged observation count multiplied by the two-minute sample interval",
        },
        "post_freeze_evaluation_baseline_sanity_check": evaluation_review,
        "workflow_order": workflow_order,
        "known_limitations": config["known_limitations"],
        "protected_artifact_hashes": {
            path: {"before": digest, "after": protected_after[path], "unchanged": protected_after[path] == digest}
            for path, digest in protected_before.items()
        },
        "validation": {"passed": True, "checks": validation},
        "explicitly_not_performed": [
            "scaler refitting",
            "detector retraining",
            "feature-set changes",
            "synthetic evaluation generation",
            "precision/recall/F1 calculation",
            "threshold revision using evaluation-baseline results",
        ],
    }
    _atomic_json(report, Path(report_path))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-config", type=Path, default=DEFAULT_CALIBRATION_CONFIG)
    parser.add_argument("--frozen-detector-config", type=Path, default=DEFAULT_FROZEN_DETECTOR_CONFIG)
    parser.add_argument("--frozen-detector-metadata", type=Path, default=DEFAULT_FROZEN_DETECTOR_METADATA)
    parser.add_argument("--frozen-threshold-config", type=Path, default=DEFAULT_FROZEN_THRESHOLD_CONFIG)
    parser.add_argument("--frozen-threshold-metadata", type=Path, default=DEFAULT_FROZEN_THRESHOLD_METADATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_threshold_calibration(
            calibration_config_path=args.calibration_config,
            frozen_detector_config_path=args.frozen_detector_config,
            frozen_detector_metadata_path=args.frozen_detector_metadata,
            frozen_threshold_config_path=args.frozen_threshold_config,
            frozen_threshold_metadata_path=args.frozen_threshold_metadata,
            report_path=args.report,
            figure_path=args.figure,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, yaml.YAMLError) as error:
        print(f"Error: {error}")
        return 1
    selected = report["selected_threshold"]
    sanity = report["post_freeze_evaluation_baseline_sanity_check"]
    print(f"Frozen threshold: {selected['threshold_value']:.17g} ({selected['id']})")
    print(f"Calibration baseline alerts: {selected['baseline_flagged_observation_count']} rows in {selected['event_count']} events")
    print(f"Post-freeze evaluation-baseline alerts: {sanity['baseline_flagged_observation_count']} rows in {sanity['event_count']} events")
    print(f"Validation passed: {report['validation']['passed']}")
    print("No fitting, synthetic evaluation generation, or performance metrics were performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
