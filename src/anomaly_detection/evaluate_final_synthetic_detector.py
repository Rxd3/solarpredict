"""Evaluate the frozen detector on the fixed Day 21 synthetic evaluation copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from src.anomaly_detection.calibrate_anomaly_threshold import (
    FEATURE_ORDER,
    group_flagged_events,
    load_frozen_artifacts,
)
from src.anomaly_detection.generate_final_synthetic_evaluation import (
    EFFECT_COLUMN,
    FROZEN_FILES,
    NO_TIME_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_INPUT = ROOT / "data/model_ready/final_synthetic_evaluation_no_time_7.csv"
BASELINE_INPUT = ROOT / "data/model_ready/baseline_evaluation.csv"
SYNTHETIC_METADATA = ROOT / "outputs/final_synthetic_evaluation_metadata.json"
FROZEN_DETECTOR_CONFIG = ROOT / "config/frozen_anomaly_detector.yaml"
FROZEN_DETECTOR_METADATA = ROOT / "models/frozen_anomaly_detector_metadata.json"
FROZEN_THRESHOLD_CONFIG = ROOT / "config/frozen_anomaly_threshold.yaml"
FROZEN_THRESHOLD_METADATA = ROOT / "models/frozen_anomaly_threshold_metadata.json"
DEFAULT_PREDICTIONS = ROOT / "outputs/final_synthetic_anomaly_predictions.csv"
DEFAULT_COMPARISON = ROOT / "outputs/final_synthetic_score_comparison.csv"
DEFAULT_REPORT = ROOT / "outputs/final_anomaly_detector_evaluation.json"
DEFAULT_FIGURE = ROOT / "outputs/figures/operational/day22_synthetic_detection_results.png"
EXPECTED_THRESHOLD = 0.7135242760182695
EXPECTED_ROWS = 337
PROTECTED_START = pd.Timestamp("2022-04-28 16:40:00")
PROTECTED_END = pd.Timestamp("2022-04-28 17:20:00")
PREDICTION_COLUMNS = (
    "Timestamp",
    "anomaly_score",
    "predicted_anomaly",
    "synthetic_anomaly",
    "synthetic_anomaly_type",
    "synthetic_anomaly_id",
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


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


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


def load_frozen_threshold() -> tuple[float, dict[str, Any]]:
    threshold_config = yaml.safe_load(FROZEN_THRESHOLD_CONFIG.read_text(encoding="utf-8"))
    threshold_metadata = json.loads(FROZEN_THRESHOLD_METADATA.read_text(encoding="utf-8"))
    if threshold_config.get("status") != "prototype_frozen" or threshold_metadata.get("status") != "prototype_frozen":
        raise ValueError("Threshold configuration and metadata must both be frozen.")
    threshold = float(threshold_config["threshold"]["value"])
    if not math.isclose(threshold, EXPECTED_THRESHOLD, rel_tol=0, abs_tol=1e-15):
        raise ValueError("Frozen threshold differs from the Day 20 value.")
    if not math.isclose(float(threshold_metadata["threshold_value"]), threshold, rel_tol=0, abs_tol=1e-15):
        raise ValueError("Threshold metadata and configuration disagree.")
    if threshold_config["threshold"]["comparison_operator"] != "strictly_greater_than":
        raise ValueError("Frozen threshold comparison must remain strict greater-than.")
    detector_config_path = _resolve(threshold_config["frozen_detector_config"])
    if sha256_file(detector_config_path) != threshold_config["frozen_detector_config_sha256"]:
        raise ValueError("Threshold configuration's frozen-detector hash does not match.")
    detector_metadata_path = _resolve(threshold_metadata["frozen_detector_metadata"]["path"])
    if sha256_file(detector_metadata_path) != threshold_metadata["frozen_detector_metadata"]["sha256"]:
        raise ValueError("Threshold metadata's frozen-detector metadata hash does not match.")
    verification = {
        "threshold_config": {"path": _relative(FROZEN_THRESHOLD_CONFIG), "sha256": sha256_file(FROZEN_THRESHOLD_CONFIG)},
        "threshold_metadata": {"path": _relative(FROZEN_THRESHOLD_METADATA), "sha256": sha256_file(FROZEN_THRESHOLD_METADATA)},
        "strategy": threshold_config["threshold"]["strategy"],
        "value": threshold,
        "comparison_operator": "strictly_greater_than",
    }
    return threshold, verification


def load_synthetic_input(path: Path = SYNTHETIC_INPUT) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.columns.tolist() != list(NO_TIME_COLUMNS):
        raise ValueError("Synthetic evaluation columns or frozen feature order changed.")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    if frame.Timestamp.isna().any() or frame.Timestamp.dt.tz is not None:
        raise ValueError("Synthetic timestamps must be valid and timezone-naive.")
    if len(frame) != EXPECTED_ROWS or frame.Timestamp.duplicated().any() or not frame.Timestamp.is_monotonic_increasing:
        raise ValueError("Synthetic evaluation must contain 337 unique chronological timestamps.")
    values = frame.loc[:, FEATURE_ORDER].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Frozen model features must be finite and numeric.")
    frame.loc[:, FEATURE_ORDER] = values
    if not frame.synthetic_anomaly.isin([0, 1]).all():
        raise ValueError("Direct reference labels must be binary.")
    if not set(frame[EFFECT_COLUMN].unique()).issubset({"none", "direct", "propagated"}):
        raise ValueError("Unknown synthetic effect context.")
    if not frame.loc[frame[EFFECT_COLUMN].eq("propagated"), "synthetic_anomaly"].eq(0).all():
        raise ValueError("Propagated rows must retain direct reference label 0.")
    return frame


def load_paired_baseline(path: Path, timestamps: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    missing = [feature for feature in FEATURE_ORDER if feature not in frame.columns]
    if missing:
        raise ValueError(f"Paired baseline is missing frozen features: {missing}")
    if len(frame) != EXPECTED_ROWS or not frame.Timestamp.equals(timestamps):
        raise ValueError("Paired baseline timestamps do not exactly match synthetic evaluation.")
    values = frame.loc[:, FEATURE_ORDER].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Paired baseline frozen features must be finite.")
    frame.loc[:, FEATURE_ORDER] = values
    return frame


def score_frame(frame: pd.DataFrame, scaler: Any, model: Any) -> np.ndarray:
    scaled = scaler.transform(frame.loc[:, FEATURE_ORDER].to_numpy(dtype=float))
    scores = -model.score_samples(scaled)
    if not np.isfinite(scaled).all() or not np.isfinite(scores).all():
        raise RuntimeError("Frozen artifacts produced non-finite values or scores.")
    return scores


def confusion_and_metrics(reference: np.ndarray, predicted: np.ndarray) -> tuple[dict[str, int], dict[str, float]]:
    reference = np.asarray(reference, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    if reference.shape != predicted.shape or not np.isin(reference, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Reference and predicted labels must be same-length binary arrays.")
    tp = int(np.logical_and(reference == 1, predicted == 1).sum())
    fp = int(np.logical_and(reference == 0, predicted == 1).sum())
    tn = int(np.logical_and(reference == 0, predicted == 0).sum())
    fn = int(np.logical_and(reference == 1, predicted == 0).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / len(reference) if len(reference) else 0.0
    return (
        {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_count": fp,
            "false_positive_rate": false_positive_rate,
            "accuracy_not_emphasized": accuracy,
        },
    )


def score_delta_summary(values: pd.Series) -> dict[str, float | int]:
    return {
        "count": len(values),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "positive_count": int(values.gt(0).sum()),
        "negative_count": int(values.lt(0).sum()),
        "zero_count": int(values.eq(0).sum()),
    }


def evaluate_events(
    scored: pd.DataFrame,
    comparison: pd.DataFrame,
    metadata: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for event in metadata["events"]:
        direct = scored.synthetic_anomaly_id.eq(event["id"])
        propagated_times = pd.to_datetime(event["propagated_effect_timestamps"])
        propagated = scored.Timestamp.isin(propagated_times) & scored[EFFECT_COLUMN].eq("propagated")
        related = direct | propagated
        direct_scores = scored.loc[direct, "anomaly_score"]
        direct_baseline = comparison.loc[direct, "baseline_score"]
        direct_delta = comparison.loc[direct, "score_delta"]
        direct_alerts = scored.loc[direct, "predicted_anomaly"].eq(1)
        related_alerts = related & scored.predicted_anomaly.eq(1)
        alert_rows = scored.loc[related_alerts]
        if alert_rows.empty:
            first_timestamp = last_timestamp = first_phase = None
            delay_minutes = None
        else:
            first = alert_rows.iloc[0]
            last = alert_rows.iloc[-1]
            first_timestamp = _timestamp(first.Timestamp)
            last_timestamp = _timestamp(last.Timestamp)
            first_phase = str(first[EFFECT_COLUMN])
            delay_minutes = (pd.Timestamp(first.Timestamp) - pd.Timestamp(event["start_timestamp"])).total_seconds() / 60
        direct_records = []
        for position in scored.index[direct]:
            direct_records.append({
                "timestamp": _timestamp(scored.loc[position, "Timestamp"]),
                "synthetic_score": float(scored.loc[position, "anomaly_score"]),
                "baseline_score": float(comparison.loc[position, "baseline_score"]),
                "score_delta": float(comparison.loc[position, "score_delta"]),
                "flagged": bool(scored.loc[position, "predicted_anomaly"]),
            })
        direct_count = int(direct.sum())
        flagged_count = int(direct_alerts.sum())
        results.append({
            "event_id": event["id"],
            "anomaly_type": event["type"],
            "start_timestamp": event["start_timestamp"],
            "end_timestamp": event["end_timestamp"],
            "direct_row_count": direct_count,
            "direct_rows_flagged": flagged_count,
            "direct_row_detection_coverage": flagged_count / direct_count,
            "strict_event_detected": flagged_count > 0,
            "maximum_direct_anomaly_score": float(direct_scores.max()),
            "mean_direct_anomaly_score": float(direct_scores.mean()),
            "corresponding_baseline_scores": {
                "minimum": float(direct_baseline.min()),
                "maximum": float(direct_baseline.max()),
                "mean": float(direct_baseline.mean()),
            },
            "paired_direct_score_delta": score_delta_summary(direct_delta),
            "direct_row_score_records": direct_records,
            "directly_changed_frozen_features": [
                feature for feature in event["affected_measurements"] if feature in FEATURE_ORDER
            ],
            "context_aware_response": {
                "propagated_effect_row_count": int(propagated.sum()),
                "first_alert_phase": first_phase,
                "first_alert_timestamp": first_timestamp,
                "detection_delay_minutes_from_event_start": delay_minutes,
                "last_related_alert_timestamp": last_timestamp,
                "propagated_alerts_after_direct_event": int(
                    (propagated & scored.predicted_anomaly.eq(1)).sum()
                ),
                "response_category": (
                    "during_direct_event" if first_phase == "direct"
                    else "during_propagated_tail" if first_phase == "propagated"
                    else "not_at_all"
                ),
            },
            "interpretation": "controlled synthetic response; not evidence of real photovoltaic fault detection",
        })
    return results


def build_alert_events(scored: pd.DataFrame) -> list[dict[str, Any]]:
    predicted = scored.loc[scored.predicted_anomaly.eq(1)]
    grouped = group_flagged_events(predicted.Timestamp, expected_gap_minutes=2)
    results: list[dict[str, Any]] = []
    for index, event in enumerate(grouped, start=1):
        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])
        rows = scored.loc[scored.Timestamp.between(start, end) & scored.predicted_anomaly.eq(1)]
        contexts = set(rows[EFFECT_COLUMN])
        direct_ids = sorted(rows.synthetic_anomaly_id.dropna().unique())
        overlaps_direct = "direct" in contexts
        propagated_only = contexts == {"propagated"}
        unaffected_only = contexts == {"none"}
        classification = (
            "overlaps_direct_synthetic_event" if overlaps_direct
            else "propagated_effect_only" if propagated_only
            else "unaffected_baseline_only" if unaffected_only
            else "mixed_propagated_and_unaffected"
        )
        results.append({
            "alert_event_id": f"alert-{index:02d}",
            **event,
            "maximum_anomaly_score": float(rows.anomaly_score.max()),
            "effect_contexts": sorted(contexts),
            "classification": classification,
            "overlaps_direct_synthetic_event": overlaps_direct,
            "overlaps_only_propagated_effects": propagated_only,
            "entirely_unaffected_baseline": unaffected_only,
            "direct_synthetic_event_ids": direct_ids,
        })
    return results


def save_figure(
    scored: pd.DataFrame,
    events: list[dict[str, Any]],
    threshold: float,
    destination: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(13, 5.5))
    axis.plot(scored.Timestamp, scored.anomaly_score, color="#2f6690", linewidth=1.1, label="Synthetic evaluation score")
    axis.axhline(threshold, color="#b23a48", linestyle="--", linewidth=1.3, label=f"Frozen threshold: {threshold:.6f}")
    colors = {
        "gradual_power_degradation": "#f59e0b",
        "sustained_power_reduction": "#8b5cf6",
        "sensor_excursion": "#14b8a6",
    }
    for event in events:
        axis.axvspan(
            pd.Timestamp(event["start_timestamp"]),
            pd.Timestamp(event["end_timestamp"]) + pd.Timedelta(minutes=2),
            color=colors.get(event["anomaly_type"], "#94a3b8"), alpha=0.14,
            label=event["anomaly_type"].replace("_", " "),
        )
    flagged = scored.predicted_anomaly.eq(1)
    axis.scatter(
        scored.loc[flagged, "Timestamp"], scored.loc[flagged, "anomaly_score"],
        color="#dc2626", s=28, zorder=4, label="Predicted alert observation",
    )
    axis.set_title("Day 22 frozen detector response on final synthetic evaluation")
    axis.set_xlabel("Source timestamp (timezone not specified)")
    axis.set_ylabel("Anomaly score = -score_samples (higher is more isolated)")
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axis.grid(alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), frameon=False, ncol=2, loc="best")
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


def run_evaluation(
    *,
    synthetic_input_path: Path = SYNTHETIC_INPUT,
    baseline_input_path: Path = BASELINE_INPUT,
    synthetic_metadata_path: Path = SYNTHETIC_METADATA,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    comparison_path: Path = DEFAULT_COMPARISON,
    report_path: Path = DEFAULT_REPORT,
    figure_path: Path = DEFAULT_FIGURE,
) -> dict[str, Any]:
    protected_paths = [
        *[path for path in (ROOT / "data").rglob("*") if path.is_file()],
        *[path for path in (ROOT / "models").rglob("*") if path.is_file()],
        *FROZEN_FILES,
        Path(synthetic_metadata_path),
    ]
    protected_before = _snapshot(protected_paths)
    threshold_before = sha256_file(FROZEN_THRESHOLD_CONFIG)
    synthetic_before = sha256_file(synthetic_input_path)
    joblib_before = _snapshot([path for path in (ROOT / "models").rglob("*.joblib")])

    detector_config, detector_metadata, scaler, model, detector_verification = load_frozen_artifacts(
        FROZEN_DETECTOR_CONFIG, FROZEN_DETECTOR_METADATA
    )
    threshold, threshold_verification = load_frozen_threshold()
    if tuple(detector_config["feature_order"]) != FEATURE_ORDER:
        raise ValueError("Frozen detector feature order changed.")

    synthetic = load_synthetic_input(Path(synthetic_input_path))
    baseline = load_paired_baseline(Path(baseline_input_path), synthetic.Timestamp)
    metadata = json.loads(Path(synthetic_metadata_path).read_text(encoding="utf-8"))
    expected_synthetic_hash = metadata["no_time_7_output"]["sha256"]
    if sha256_file(synthetic_input_path) != expected_synthetic_hash:
        raise ValueError("Synthetic evaluation hash differs from Day 21 metadata.")
    if len(metadata["events"]) != 3:
        raise ValueError("Day 22 requires the fixed three-event Day 21 dataset.")

    synthetic_scores = score_frame(synthetic, scaler, model)
    baseline_scores = score_frame(baseline, scaler, model)
    predicted = (synthetic_scores > threshold).astype(np.int8)
    scored = pd.DataFrame({
        "Timestamp": synthetic.Timestamp,
        "anomaly_score": synthetic_scores,
        "predicted_anomaly": predicted,
        "synthetic_anomaly": synthetic.synthetic_anomaly.astype(np.int8),
        "synthetic_anomaly_type": synthetic.synthetic_anomaly_type,
        "synthetic_anomaly_id": synthetic.synthetic_anomaly_id,
        EFFECT_COLUMN: synthetic[EFFECT_COLUMN],
    })
    comparison = pd.DataFrame({
        "Timestamp": synthetic.Timestamp,
        "synthetic_score": synthetic_scores,
        "baseline_score": baseline_scores,
        "score_delta": synthetic_scores - baseline_scores,
    })
    _atomic_csv(scored, Path(predictions_path))
    _atomic_csv(comparison, Path(comparison_path))

    confusion, metrics = confusion_and_metrics(
        scored.synthetic_anomaly.to_numpy(dtype=int), predicted
    )
    context_alerts = {
        "direct": int((scored.predicted_anomaly.eq(1) & scored[EFFECT_COLUMN].eq("direct")).sum()),
        "propagated": int((scored.predicted_anomaly.eq(1) & scored[EFFECT_COLUMN].eq("propagated")).sum()),
        "none": int((scored.predicted_anomaly.eq(1) & scored[EFFECT_COLUMN].eq("none")).sum()),
    }
    event_results = evaluate_events(scored, comparison, metadata, threshold)
    detected_events = sum(event["strict_event_detected"] for event in event_results)
    event_metrics = {
        "definition": "detected when at least one directly modified event row has anomaly_score > frozen threshold",
        "total_events": len(event_results),
        "detected_events": detected_events,
        "missed_events": len(event_results) - detected_events,
        "event_level_recall": detected_events / len(event_results),
    }
    alert_events = build_alert_events(scored)

    protected = scored.Timestamp.between(PROTECTED_START, PROTECTED_END)
    transition = scored.Timestamp.eq(pd.Timestamp("2022-04-28 17:00:00"))
    protected_review = {
        "start": _timestamp(PROTECTED_START),
        "end": _timestamp(PROTECTED_END),
        "row_count": int(protected.sum()),
        "synthetic_effect_context_counts": scored.loc[protected, EFFECT_COLUMN].value_counts().to_dict(),
        "no_synthetic_effects_present": bool(scored.loc[protected, EFFECT_COLUMN].eq("none").all()),
        "synthetic_and_baseline_scores_match_exactly": bool(
            np.array_equal(synthetic_scores[protected.to_numpy()], baseline_scores[protected.to_numpy()])
        ),
        "flagged_row_count": int(scored.loc[protected, "predicted_anomaly"].sum()),
        "recurring_17_00_transition": {
            "anomaly_score": float(scored.loc[transition, "anomaly_score"].iloc[0]),
            "baseline_score": float(comparison.loc[transition, "baseline_score"].iloc[0]),
            "score_delta": float(comparison.loc[transition, "score_delta"].iloc[0]),
            "flagged": bool(scored.loc[transition, "predicted_anomaly"].iloc[0]),
            "interpretation": "recurring operational transition; not a synthetic event or verified fault",
        },
    }
    delta_by_context = {
        context: score_delta_summary(comparison.loc[scored[EFFECT_COLUMN].eq(context), "score_delta"])
        for context in ("direct", "propagated", "none")
    }
    performance_table = {
        "point_level": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "false_positive_rate": metrics["false_positive_rate"],
        },
        "event_level": {
            "events_detected_over_total": f"{detected_events}/{len(event_results)}",
            "event_level_recall": event_metrics["event_level_recall"],
        },
    }
    save_figure(scored, event_results, threshold, Path(figure_path))

    protected_after = _snapshot(protected_paths)
    joblib_after = _snapshot([path for path in (ROOT / "models").rglob("*.joblib")])
    validation = {
        "frozen_detector_hashes_verified": all(
            detector_verification[name]["sha256"] == sha256_file(_resolve(detector_verification[name]["path"]))
            for name in ("detector_config", "detector_metadata", "scaler", "model")
        ),
        "frozen_threshold_hashes_verified": threshold_verification["value"] == threshold,
        "no_scaler_or_model_fit_calls": detector_verification["fit_called"] is False,
        "exact_337_timestamps": len(scored) == EXPECTED_ROWS and scored.Timestamp.equals(synthetic.Timestamp),
        "exact_frozen_seven_feature_order": tuple(synthetic.columns[1:8]) == FEATURE_ORDER,
        "strict_greater_than_prediction_rule": np.array_equal(predicted, (synthetic_scores > threshold).astype(np.int8)),
        "confusion_matrix_sums_to_337": sum(confusion.values()) == EXPECTED_ROWS,
        "confusion_matrix_arithmetic": confusion["true_positive"] + confusion["false_negative"] == int(scored.synthetic_anomaly.sum()) and confusion["false_positive"] + confusion["true_negative"] == int(scored.synthetic_anomaly.eq(0).sum()),
        "propagated_rows_retain_reference_zero": bool(scored.loc[scored[EFFECT_COLUMN].eq("propagated"), "synthetic_anomaly"].eq(0).all()),
        "strict_event_definition_applied": all(event["strict_event_detected"] == (event["direct_rows_flagged"] > 0) for event in event_results),
        "paired_baseline_timestamps_match": comparison.Timestamp.equals(baseline.Timestamp),
        "prediction_file_columns_exact": scored.columns.tolist() == list(PREDICTION_COLUMNS),
        "threshold_unchanged": sha256_file(FROZEN_THRESHOLD_CONFIG) == threshold_before,
        "synthetic_dataset_unchanged": sha256_file(synthetic_input_path) == synthetic_before,
        "protected_sources_unchanged": protected_before == protected_after,
        "no_retraining_or_new_model_artifact": joblib_before == joblib_after,
        "no_synthetic_regeneration": sha256_file(synthetic_input_path) == metadata["no_time_7_output"]["sha256"],
    }
    if not all(validation.values()):
        failed = [name for name, passed in validation.items() if not passed]
        raise RuntimeError(f"Day 22 validation failed: {failed}")

    report = {
        "stage": "Day 22 frozen anomaly-detector evaluation on final synthetic data",
        "frozen_configuration": {
            "feature_order": list(FEATURE_ORDER),
            "threshold": threshold,
            "threshold_rule": "anomaly_score > threshold",
            "score_convention": "higher score = more anomalous; score = -model.score_samples",
            "detector_verification": detector_verification,
            "threshold_verification": threshold_verification,
            "detector_metadata_status": detector_metadata["status"],
        },
        "inputs": {
            "synthetic": {"path": _relative(synthetic_input_path), "sha256": sha256_file(synthetic_input_path), "shape": list(synthetic.shape)},
            "paired_baseline": {"path": _relative(baseline_input_path), "sha256": sha256_file(baseline_input_path), "rows": len(baseline)},
            "synthetic_metadata": {"path": _relative(synthetic_metadata_path), "sha256": sha256_file(synthetic_metadata_path)},
        },
        "outputs": {
            "predictions": {"path": _relative(predictions_path), "sha256": sha256_file(predictions_path)},
            "paired_score_comparison": {"path": _relative(comparison_path), "sha256": sha256_file(comparison_path)},
            "figure": _relative(figure_path),
        },
        "direct_label_definition": "synthetic_anomaly=1 only for deliberately modified original measurements; propagated rows remain reference-negative",
        "confusion_matrix": confusion,
        "point_metrics": metrics,
        "predicted_alert_context": context_alerts,
        "strict_point_metric_limitation": "alerts on propagated-effect rows count as false positives in the strict direct-row confusion matrix",
        "event_metrics": event_metrics,
        "per_event_results": event_results,
        "baseline_vs_synthetic_score_differences": {
            "overall": score_delta_summary(comparison.score_delta),
            "by_effect_context": delta_by_context,
        },
        "detector_alert_events": {
            "event_grouping_rule": "predicted rows exactly two minutes apart belong to the same detector alert event; no smoothing or distant merging",
            "count": len(alert_events),
            "events": alert_events,
        },
        "protected_transition_review": protected_review,
        "performance_table": performance_table,
        "validation": {"passed": True, "checks": validation},
        "protected_artifact_hashes": {
            path: {"before": digest, "after": protected_after[path], "unchanged": protected_after[path] == digest}
            for path, digest in protected_before.items()
        },
        "limitations": [
            "only three synthetic events were successfully placed",
            "synthetic signatures are controlled engineering cases, not verified real photovoltaic faults",
            "strict point metrics count propagated-tail alerts as false positives",
            "the source spans approximately 33.6 hours and one complete calendar day",
            "the threshold remains an internship-prototype calibration, not a validated operational alarm limit",
        ],
        "explicitly_not_performed": [
            "model retraining", "scaler refitting", "feature changes", "threshold changes",
            "synthetic event regeneration", "parameter tuning", "computer vision", "dashboard work",
        ],
        "required_interpretation": (
            "The evaluation uses controlled synthetic anomalies and only three successfully placed events. "
            "The results therefore demonstrate prototype behavior and do not establish real-world photovoltaic fault-detection performance."
        ),
    }
    _atomic_json(report, Path(report_path))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-input", type=Path, default=SYNTHETIC_INPUT)
    parser.add_argument("--baseline-input", type=Path, default=BASELINE_INPUT)
    parser.add_argument("--synthetic-metadata", type=Path, default=SYNTHETIC_METADATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_evaluation(
            synthetic_input_path=args.synthetic_input,
            baseline_input_path=args.baseline_input,
            synthetic_metadata_path=args.synthetic_metadata,
            predictions_path=args.predictions,
            comparison_path=args.comparison,
            report_path=args.report,
            figure_path=args.figure,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, yaml.YAMLError) as error:
        print(f"Error: {error}")
        return 1
    matrix = report["confusion_matrix"]
    metrics = report["point_metrics"]
    events = report["event_metrics"]
    print(
        f"Predicted alerts: {matrix['true_positive'] + matrix['false_positive']} | "
        f"TP={matrix['true_positive']} FP={matrix['false_positive']} "
        f"TN={matrix['true_negative']} FN={matrix['false_negative']}"
    )
    print(f"Precision={metrics['precision']:.6f} Recall={metrics['recall']:.6f} F1={metrics['f1']:.6f}")
    print(f"Events detected: {events['detected_events']}/{events['total_events']}")
    print(f"Validation passed: {report['validation']['passed']}")
    print("Frozen detector and threshold were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
