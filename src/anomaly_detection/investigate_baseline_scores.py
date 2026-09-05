"""Investigate Day 16 baseline-score distribution shift without refitting.

This read-only diagnostic loads the saved scaler and Isolation Forest. It does
not fit either artifact, choose a threshold, create labels, or generate data.
"""

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

from src.anomaly_detection.train_baseline_detector import (
    CORE_UNSUPERVISED_FEATURES,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODELS_DIRECTORY,
    DEFAULT_OUTPUTS_DIRECTORY,
    METADATA_FILENAME,
    MODEL_FILENAME,
    PARTITION_ORDER,
    SCALER_FILENAME,
    SCORE_FILENAMES,
    SYNTHETIC_LABEL_COLUMNS,
    TRANSITION_TIMESTAMPS,
    load_core_partitions,
    load_model_config,
)
from src.data_processing.engineer_basic_features import PROCESSED_TIMESTAMP_FORMAT


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = ROOT / "outputs/baseline_score_investigation.json"
DEFAULT_FIGURE_PATH = ROOT / "outputs/figures/operational/day17_score_context_review.png"
CONTEXT_THRESHOLD_W_M2 = 5.0
CONTEXT_FEATURES = (
    "rtd_mean",
    "Solar_Radiation",
    "Power_Generated",
    "Air_Temp",
    "Relative_Humidity",
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


def load_saved_scores(
    partitions: dict[str, pd.DataFrame],
    outputs_directory: Path = DEFAULT_OUTPUTS_DIRECTORY,
) -> dict[str, pd.DataFrame]:
    """Load the immutable Day 16 continuous-score files."""
    result: dict[str, pd.DataFrame] = {}
    for name in PARTITION_ORDER:
        path = Path(outputs_directory) / SCORE_FILENAMES[name]
        frame = pd.read_csv(path)
        if frame.columns.tolist() != ["Timestamp", "anomaly_score"]:
            raise ValueError(f"Unexpected score schema: {path}")
        frame["Timestamp"] = pd.to_datetime(
            frame["Timestamp"], format=PROCESSED_TIMESTAMP_FORMAT, errors="coerce"
        )
        frame["anomaly_score"] = pd.to_numeric(frame["anomaly_score"], errors="coerce")
        if not frame.Timestamp.equals(partitions[name].Timestamp):
            raise ValueError(f"Saved score timestamps differ for {name}.")
        if not np.isfinite(frame.anomaly_score).all():
            raise ValueError(f"Saved scores contain non-finite values for {name}.")
        result[name] = frame
    return result


def reproduce_saved_scores(
    partitions: dict[str, pd.DataFrame], scaler: Any, detector: Any,
    saved_scores: dict[str, pd.DataFrame],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transform/score with saved artifacts only; never call fit."""
    transformed: dict[str, np.ndarray] = {}
    checks: dict[str, Any] = {}
    for name in PARTITION_ORDER:
        raw = partitions[name].loc[:, CORE_UNSUPERVISED_FEATURES].to_numpy(dtype=float)
        scaled = scaler.transform(raw)
        reproduced = -detector.score_samples(scaled)
        transformed[name] = scaled
        absolute = np.abs(reproduced - saved_scores[name].anomaly_score.to_numpy())
        checks[name] = {
            "row_count": len(reproduced),
            "maximum_absolute_score_difference": float(absolute.max()),
            "exact_match": bool(np.array_equal(reproduced, saved_scores[name].anomaly_score.to_numpy())),
            "within_csv_precision": bool(np.allclose(
                reproduced, saved_scores[name].anomaly_score.to_numpy(), rtol=0, atol=1e-15
            )),
        }
    return transformed, checks


def feature_distribution_review(
    partitions: dict[str, pd.DataFrame], transformed: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Describe raw/scaled features and later-period training-range departures."""
    result: dict[str, Any] = {}
    training = partitions["training"]
    for index, feature in enumerate(CORE_UNSUPERVISED_FEATURES):
        train_min = float(training[feature].min())
        train_max = float(training[feature].max())
        per_partition: dict[str, Any] = {}
        for name in PARTITION_ORDER:
            raw = partitions[name][feature]
            scaled = transformed[name][:, index]
            per_partition[name] = {
                "raw": {
                    "minimum": float(raw.min()),
                    "maximum": float(raw.max()),
                    "mean": float(raw.mean()),
                    "median": float(raw.median()),
                    "population_standard_deviation": float(raw.std(ddof=0)),
                },
                "training_scaled": {
                    "minimum": float(scaled.min()),
                    "maximum": float(scaled.max()),
                    "mean": float(scaled.mean()),
                    "population_standard_deviation": float(scaled.std(ddof=0)),
                },
                "counts_relative_to_training": {
                    "below_training_minimum": int(raw.lt(train_min).sum()),
                    "above_training_maximum": int(raw.gt(train_max).sum()),
                    "absolute_scaled_value_above_2": int((np.abs(scaled) > 2).sum()),
                    "absolute_scaled_value_above_3": int((np.abs(scaled) > 3).sum()),
                    "absolute_scaled_value_above_5": int((np.abs(scaled) > 5).sum()),
                },
            }
        result[feature] = {
            "training_raw_range": {"minimum": train_min, "maximum": train_max},
            "partitions": per_partition,
        }
    return result


def rtd_std_review(
    partitions: dict[str, pd.DataFrame], transformed: dict[str, np.ndarray]
) -> dict[str, Any]:
    index = list(CORE_UNSUPERVISED_FEATURES).index("rtd_std")
    summary: dict[str, Any] = {}
    for name in PARTITION_ORDER:
        values = partitions[name]["rtd_std"]
        scaled = transformed[name][:, index]
        summary[name] = {
            "raw_minimum": float(values.min()),
            "raw_maximum": float(values.max()),
            "raw_median": float(values.median()),
            "raw_percentiles": {
                "90": float(values.quantile(0.90)),
                "95": float(values.quantile(0.95)),
                "97.5": float(values.quantile(0.975)),
                "99": float(values.quantile(0.99)),
            },
            "scaled_minimum": float(scaled.min()),
            "scaled_maximum": float(scaled.max()),
        }

    calibration = partitions["calibration"].copy()
    calibration["rtd_std_scaled"] = transformed["calibration"][:, index]
    calibration["light_context"] = np.where(
        calibration.Solar_Radiation.le(CONTEXT_THRESHOLD_W_M2), "low_light", "daylight_like"
    )
    highest: list[dict[str, Any]] = []
    for _, row in calibration.nlargest(10, "rtd_std").iterrows():
        time = row.Timestamp
        nearby = calibration.loc[
            calibration.Timestamp.between(
                time - pd.Timedelta(minutes=10), time + pd.Timedelta(minutes=10)
            )
            & calibration.Timestamp.ne(time),
            ["rtd_std", *CONTEXT_FEATURES],
        ]
        highest.append({
            "timestamp": _timestamp(time),
            "clock_time": time.strftime("%H:%M"),
            "rtd_std": float(row.rtd_std),
            "rtd_std_training_scaled": float(row.rtd_std_scaled),
            **{feature: float(row[feature]) for feature in CONTEXT_FEATURES},
            "light_context": row.light_context,
            "nearby_10_minute_means_excluding_observation": {
                feature: float(nearby[feature].mean()) for feature in nearby.columns
            },
        })
    daylight = calibration.Solar_Radiation.gt(CONTEXT_THRESHOLD_W_M2)
    summary["calibration_context_association"] = {
        "rtd_std_solar_radiation_pearson_correlation": float(
            calibration.rtd_std.corr(calibration.Solar_Radiation)
        ),
        "daylight_like_rtd_std_mean": float(calibration.loc[daylight, "rtd_std"].mean()),
        "low_light_rtd_std_mean": float(calibration.loc[~daylight, "rtd_std"].mean()),
        "top_10_daylight_like_count": sum(item["light_context"] == "daylight_like" for item in highest),
        "top_10_radiation_above_100_w_m2_count": sum(item["Solar_Radiation"] > 100 for item in highest),
        "interpretation_limit": "Descriptive association only; RTD placement and units remain unconfirmed.",
    }
    summary["highest_calibration_observations"] = highest
    return summary


def time_of_day_review(partitions: dict[str, pd.DataFrame]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    represented: dict[str, list[int]] = {}
    for name in PARTITION_ORDER:
        hourly = partitions[name].Timestamp.dt.hour.value_counts().reindex(range(24), fill_value=0)
        counts[name] = {str(hour): int(hourly.loc[hour]) for hour in range(24)}
        represented[name] = [hour for hour in range(24) if hourly.loc[hour] > 0]
    training_counts = counts["training"]
    comparison: dict[str, Any] = {}
    for name in ("calibration", "evaluation_baseline"):
        comparison[name] = {
            "hours_absent_from_training": [hour for hour in represented[name] if training_counts[str(hour)] == 0],
            "hours_not_fully_represented_in_training": [hour for hour in represented[name] if training_counts[str(hour)] < 30],
            "definition_of_not_fully_represented": "fewer than 30 two-minute observations in the training clock hour",
        }
    return {
        "observations_per_clock_hour": counts,
        "hours_represented": represented,
        "later_partition_comparison_to_training": comparison,
        "cyclical_feature_note": (
            "Calibration includes many clock hours absent from training, so sine/cosine time coordinates can contribute to isolation scores. "
            "They are retained unchanged for future comparison."
        ),
    }


def _score_stats(values: pd.Series) -> dict[str, Any]:
    return {
        "count": len(values), "minimum": float(values.min()), "maximum": float(values.max()),
        "mean": float(values.mean()), "median": float(values.median()),
        "population_standard_deviation": float(values.std(ddof=0)),
        "90th_percentile": float(values.quantile(0.90)),
        "95th_percentile": float(values.quantile(0.95)),
    }


def context_score_review(
    partitions: dict[str, pd.DataFrame], scores: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in PARTITION_ORDER:
        low = partitions[name].Solar_Radiation.le(CONTEXT_THRESHOLD_W_M2).to_numpy()
        values = scores[name].anomaly_score
        result[name] = {
            "daylight_like": _score_stats(values.loc[~low]),
            "low_light": _score_stats(values.loc[low]),
            "context_rule": "Solar_Radiation <= 5 W/m^2 defines low light for exploration only",
        }
    return result


def _safe_correlation(left: pd.Series, right: pd.Series, method: str) -> float | None:
    if left.nunique() <= 1 or right.nunique() <= 1:
        return None
    value = left.corr(right, method=method)
    return float(value) if pd.notna(value) and math.isfinite(value) else None


def diagnostic_frames(
    partitions: dict[str, pd.DataFrame], transformed: dict[str, np.ndarray],
    scores: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    correlations: dict[str, Any] = {}
    feature_names = np.asarray(CORE_UNSUPERVISED_FEATURES)
    for name in PARTITION_ORDER:
        absolute = np.abs(transformed[name])
        largest_index = absolute.argmax(axis=1)
        diagnostic = scores[name].copy()
        diagnostic["max_absolute_scaled_feature_value"] = absolute.max(axis=1)
        diagnostic["largest_absolute_scaled_feature"] = feature_names[largest_index]
        for boundary in (2, 3, 5):
            diagnostic[f"core_feature_count_abs_z_above_{boundary}"] = (absolute > boundary).sum(axis=1)
        diagnostic["light_context"] = np.where(
            partitions[name].Solar_Radiation.le(CONTEXT_THRESHOLD_W_M2),
            "low_light", "daylight_like",
        )
        frames[name] = diagnostic
        correlations[name] = {
            column: {
                "pearson": _safe_correlation(diagnostic[column], diagnostic.anomaly_score, "pearson"),
                "spearman": _safe_correlation(diagnostic[column], diagnostic.anomaly_score, "spearman"),
            }
            for column in [
                "max_absolute_scaled_feature_value",
                "core_feature_count_abs_z_above_2",
                "core_feature_count_abs_z_above_3",
                "core_feature_count_abs_z_above_5",
            ]
        }
    return frames, correlations


def top_score_observations(
    partitions: dict[str, pd.DataFrame], diagnostics: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    reported_features = (
        "Power_Generated", "Solar_Radiation", "Air_Temp", "Relative_Humidity",
        "Wind_Speed", "rtd_mean", "rtd_std",
    )
    for name in PARTITION_ORDER:
        records: list[dict[str, Any]] = []
        for index, row in diagnostics[name].nlargest(10, "anomaly_score").iterrows():
            source = partitions[name].loc[index]
            records.append({
                "timestamp": _timestamp(row.Timestamp),
                "clock_time": row.Timestamp.strftime("%H:%M"),
                "anomaly_score": float(row.anomaly_score),
                **{feature: float(source[feature]) for feature in reported_features},
                "light_context": row.light_context,
                "largest_absolute_scaled_feature": row.largest_absolute_scaled_feature,
                "largest_absolute_scaled_value": float(row.max_absolute_scaled_feature_value),
                "core_feature_count_abs_z_above_2": int(row.core_feature_count_abs_z_above_2),
                "core_feature_count_abs_z_above_3": int(row.core_feature_count_abs_z_above_3),
                "core_feature_count_abs_z_above_5": int(row.core_feature_count_abs_z_above_5),
                "interpretation": "high-score baseline observation; not a confirmed anomaly or fault",
            })
        result[name] = records
    return result


def recurring_transition_review(
    diagnostics: dict[str, pd.DataFrame], transformed: dict[str, np.ndarray]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for transition in TRANSITION_TIMESTAMPS:
        located = [name for name, frame in diagnostics.items() if frame.Timestamp.eq(transition).any()]
        if len(located) != 1:
            raise ValueError(f"Transition {transition} must occur exactly once.")
        name = located[0]
        frame = diagnostics[name]
        row_index = frame.index[frame.Timestamp.eq(transition)][0]
        row = frame.loc[row_index]
        scores = frame.anomaly_score
        nearby = frame.loc[
            frame.Timestamp.between(transition - pd.Timedelta(minutes=10), transition + pd.Timedelta(minutes=10))
            & frame.Timestamp.ne(transition), "anomaly_score"
        ]
        result[_timestamp(transition)] = {
            "partition": name,
            "anomaly_score": float(row.anomaly_score),
            "highest_score_rank_in_partition": int(scores.rank(method="min", ascending=False).loc[row_index]),
            "score_percentile_within_partition": float(scores.le(row.anomaly_score).mean() * 100),
            "largest_absolute_scaled_feature": row.largest_absolute_scaled_feature,
            "largest_absolute_scaled_value": float(row.max_absolute_scaled_feature_value),
            "nearby_10_minute_scores_excluding_transition": _score_stats(nearby),
            "interpretation": "recurring operational transition; not classified as a fault",
        }
    return result


def save_context_figure(
    partitions: dict[str, pd.DataFrame], scores: dict[str, pd.DataFrame], destination: Path
) -> None:
    combined = []
    for name in PARTITION_ORDER:
        frame = scores[name].copy()
        frame["low_light"] = partitions[name].Solar_Radiation.le(CONTEXT_THRESHOLD_W_M2).to_numpy()
        combined.append(frame)
    data = pd.concat(combined, ignore_index=True)
    fig, axis = plt.subplots(figsize=(13, 5))
    axis.plot(data.Timestamp, data.anomaly_score, color="#9aa0a6", linewidth=0.7, alpha=0.65)
    for low_light, label, color in (
        (False, "Daylight-like (radiation > 5 W/m²)", "#e6a23c"),
        (True, "Low light (radiation ≤ 5 W/m²)", "#355c8a"),
    ):
        subset = data.loc[data.low_light.eq(low_light)]
        axis.scatter(subset.Timestamp, subset.anomaly_score, s=11, color=color, label=label, alpha=0.8)
    for name in PARTITION_ORDER[1:]:
        axis.axvline(scores[name].Timestamp.iloc[0], color="#555", linestyle="--", linewidth=1)
    for transition in TRANSITION_TIMESTAMPS:
        axis.axvline(transition, color="#8b1e3f", linestyle=":", linewidth=1.2)
    axis.set_title("Baseline anomaly scores by exploratory light context (no threshold)")
    axis.set_xlabel("Source timestamp (timezone not specified)")
    axis.set_ylabel("Anomaly score = -score_samples")
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2, loc="upper center")
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


def run_investigation(
    *, config_path: Path = DEFAULT_CONFIG_PATH,
    models_directory: Path = DEFAULT_MODELS_DIRECTORY,
    outputs_directory: Path = DEFAULT_OUTPUTS_DIRECTORY,
    report_path: Path = DEFAULT_REPORT_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
) -> dict[str, Any]:
    """Create the Day 17 evidence report while protecting saved artifacts."""
    config = load_model_config(config_path)
    models_directory = Path(models_directory)
    outputs_directory = Path(outputs_directory)
    scaler_path = models_directory / SCALER_FILENAME
    model_path = models_directory / MODEL_FILENAME
    metadata_path = models_directory / METADATA_FILENAME
    score_paths = [outputs_directory / SCORE_FILENAMES[name] for name in PARTITION_ORDER]
    protected_paths = [
        *sorted(path for path in (ROOT / "data").rglob("*") if path.is_file()),
        *sorted(path for path in models_directory.rglob("*") if path.is_file()),
        *score_paths,
    ]
    before = _snapshot(protected_paths)
    scaler = joblib.load(scaler_path)
    detector = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    partitions = load_core_partitions(config)
    saved_scores = load_saved_scores(partitions, outputs_directory)
    transformed, reproduction = reproduce_saved_scores(partitions, scaler, detector, saved_scores)

    feature_review = feature_distribution_review(partitions, transformed)
    rtd_review = rtd_std_review(partitions, transformed)
    time_review = time_of_day_review(partitions)
    context_review = context_score_review(partitions, saved_scores)
    diagnostics, correlations = diagnostic_frames(partitions, transformed, saved_scores)
    top_scores = top_score_observations(partitions, diagnostics)
    transitions = recurring_transition_review(diagnostics, transformed)
    save_context_figure(partitions, saved_scores, Path(figure_path))

    after = _snapshot(protected_paths)
    validation = {
        "saved_scores_reproduced_with_existing_artifacts": all(item["within_csv_precision"] for item in reproduction.values()),
        "no_scaler_or_model_refitting_performed": True,
        "all_nine_core_features_reviewed": set(feature_review) == set(CORE_UNSUPERVISED_FEATURES),
        "partition_timestamps_unchanged": all(saved_scores[name].Timestamp.equals(partitions[name].Timestamp) for name in PARTITION_ORDER),
        "context_counts_cover_each_partition": all(
            context_review[name]["daylight_like"]["count"] + context_review[name]["low_light"]["count"] == len(partitions[name])
            for name in PARTITION_ORDER
        ),
        "no_anomaly_label_columns_created": all(not SYNTHETIC_LABEL_COLUMNS.intersection(frame.columns) for frame in diagnostics.values()),
        "all_protected_data_model_and_score_files_unchanged": before == after,
        "saved_artifact_feature_order_matches": metadata["feature_order"] == list(CORE_UNSUPERVISED_FEATURES),
    }
    if not all(validation.values()):
        failed = [name for name, value in validation.items() if not value]
        raise RuntimeError(f"Day 17 validation failed: {failed}")

    report = {
        "stage": "Day 17 baseline anomaly-score review and distribution-shift investigation",
        "scope": "descriptive baseline diagnostics using saved Day 16 artifacts only",
        "score_convention": "anomaly_score = -IsolationForest.score_samples(X_scaled); higher is more isolated",
        "artifact_reproduction": reproduction,
        "per_feature_distribution_review": feature_review,
        "rtd_std_review": rtd_review,
        "time_of_day_coverage": time_review,
        "daylight_low_light_score_review": context_review,
        "score_extremeness_correlations": correlations,
        "top_10_high_score_baseline_observations": top_scores,
        "recurring_transition_diagnostics": transitions,
        "recommendations_for_next_stage": [
            "Retain the current Day 16 model as a fixed reference; do not reinterpret high baseline scores as faults.",
            "Prioritize collecting more independent days with representative daylight and low-light coverage for training.",
            "Compare a prespecified model without minute_of_day_sin and minute_of_day_cos to quantify time-context sensitivity.",
            "Compare a prespecified model without rtd_std and separately review the five RTD channels before assigning physical meaning.",
            "Keep the current Core definition unchanged until controlled comparison models are evaluated on the same frozen partitions.",
        ],
        "threshold_planning_note": {
            "reason_to_defer": "Baseline score distributions differ substantially by chronological operating regime; selecting a threshold now could encode the short record's time coverage rather than a stable operating policy.",
            "future_strategies_to_compare_without_selection": [
                "a predeclared calibration-score percentile",
                "a robust calibration statistic such as median plus a multiple of MAD",
                "a threshold chosen to meet a predefined acceptable baseline false-positive rate",
            ],
            "threshold_selected": False,
        },
        "protected_artifact_hashes": {
            path: {"before": digest, "after": after[path], "unchanged": after[path] == digest}
            for path, digest in before.items()
        },
        "new_artifacts": {
            "report": _relative(Path(report_path)),
            "figure": _relative(Path(figure_path)),
        },
        "validation": {"passed": True, "checks": validation},
        "explicitly_not_performed": [
            "scaler fitting", "model fitting or replacement", "threshold selection",
            "final anomaly labels", "synthetic evaluation generation",
            "precision/recall/F1 calculation",
        ],
    }
    _atomic_json(report, Path(report_path))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--models-directory", type=Path, default=DEFAULT_MODELS_DIRECTORY)
    parser.add_argument("--outputs-directory", type=Path, default=DEFAULT_OUTPUTS_DIRECTORY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_investigation(
            config_path=args.config, models_directory=args.models_directory,
            outputs_directory=args.outputs_directory, report_path=args.report,
            figure_path=args.figure,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Error: {error}")
        return 1
    print("Saved Day 16 scores reproduced without fitting.")
    print(f"Core features reviewed: {len(report['per_feature_distribution_review'])}")
    print(f"Validation passed: {report['validation']['passed']}")
    print("No threshold, labels, new synthetic data, scaler fit, or model fit was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
