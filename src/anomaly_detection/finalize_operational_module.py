"""Create Day 23 dashboard example and final status without training anything."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.anomaly_detection.inference import (
    FEATURE_ORDER,
    group_alert_events,
    load_frozen_detector,
    score_dataframe,
)


ROOT = Path(__file__).resolve().parents[2]
BASELINE_INPUT = ROOT / "data/model_ready/baseline_evaluation.csv"
DAY22_REPORT = ROOT / "outputs/final_anomaly_detector_evaluation.json"
DEFAULT_FEED = ROOT / "outputs/dashboard_anomaly_feed_example.csv"
DEFAULT_STATUS = ROOT / "outputs/anomaly_module_final_status.json"
EXPECTED_DAY22_HASHES = {
    "outputs/final_anomaly_detector_evaluation.json": "e275debda93cbc9196109c3a8b06c822768d6428c9e4d86e9e309980045dbe4a",
    "outputs/final_synthetic_anomaly_predictions.csv": "6ea5fa316221317c5e50257df14227651f333b929f347c35b36551be17adc177",
    "outputs/final_synthetic_score_comparison.csv": "1b7937abe09be6b46404d6fd2c2a518d18ea1273cf9fb64cc05c4f0f3c663b56",
    "outputs/final_synthetic_evaluation_metadata.json": "01364f6158b902c0cd727b55637234b3a34d9a15a43107771bc850621a9cbcba",
}
FEED_COLUMNS = (
    "Timestamp",
    "Power_Generated",
    "Solar_Radiation",
    "anomaly_score",
    "threshold",
    "threshold_margin",
    "status",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_day22() -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in EXPECTED_DAY22_HASHES.items():
        value = sha256_file(ROOT / relative)
        if value != expected:
            raise ValueError(f"Protected Day 22 artifact changed: {relative}")
        actual[relative] = value
    return actual


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".csv.tmp",
        dir=destination.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False, lineterminator="\n", float_format="%.17g")
    temporary.replace(destination)


def _atomic_json(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".json.tmp",
        dir=destination.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(destination)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_outputs(
    feed_path: Path = DEFAULT_FEED,
    status_path: Path = DEFAULT_STATUS,
) -> dict[str, Any]:
    """Score untouched baseline data and write interface/status artifacts only."""

    protected_before = _verify_day22()
    detector = load_frozen_detector()
    baseline = pd.read_csv(BASELINE_INPUT)
    scored = score_dataframe(baseline, detector)
    feed = scored.loc[:, FEED_COLUMNS]
    _atomic_csv(feed, feed_path)
    events = group_alert_events(feed)

    day22 = json.loads(DAY22_REPORT.read_text(encoding="utf-8"))
    detector_config = yaml.safe_load(
        (ROOT / "config/frozen_anomaly_detector.yaml").read_text(encoding="utf-8")
    )
    status = {
        "status": "operational_module_complete_for_prototype",
        "frozen_configuration": {
            "features": list(FEATURE_ORDER),
            "model": "IsolationForest",
            "model_parameters": detector_config["model"]["parameters"],
            "threshold": detector.threshold,
            "threshold_rule": "anomaly_score > threshold",
            "score_convention": "anomaly_score = -IsolationForest.score_samples; higher is more isolated",
        },
        "day22_evaluation_summary": {
            "rows": day22["inputs"]["synthetic"]["shape"][0],
            "directly_modified_rows": day22["confusion_matrix"]["false_negative"],
            "synthetic_events": day22["event_metrics"]["total_events"],
            "predicted_alert_rows": sum(day22["predicted_alert_context"].values()),
            "detected_events": day22["event_metrics"]["detected_events"],
            "false_positive_rate": day22["point_metrics"]["false_positive_rate"],
            "interpretation": "moderate controlled synthetic evaluation; not evidence of real fault performance",
        },
        "known_limitations": [
            "not validated for real photovoltaic fault classification",
            "did not detect any of the three moderate Day 22 synthetic events",
            "training coverage is approximately 33.6 hours at one source/site",
            "seasonal and cross-site robustness are unverified",
            "RTD sensor placement and units are unconfirmed",
            "the seven-feature model does not consume rolling or change features",
            "the threshold is a conservative prototype limit, not a generally validated alarm boundary",
        ],
        "inference_interface": {
            "module": "src/anomaly_detection/inference.py",
            "functions": [
                "load_frozen_detector",
                "score_observation",
                "score_dataframe",
                "group_alert_events",
            ],
        },
        "dashboard_feed": {
            "path": _display_path(feed_path),
            "source": "untouched evaluation baseline",
            "rows": len(feed),
            "alert_rows": int(feed.status.eq("ALERT").sum()),
            "alert_events": len(events),
            "schema": list(FEED_COLUMNS),
            "sha256": sha256_file(feed_path),
            "alerts_preserved_without_fabrication": True,
        },
        "artifact_hashes": {
            "scaler": detector.verification["scaler"],
            "model": detector.verification["model"],
            "detector_config": detector.verification["detector_config"],
            "detector_metadata": detector.verification["detector_metadata"],
            "threshold_config": detector.verification["threshold_config"],
            "threshold_metadata": detector.verification["threshold_metadata"],
            "protected_day22_outputs": protected_before,
        },
        "validation": {
            "frozen_artifacts_hash_verified": True,
            "day22_files_unchanged": protected_before == _verify_day22(),
            "fit_called": False,
            "strict_threshold_rule": True,
            "feed_matches_inference_output": True,
            "new_training_performed": False,
        },
    }
    _atomic_json(status, status_path)
    return status


def main() -> int:
    status = build_outputs()
    print(json.dumps({
        "status": status["status"],
        "dashboard_feed": status["dashboard_feed"],
        "validation": status["validation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
