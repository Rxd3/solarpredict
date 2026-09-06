"""Finalize one completed YOLO run into reproducible prototype artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import ultralytics
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config/computer_vision_training.yaml"
DEFAULT_DATASET_SUMMARY = ROOT / "outputs/computer_vision/dataset_summary.json"
DEFAULT_VALIDATION_METRICS = ROOT / "outputs/computer_vision/validation_metrics.json"
DEFAULT_SELECTED_WEIGHTS = ROOT / "models/computer_vision/solar_panel_detector_best.pt"
DEFAULT_METADATA = ROOT / "models/computer_vision/solar_panel_detector_metadata.json"
DEFAULT_TRAINING_SUMMARY = ROOT / "outputs/computer_vision/training_summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_metrics(row: pd.Series) -> dict[str, float | int]:
    fields = (
        "train/box_loss", "train/cls_loss", "train/dfl_loss",
        "val/box_loss", "val/cls_loss", "val/dfl_loss",
        "metrics/precision(B)", "metrics/recall(B)",
        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    )
    return {"epoch": int(row["epoch"]), **{field: float(row[field]) for field in fields}}


def finalize_training(
    config_path: Path = DEFAULT_CONFIG,
    validation_metrics_path: Path = DEFAULT_VALIDATION_METRICS,
    *,
    selected_weights: Path = DEFAULT_SELECTED_WEIGHTS,
    metadata_path: Path = DEFAULT_METADATA,
    summary_path: Path = DEFAULT_TRAINING_SUMMARY,
) -> dict[str, Any]:
    """Copy the best weights and write metadata from real run artifacts only."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    run_dir = (
        ROOT / config["output_project"] / config["run_name"]
        if not Path(config["output_project"]).is_absolute()
        else Path(config["output_project"]) / config["run_name"]
    ).resolve()
    results_path = run_dir / "results.csv"
    args_path = run_dir / "args.yaml"
    best_path = run_dir / "weights/best.pt"
    last_path = run_dir / "weights/last.pt"
    for required in (args_path, results_path, best_path, last_path, validation_metrics_path):
        if not Path(required).is_file():
            raise FileNotFoundError(f"Required completed-run artifact not found: {required}")

    results = pd.read_csv(results_path)
    results.columns = [column.strip() for column in results.columns]
    if results.empty:
        raise ValueError("Training results contain no completed epochs.")
    requested_epochs = int(config["epochs"])
    completed_epochs = int(results["epoch"].max())
    fitness = 0.1 * results["metrics/mAP50(B)"] + 0.9 * results["metrics/mAP50-95(B)"]
    best_position = int(fitness.to_numpy().argmax())
    best_row = results.iloc[best_position]
    last_row = results.iloc[-1]
    best_epoch = int(best_row["epoch"])
    training_seconds = float(last_row["time"])
    started_timestamp = args_path.stat().st_mtime
    completed_timestamp = max(results_path.stat().st_mtime, last_path.stat().st_mtime)
    wall_clock_seconds = completed_timestamp - started_timestamp

    selected_weights = Path(selected_weights)
    selected_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, selected_weights)
    if _sha256(selected_weights) != _sha256(best_path):
        raise OSError("Selected checkpoint hash does not match the original best.pt.")

    dataset = json.loads(DEFAULT_DATASET_SUMMARY.read_text(encoding="utf-8"))
    validation = json.loads(Path(validation_metrics_path).read_text(encoding="utf-8"))
    per_class = validation["per_class"]
    ranked = sorted(per_class, key=lambda item: item["map50_95"], reverse=True)
    checkpoints = {
        "run_best": {"path": str(best_path), "sha256": _sha256(best_path)},
        "run_last": {"path": str(last_path), "sha256": _sha256(last_path)},
        "selected_best_copy": {
            "path": str(selected_weights.resolve()), "sha256": _sha256(selected_weights),
        },
    }
    early_stopped = completed_epochs < requested_epochs
    common = {
        "source_pretrained_model": config["pretrained_model"],
        "dataset": {
            "version": 1,
            "yaml_path": dataset["dataset_yaml"],
            "active_split_images": {
                split: dataset["splits"][split]["images"] for split in ("train", "val", "test")
            },
            "total_active_images": dataset["total_images"],
            "total_bounding_boxes": dataset["total_bounding_boxes"],
        },
        "class_names": dataset["class_names"],
        "training_configuration": config,
        "random_seed": int(config["random_seed"]),
        "epochs_requested": requested_epochs,
        "epochs_completed": completed_epochs,
        "best_epoch": best_epoch,
        "training_duration_seconds": training_seconds,
        "wall_clock_training_duration_seconds": wall_clock_seconds,
        "training_started_at": datetime.fromtimestamp(started_timestamp).astimezone().isoformat(),
        "training_completed_at": datetime.fromtimestamp(completed_timestamp).astimezone().isoformat(),
        "average_seconds_per_completed_epoch": training_seconds / completed_epochs,
        "early_stopping": {
            "occurred": early_stopped,
            "reason": (
                f"validation fitness did not improve for patience={config['patience']} epochs"
                if early_stopped else "maximum requested epochs completed"
            ),
        },
        "model_selection_split": "val",
        "test_split_used_for_training_or_model_selection": False,
        "checkpoints": checkpoints,
        "package_versions": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
    }
    metadata = {
        "artifact_status": "trained_prototype",
        **common,
    }
    training_summary = {
        **common,
        "hardware": {"device": str(config["device"]), "cuda_available": torch.cuda.is_available()},
        "overall_validation_metrics": validation["overall"],
        "per_class_validation_metrics": per_class,
        "strongest_classes_by_map50_95": [item["class_name"] for item in ranked[:2]],
        "weakest_classes_by_map50_95": [item["class_name"] for item in ranked[-2:]],
        "losses_and_epoch_metrics": {
            "best_epoch": _row_metrics(best_row),
            "final_epoch": _row_metrics(last_row),
            "minimum_training_box_loss": float(results["train/box_loss"].min()),
            "minimum_validation_box_loss": float(results["val/box_loss"].min()),
            "minimum_training_classification_loss": float(results["train/cls_loss"].min()),
            "minimum_validation_classification_loss": float(results["val/cls_loss"].min()),
        },
        "validation_metrics_artifact": str(Path(validation_metrics_path).resolve()),
    }
    metadata_path = Path(metadata_path)
    summary_path = Path(summary_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(training_summary, indent=2) + "\n", encoding="utf-8")
    return training_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validation-metrics", type=Path, default=DEFAULT_VALIDATION_METRICS)
    parser.add_argument("--selected-weights", type=Path, default=DEFAULT_SELECTED_WEIGHTS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--summary", type=Path, default=DEFAULT_TRAINING_SUMMARY)
    args = parser.parse_args()
    try:
        summary = finalize_training(
            args.config, args.validation_metrics, selected_weights=args.selected_weights,
            metadata_path=args.metadata, summary_path=args.summary,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.exit(2, f"Training finalization blocked: {error}\n")
    print(json.dumps({
        "epochs_completed": summary["epochs_completed"],
        "best_epoch": summary["best_epoch"],
        "training_duration_seconds": summary["training_duration_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
