"""Explicit Ultralytics training entry point with a non-training dry-run mode."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from src.computer_vision.inspect_vision_dataset import inspect_dataset, load_dataset_yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config/computer_vision_training.yaml"
REQUIRED_FIELDS = (
    "pretrained_model", "dataset_yaml", "image_size", "epochs", "batch_size",
    "patience", "device", "workers", "random_seed", "output_project", "run_name",
    "save", "save_period",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def load_training_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Training configuration not found: {source}")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping.")
    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Training configuration is missing fields: {missing}")
    if config.get("status") not in {"prepared_not_trained", "trained_prototype"}:
        raise ValueError("Configuration status must be prepared_not_trained or trained_prototype.")
    if config.get("task") != "detect":
        raise ValueError("Configuration task must be detection.")
    for field in ("image_size", "epochs", "batch_size", "patience"):
        if int(config[field]) <= 0:
            raise ValueError(f"{field} must be positive.")
    if int(config["workers"]) < 0:
        raise ValueError("workers cannot be negative.")
    if not bool(config["save"]):
        raise ValueError("save must be true so best.pt and last.pt are retained.")
    if int(config["save_period"]) <= 0:
        raise ValueError("save_period must be positive for periodic resumable checkpoints.")
    device = str(config["device"]).lower()
    if device != "cpu" and not torch.cuda.is_available():
        raise ValueError(f"Configured device '{device}' requires CUDA, but CUDA is unavailable.")
    return config


def validate_training_setup(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    resolve_model: bool = True,
) -> tuple[dict[str, Any], YOLO | None]:
    """Validate config, hardware, dataset paths, and checkpoint without training."""

    config = load_training_config(config_path)
    data_yaml = _resolve(config["dataset_yaml"])
    checks: dict[str, Any] = {
        "config_loaded": True,
        "data_yaml": str(data_yaml),
        "data_yaml_loaded": False,
        "train_path_exists": False,
        "validation_path_exists": False,
        "test_path_exists": None,
        "model_checkpoint_resolved": False,
        "device_valid": True,
        "training_started": False,
    }
    errors: list[str] = []
    dataset = None
    try:
        dataset = load_dataset_yaml(data_yaml)
        inventory = inspect_dataset(dataset)
        checks["data_yaml_loaded"] = True
        checks["train_path_exists"] = bool(dataset.split_paths["train"].is_dir())
        checks["validation_path_exists"] = bool(dataset.split_paths["val"].is_dir())
        checks["test_path_exists"] = (
            None if dataset.split_paths["test"] is None else dataset.split_paths["test"].is_dir()
        )
        if not checks["train_path_exists"]:
            errors.append("Configured training image path does not exist.")
        if not checks["validation_path_exists"]:
            errors.append("Configured validation image path does not exist.")
        checks["split_image_counts"] = {
            key: inventory["splits"][key]["images"] for key in ("train", "val", "test")
        }
        checks["total_active_images"] = inventory["total_images"]
        checks["class_count"] = inventory["class_count"]
        checks["invalid_label_count"] = inventory["invalid_labels"]["count"]
        checks["empty_label_count"] = inventory["images_with_empty_label_files"]["count"]
        checks["image_without_label_count"] = inventory["images_without_labels"]["count"]
        checks["label_without_image_count"] = inventory["labels_without_images"]["count"]
        checks["unreadable_image_count"] = inventory["unreadable_images"]["count"]
        checks["cross_split_duplicate_image_groups"] = len(
            inventory["duplicates"]["exact_duplicate_images_across_splits"]
        )
        checks["cross_split_duplicate_filename_count"] = len(
            inventory["duplicates"]["duplicate_filenames_across_splits"]
        )
        checks["all_classes_present_in_every_split"] = inventory[
            "all_classes_present_in_every_split"
        ]
        quality_counts = {
            "invalid label lines": checks["invalid_label_count"],
            "empty label files": checks["empty_label_count"],
            "images without labels": checks["image_without_label_count"],
            "labels without images": checks["label_without_image_count"],
            "unreadable images": checks["unreadable_image_count"],
            "cross-split duplicate image groups": checks["cross_split_duplicate_image_groups"],
            "cross-split duplicate filenames": checks["cross_split_duplicate_filename_count"],
        }
        errors.extend(f"Active dataset has {count} {name}." for name, count in quality_counts.items() if count)
        if not checks["all_classes_present_in_every_split"]:
            errors.append("Not all configured classes are represented in every active split.")
    except (FileNotFoundError, ValueError) as error:
        errors.append(str(error))

    model = None
    if resolve_model:
        try:
            model = YOLO(str(config["pretrained_model"]), task="detect")
            checks["model_checkpoint_resolved"] = True
        except Exception as error:  # Ultralytics may raise network/checkpoint-specific errors.
            errors.append(f"Could not resolve pretrained checkpoint: {error}")
    checks["checkpoint_saving_enabled"] = bool(config["save"])
    checks["checkpoint_save_period_epochs"] = int(config["save_period"])
    run_root = _resolve(config["output_project"]) / str(config["run_name"]) / "weights"
    checks["expected_best_checkpoint"] = str(run_root / "best.pt")
    checks["expected_last_checkpoint"] = str(run_root / "last.pt")
    checks["ready_for_training"] = not errors and all([
        checks["data_yaml_loaded"], checks["train_path_exists"],
        checks["validation_path_exists"], checks["model_checkpoint_resolved"],
        checks["device_valid"],
    ])
    report = {
        "mode": "dry_run",
        "configuration": config,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "planned_training_device": str(config["device"]),
        },
        "dataset_class_names": list(dataset.class_names) if dataset else None,
        "checks": checks,
        "errors": errors,
    }
    return report, model


def run_training(config_path: str | Path = DEFAULT_CONFIG, *, resume: bool = False) -> Any:
    """Start or resume fitting only through an explicit non-dry-run call."""

    report, model = validate_training_setup(config_path, resolve_model=True)
    if not report["checks"]["ready_for_training"] or model is None:
        raise RuntimeError(f"Training setup is not ready: {report['errors']}")
    config = report["configuration"]
    if config["status"] == "trained_prototype":
        raise RuntimeError(
            "The configured prototype run is already complete; refusing to start or resume it."
        )
    if resume:
        last_checkpoint = (
            _resolve(config["output_project"]) / str(config["run_name"]) / "weights/last.pt"
        )
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"Cannot resume; last checkpoint not found: {last_checkpoint}")
        resume_model = YOLO(str(last_checkpoint), task="detect")
        return resume_model.train(resume=True)
    return model.train(
        data=str(_resolve(config["dataset_yaml"])),
        imgsz=int(config["image_size"]),
        epochs=int(config["epochs"]),
        batch=int(config["batch_size"]),
        patience=int(config["patience"]),
        device=str(config["device"]),
        workers=int(config["workers"]),
        seed=int(config["random_seed"]),
        deterministic=bool(config.get("deterministic", True)),
        save=bool(config["save"]),
        save_period=int(config["save_period"]),
        project=str(_resolve(config["output_project"])),
        name=str(config["run_name"]),
        exist_ok=bool(config.get("exist_ok", False)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate only; never call model.train().")
    mode.add_argument("--resume", action="store_true", help="Resume from the configured run's last.pt.")
    args = parser.parse_args()
    if args.dry_run:
        try:
            report, _ = validate_training_setup(args.config)
        except (FileNotFoundError, ValueError) as error:
            parser.exit(2, f"Dry run failed: {error}\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["checks"]["ready_for_training"] else 2
    try:
        run_training(args.config, resume=args.resume)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Training blocked: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
