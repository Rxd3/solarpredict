"""Evaluate a trained YOLO detector on an explicitly selected dataset split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.computer_vision.inspect_vision_dataset import DEFAULT_DATA_YAML, load_dataset_yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models/computer_vision/solar_panel_detector_best.pt"
DEFAULT_PROJECT = ROOT / "outputs/computer_vision/evaluation"
DEFAULT_METRICS = ROOT / "outputs/computer_vision/validation_metrics.json"
DEFAULT_PREDICTIONS = ROOT / "outputs/computer_vision/validation_prediction_summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list):
        values = [values]
    return [float(value) for value in values]


def serialize_metrics(metrics: Any, class_names: tuple[str, ...]) -> dict[str, Any]:
    """Convert Ultralytics detection metrics into stable JSON-friendly fields."""

    box = metrics.box
    class_ids = [int(value) for value in _float_list(box.ap_class_index)]
    precision = _float_list(box.p)
    recall = _float_list(box.r)
    map50 = _float_list(box.ap50)
    map50_95 = _float_list(box.maps)
    per_class = []
    for position, class_id in enumerate(class_ids):
        per_class.append({
            "class_id": class_id,
            "class_name": class_names[class_id],
            "precision": precision[position],
            "recall": recall[position],
            "map50": map50[position],
            "map50_95": map50_95[position],
        })
    confusion = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
    return {
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "per_class": per_class,
        "speed_milliseconds_per_image": {
            str(key): float(value) for key, value in (metrics.speed or {}).items()
        },
        "results_dict": {
            str(key): float(value) for key, value in metrics.results_dict.items()
        },
        "confusion_matrix": {
            "orientation": "rows are predicted classes; columns are true classes",
            "labels": [*class_names, "background"],
            "counts": confusion.tolist() if confusion is not None else None,
        },
    }


def export_prediction_summary(
    model: YOLO,
    image_root: Path,
    output_path: Path,
    *,
    class_names: tuple[str, ...],
    image_size: int,
    device: str,
) -> dict[str, Any]:
    """Run inference over one selected split and export bounded detections."""

    images: list[dict[str, Any]] = []
    detection_count = 0
    results = model.predict(
        source=str(image_root), imgsz=image_size, device=device,
        stream=True, save=False, verbose=False,
    )
    for result in results:
        height, width = (int(value) for value in result.orig_shape)
        detections = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            for coordinates, class_value, confidence in zip(xyxy, classes, confidences):
                class_id = int(class_value)
                x1, y1, x2, y2 = (float(value) for value in coordinates)
                detections.append({
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "confidence": float(confidence),
                    "xyxy": [x1, y1, x2, y2],
                    "within_image_bounds": bool(
                        0.0 <= x1 <= x2 <= width and 0.0 <= y1 <= y2 <= height
                    ),
                })
        detection_count += len(detections)
        images.append({
            "image_path": str(Path(result.path).resolve()),
            "image_width": width,
            "image_height": height,
            "detections": detections,
        })
    summary = {
        "split_image_root": str(image_root.resolve()),
        "image_count": len(images),
        "detection_count": detection_count,
        "images": images,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def evaluate_detector(
    weights: Path,
    data_yaml: Path = DEFAULT_DATA_YAML,
    *,
    split: str = "val",
    metrics_output: Path | None = None,
    predictions_output: Path | None = None,
    image_size: int = 512,
    batch_size: int = 8,
    device: str = "cpu",
    workers: int = 0,
) -> dict[str, Any]:
    """Evaluate one checkpoint without using any unrequested dataset split."""

    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(
            f"Trained weights not found: {weights}. Complete a future YOLO training stage first."
        )
    dataset = load_dataset_yaml(data_yaml)
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be 'val' or 'test'.")
    image_root = dataset.split_paths[split]
    if image_root is None or not image_root.is_dir():
        raise ValueError(f"Dataset data.yaml does not configure an available '{split}' split.")

    model = YOLO(str(weights), task="detect")
    run_name = f"{split}_evaluation"
    metrics = model.val(
        data=str(dataset.yaml_path), split=split, imgsz=image_size, batch=batch_size,
        device=device, workers=workers, project=str(DEFAULT_PROJECT), name=run_name,
        exist_ok=True, plots=True, save_json=False,
    )
    report = {
        "checkpoint": str(weights),
        "checkpoint_sha256": _sha256(weights),
        "dataset_yaml": str(dataset.yaml_path),
        "evaluated_split": split,
        "test_split_used": split == "test",
        "class_names": list(dataset.class_names),
        **serialize_metrics(metrics, dataset.class_names),
    }
    if predictions_output is not None:
        prediction_summary = export_prediction_summary(
            model, image_root, Path(predictions_output), class_names=dataset.class_names,
            image_size=image_size, device=device,
        )
        report["prediction_summary"] = {
            "path": str(Path(predictions_output).resolve()),
            "image_count": prediction_summary["image_count"],
            "detection_count": prediction_summary["detection_count"],
        }
    if metrics_output is not None:
        destination = Path(metrics_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    try:
        report = evaluate_detector(
            args.weights, args.data, split=args.split,
            metrics_output=args.metrics_output, predictions_output=args.predictions_output,
            image_size=args.image_size, batch_size=args.batch_size,
            device=args.device, workers=args.workers,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Evaluation blocked: {error}\n")
    print(json.dumps(report["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
