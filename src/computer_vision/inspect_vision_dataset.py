"""Inspect and validate an Ultralytics-compatible YOLO detection dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import torch
import ultralytics
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_YAML = ROOT / "data/vision/solar_panel_defects/data.yaml"
DEFAULT_SUMMARY = ROOT / "outputs/computer_vision/dataset_summary.json"
TRAINING_CONFIG = ROOT / "config/computer_vision_training.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_KEYS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetConfig:
    yaml_path: Path
    dataset_root: Path
    split_paths: dict[str, Path | None]
    class_names: tuple[str, ...]
    raw: dict[str, Any]


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_dataset_yaml(path: str | Path = DEFAULT_DATA_YAML) -> DatasetConfig:
    """Load data.yaml and validate its real class mapping and split paths."""

    yaml_path = Path(path).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {yaml_path}. Download the Roboflow YOLO export first."
        )
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("data.yaml must contain a YAML mapping.")
    names = raw.get("names")
    if isinstance(names, list):
        class_names = tuple(str(name) for name in names)
    elif isinstance(names, dict):
        try:
            converted = {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("Class mapping keys in data.yaml must be integer IDs.") from error
        expected = list(range(len(converted)))
        if sorted(converted) != expected:
            raise ValueError(f"Class IDs must be consecutive from 0; found {sorted(converted)}.")
        class_names = tuple(converted[index] for index in expected)
    else:
        raise ValueError("data.yaml must define class names as a list or ID-to-name mapping.")
    if not class_names or any(not name.strip() for name in class_names):
        raise ValueError("data.yaml must define at least one non-empty class name.")
    if "nc" in raw and int(raw["nc"]) != len(class_names):
        raise ValueError(f"data.yaml nc={raw['nc']} disagrees with {len(class_names)} names.")

    dataset_root = _resolve(yaml_path.parent, raw.get("path", "."))
    split_paths: dict[str, Path | None] = {}
    for key in SPLIT_KEYS:
        value = raw.get(key)
        if value is None:
            split_paths[key] = None
        elif not isinstance(value, (str, Path)):
            raise ValueError(f"data.yaml '{key}' must be one path string for this project.")
        else:
            split_paths[key] = _resolve(dataset_root, value)
    if split_paths["train"] is None or split_paths["val"] is None:
        raise ValueError("data.yaml must define both train and val paths.")
    return DatasetConfig(yaml_path, dataset_root, split_paths, class_names, raw)


def validate_yolo_line(
    line: str,
    *,
    class_count: int,
    label_path: str | Path | None = None,
    line_number: int | None = None,
) -> tuple[int, tuple[float, float, float, float]]:
    """Validate one YOLO ``class x_center y_center width height`` line."""

    location = f"{label_path}:{line_number}" if label_path is not None else "label line"
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"{location}: expected exactly 5 values, found {len(parts)}.")
    try:
        values = [float(part) for part in parts]
    except ValueError as error:
        raise ValueError(f"{location}: all five values must be numeric.") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{location}: values must be finite.")
    if not values[0].is_integer():
        raise ValueError(f"{location}: class ID must be an integer.")
    class_id = int(values[0])
    if not 0 <= class_id < class_count:
        raise ValueError(f"{location}: class ID {class_id} is outside 0..{class_count - 1}.")
    x_center, y_center, width, height = values[1:]
    if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
        raise ValueError(f"{location}: box center coordinates must be within [0, 1].")
    if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
        raise ValueError(f"{location}: box width and height must be within (0, 1].")
    return class_id, (x_center, y_center, width, height)


def parse_label_file(path: Path, class_count: int) -> tuple[list[tuple[int, tuple[float, ...]]], list[str]]:
    boxes: list[tuple[int, tuple[float, ...]]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        return boxes, [f"{path}: unreadable label file: {error}"]
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            boxes.append(validate_yolo_line(
                line, class_count=class_count, label_path=path, line_number=number
            ))
        except ValueError as error:
            errors.append(str(error))
    return boxes, errors


def _image_files(path: Path | None) -> list[Path]:
    if path is None or not path.is_dir():
        return []
    return sorted(file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS)


def _label_root(image_root: Path) -> Path:
    if image_root.name.lower() == "images":
        return image_root.parent / "labels"
    return image_root.parent / "labels"


def _matching_label(image: Path, image_root: Path, label_root: Path) -> Path:
    return (label_root / image.relative_to(image_root)).with_suffix(".txt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dataset(config: DatasetConfig) -> dict[str, Any]:
    """Inventory all configured splits and validate every label line."""

    class_count = len(config.class_names)
    total_boxes = Counter()
    total_images_by_class: dict[int, set[str]] = defaultdict(set)
    split_summaries: dict[str, Any] = {}
    all_dimensions: list[tuple[int, int]] = []
    unreadable: list[str] = []
    invalid_labels: list[str] = []
    image_without_label: list[str] = []
    images_with_empty_labels: list[str] = []
    label_without_image: list[str] = []
    filename_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, list[dict[str, str]]] = defaultdict(list)
    label_filename_splits: dict[str, set[str]] = defaultdict(set)
    label_hash_splits: dict[str, list[dict[str, str]]] = defaultdict(list)
    extensions = Counter()

    for split, image_root in config.split_paths.items():
        if image_root is None:
            split_summaries[split] = {"configured": False, "images": 0, "labels": 0, "boxes": 0}
            continue
        images = _image_files(image_root)
        label_root = _label_root(image_root)
        labels = sorted(label_root.rglob("*.txt")) if label_root.is_dir() else []
        image_keys = {
            image.relative_to(image_root).with_suffix("").as_posix().lower(): image for image in images
        }
        label_keys = {
            label.relative_to(label_root).with_suffix("").as_posix().lower(): label for label in labels
        }
        split_box_count = 0
        parsed_labels: dict[str, list[tuple[int, tuple[float, ...]]]] = {}
        for key, label in label_keys.items():
            boxes, errors = parse_label_file(label, class_count)
            parsed_labels[key] = boxes
            invalid_labels.extend(errors)
            split_box_count += len(boxes)
            label_filename_splits[label.name.lower()].add(split)
            label_hash_splits[_sha256(label)].append({"split": split, "path": str(label)})
            for class_id, _ in boxes:
                total_boxes[class_id] += 1
        for image in images:
            relative = image.relative_to(image_root).as_posix()
            extensions[image.suffix.lower()] += 1
            filename_splits[image.name.lower()].add(split)
            key = image.relative_to(image_root).with_suffix("").as_posix().lower()
            matching_label = label_keys.get(key)
            image_record = {"split": split, "path": str(image)}
            if matching_label is not None:
                image_record["label_path"] = str(matching_label)
                image_record["label_sha256"] = _sha256(matching_label)
            hash_splits[_sha256(image)].append(image_record)
            loaded = cv2.imread(str(image))
            if loaded is None:
                unreadable.append(str(image))
            else:
                height, width = loaded.shape[:2]
                all_dimensions.append((width, height))
            if key not in parsed_labels:
                image_without_label.append(str(image))
                continue
            if not parsed_labels[key]:
                images_with_empty_labels.append(str(image))
            present: set[int] = set()
            for class_id, _ in parsed_labels[key]:
                present.add(class_id)
            for class_id in present:
                total_images_by_class[class_id].add(str(image))
        for key, label in label_keys.items():
            if key not in image_keys:
                label_without_image.append(str(label))
        split_summaries[split] = {
            "configured": True,
            "image_path": str(image_root),
            "label_path": str(label_root.resolve()),
            "paths_exist": image_root.is_dir() and label_root.is_dir(),
            "images": len(images),
            "labels": len(labels),
            "boxes": split_box_count,
        }

    total_images = sum(item["images"] for item in split_summaries.values())
    total_labels = sum(item["labels"] for item in split_summaries.values())
    cross_split_names = [
        {"filename": name, "splits": sorted(splits)}
        for name, splits in sorted(filename_splits.items()) if len(splits) > 1
    ]
    cross_split_hashes = [
        {"sha256": digest, "files": files}
        for digest, files in hash_splits.items()
        if len({item["split"] for item in files}) > 1
    ]
    cross_split_label_names = [
        {"filename": name, "splits": sorted(splits)}
        for name, splits in sorted(label_filename_splits.items()) if len(splits) > 1
    ]
    cross_split_label_hashes = [
        {"sha256": digest, "files": files}
        for digest, files in label_hash_splits.items()
        if len({item["split"] for item in files}) > 1
    ]
    resolution_counts = Counter(f"{width}x{height}" for width, height in all_dimensions)
    if all_dimensions:
        widths, heights = zip(*all_dimensions)
        resolution = {
            "readable_image_count": len(all_dimensions),
            "minimum_width": min(widths), "maximum_width": max(widths),
            "median_width": median(widths),
            "minimum_height": min(heights), "maximum_height": max(heights),
            "median_height": median(heights),
            "unique_resolution_count": len(resolution_counts),
            "resolutions": dict(sorted(resolution_counts.items())),
        }
    else:
        resolution = {"readable_image_count": 0, "resolutions": {}}
    box_total = sum(total_boxes.values())
    classes = []
    for class_id, name in enumerate(config.class_names):
        boxes = total_boxes[class_id]
        classes.append({
            "class_id": class_id,
            "class_name": name,
            "image_count": len(total_images_by_class[class_id]),
            "bounding_box_count": boxes,
            "percentage_of_boxes": (100.0 * boxes / box_total) if box_total else 0.0,
        })
    box_counts = [item["bounding_box_count"] for item in classes]
    maximum = max(box_counts) if box_counts else 0
    minimum = min(box_counts) if box_counts else 0
    imbalance_ratio = (maximum / minimum) if minimum else None
    distribution_review = {
        "majority_classes": [item["class_name"] for item in classes if item["bounding_box_count"] == maximum],
        "minority_classes": [item["class_name"] for item in classes if item["bounding_box_count"] == minimum],
        "maximum_to_minimum_box_ratio": imbalance_ratio,
        "possible_imbalance": bool(maximum > 0 and (minimum == 0 or maximum / minimum >= 1.5)),
        "heuristic": "possible imbalance when largest class has at least 1.5 times the boxes of the smallest class",
    }
    percentages = {
        split: (100.0 * values["images"] / total_images) if total_images else 0.0
        for split, values in split_summaries.items()
    }
    return {
        "status": "inspection_complete",
        "dataset_yaml": str(config.yaml_path),
        "dataset_root": str(config.dataset_root),
        "configured_paths": {
            key: str(value) if value is not None else None for key, value in config.split_paths.items()
        },
        "class_count": class_count,
        "class_names": list(config.class_names),
        "classes": classes,
        "class_distribution_review": distribution_review,
        "splits": split_summaries,
        "split_image_percentages": percentages,
        "total_images": total_images,
        "total_annotation_files": total_labels,
        "total_bounding_boxes": box_total,
        "images_without_labels": {"count": len(image_without_label), "files": image_without_label},
        "images_with_empty_label_files": {
            "count": len(images_with_empty_labels), "files": images_with_empty_labels,
            "interpretation": "valid YOLO negative/background rows unless source review indicates accidental omissions",
        },
        "labels_without_images": {"count": len(label_without_image), "files": label_without_image},
        "unreadable_images": {"count": len(unreadable), "files": unreadable},
        "invalid_labels": {"count": len(invalid_labels), "findings": invalid_labels},
        "file_extensions": dict(sorted(extensions.items())),
        "image_resolutions": resolution,
        "duplicates": {
            "duplicate_filenames_across_splits": cross_split_names,
            "exact_duplicate_images_across_splits": cross_split_hashes,
            "duplicate_label_filenames_across_splits": cross_split_label_names,
            "exact_duplicate_label_files_across_splits": cross_split_label_hashes,
        },
        "split_policy": "Preserve supplied splits; all frames from one future source video must remain in one split.",
    }


def save_summary(summary: dict[str, Any], path: str | Path = DEFAULT_SUMMARY) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_project_summary(
    data_yaml: str | Path = DEFAULT_DATA_YAML,
    *,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Combine the dataset inventory with environment and planned-run metadata."""

    try:
        summary = inspect_dataset(load_dataset_yaml(data_yaml))
    except FileNotFoundError as error:
        if not allow_missing:
            raise
        summary = {
            "status": "blocked_dataset_not_downloaded",
            "blocking_reason": str(error),
            "dataset_yaml": str(Path(data_yaml).resolve()),
            "dataset_root": str(Path(data_yaml).resolve().parent),
            "configured_paths": None,
            "class_count": None,
            "class_names": None,
            "classes": None,
            "class_distribution_review": None,
            "splits": {"train": None, "val": None, "test": None},
            "split_image_percentages": None,
            "total_images": None,
            "total_annotation_files": None,
            "total_bounding_boxes": None,
            "images_without_labels": None,
            "images_with_empty_label_files": None,
            "labels_without_images": None,
            "unreadable_images": None,
            "invalid_labels": None,
            "file_extensions": None,
            "image_resolutions": None,
            "duplicates": None,
            "split_policy": "Preserve supplied splits; all frames from one future source video must remain in one split.",
        }
    training = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    summary["environment"] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "planned_training_device": str(training["device"]),
    }
    summary["planned_training_configuration"] = training
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_yaml", nargs="?", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Write an explicitly blocked summary when the dataset has not been downloaded.",
    )
    args = parser.parse_args()
    try:
        summary = build_project_summary(args.data_yaml, allow_missing=args.allow_missing)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Dataset inspection blocked: {error}\n")
    save_summary(summary, args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
