"""Render representative stored test predictions without rerunning test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.computer_vision.inspect_vision_dataset import (
    DEFAULT_DATA_YAML,
    _image_files,
    _label_root,
    _matching_label,
    load_dataset_yaml,
    parse_label_file,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = ROOT / "outputs/computer_vision/test_prediction_summary.json"
DEFAULT_OUTPUT = ROOT / "outputs/computer_vision/figures/day26_test_predictions.png"


def _select_examples(
    config: Any, prediction_lookup: dict[str, dict[str, Any]], maximum: int
) -> list[tuple[Path, list[Any], dict[str, Any]]]:
    image_root = config.split_paths["test"]
    if image_root is None or not image_root.is_dir():
        raise ValueError("A readable test image directory is required.")
    label_root = _label_root(image_root)
    candidates: list[tuple[Path, list[Any], dict[str, Any]]] = []
    for image_path in _image_files(image_root):
        prediction = prediction_lookup.get(str(image_path.resolve()))
        if prediction is None:
            continue
        boxes, errors = parse_label_file(
            _matching_label(image_path, image_root, label_root), len(config.class_names)
        )
        if boxes and not errors:
            candidates.append((image_path, boxes, prediction))
    if not candidates:
        raise ValueError("Stored predictions do not match readable annotated test images.")

    selected: list[tuple[Path, list[Any], dict[str, Any]]] = []
    # Explicitly include a missed case when one exists, prioritizing difficult classes.
    missed = [candidate for candidate in candidates if not candidate[2]["detections"]]
    if missed:
        difficult = {2, 3, 4}
        missed.sort(
            key=lambda item: (
                -len({box[0] for box in item[1]} & difficult),
                item[0].name,
            )
        )
        selected.append(missed[0])
        candidates.remove(missed[0])

    covered = {box[0] for item in selected for box in item[1]}
    all_classes = set(range(len(config.class_names)))
    while candidates and len(selected) < maximum:
        uncovered = all_classes - covered
        best = max(
            candidates,
            key=lambda item: (
                len({box[0] for box in item[1]} & uncovered),
                -len(item[2]["detections"]),
                item[0].name,
            ),
        )
        candidates.remove(best)
        selected.append(best)
        covered.update(box[0] for box in best[1])
    return selected


def render_test_predictions(
    prediction_summary_path: str | Path = DEFAULT_PREDICTIONS,
    data_yaml: str | Path = DEFAULT_DATA_YAML,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    maximum: int = 6,
) -> dict[str, Any]:
    """Draw stored confidence-0.25 predictions and ground truth for context."""

    if maximum < 1:
        raise ValueError("maximum must be positive.")
    source = Path(prediction_summary_path)
    if not source.is_file():
        raise FileNotFoundError(f"Prediction summary not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    prediction_lookup = {
        str(Path(item["image_path"]).resolve()): item for item in payload["images"]
    }
    config = load_dataset_yaml(Path(data_yaml))
    selected = _select_examples(config, prediction_lookup, maximum)

    columns = min(3, len(selected))
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False)
    colors = plt.cm.tab10.colors
    covered_ground_truth: set[int] = set()
    missed_images = 0
    for axis, (image_path, ground_truth, prediction) in zip(axes.flat, selected):
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Test image became unreadable: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        axis.imshow(rgb)
        for class_id, (x_center, y_center, box_width, box_height) in ground_truth:
            covered_ground_truth.add(class_id)
            left = (x_center - box_width / 2) * width
            top = (y_center - box_height / 2) * height
            axis.add_patch(
                Rectangle(
                    (left, top),
                    box_width * width,
                    box_height * height,
                    fill=False,
                    edgecolor="white",
                    linewidth=1.5,
                    linestyle="--",
                )
            )
            axis.text(
                max(0, left),
                max(8, top),
                f"GT: {config.class_names[class_id]}",
                color="black",
                fontsize=7,
                verticalalignment="bottom",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 1, "edgecolor": "none"},
            )
        detections = prediction["detections"]
        if not detections:
            missed_images += 1
            axis.text(
                0.5,
                0.04,
                "NO PREDICTIONS AT VIEWING THRESHOLD",
                transform=axis.transAxes,
                ha="center",
                color="white",
                fontsize=8,
                bbox={"facecolor": "#8b1e1e", "alpha": 0.9, "pad": 2},
            )
        for detection in detections:
            class_id = int(detection["class_id"])
            x1, y1, x2, y2 = (float(value) for value in detection["xyxy"])
            color = colors[class_id % len(colors)]
            axis.add_patch(
                Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor=color, linewidth=2.0,
                )
            )
            axis.text(
                max(0, x1),
                min(height - 2, max(8, y1)),
                f"PRED: {config.class_names[class_id]} {float(detection['confidence']):.2f}",
                color="white",
                fontsize=7,
                verticalalignment="bottom",
                bbox={"facecolor": color, "alpha": 0.9, "pad": 1, "edgecolor": "none"},
            )
        axis.set_title(f"test: {image_path.name}", fontsize=8)
        axis.axis("off")
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    figure.suptitle("Test predictions (solid) vs ground truth (white dashed)")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return {
        "output": str(output.resolve()),
        "test_images_shown": len(selected),
        "stored_predictions_used": True,
        "additional_test_evaluation_run": False,
        "missed_images_shown": missed_images,
        "ground_truth_class_ids_covered": sorted(covered_ground_truth),
        "figure_title_contains_day_number": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum", type=int, default=6)
    args = parser.parse_args()
    try:
        report = render_test_predictions(args.predictions, args.data, args.output, maximum=args.maximum)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Test visualization blocked: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
