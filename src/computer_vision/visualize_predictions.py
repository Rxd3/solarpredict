"""Render fixed validation examples with ground truth and model predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO

from src.computer_vision.inspect_vision_dataset import (
    DEFAULT_DATA_YAML,
    _image_files,
    _label_root,
    _matching_label,
    load_dataset_yaml,
    parse_label_file,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models/computer_vision/solar_panel_detector_best.pt"
DEFAULT_OUTPUT = ROOT / "outputs/computer_vision/figures/day25_validation_predictions.png"


def select_validation_examples(config: Any, maximum: int = 6) -> list[tuple[Path, list[Any]]]:
    """Deterministically select validation images that greedily cover all classes."""

    image_root = config.split_paths["val"]
    if image_root is None or not image_root.is_dir():
        raise ValueError("A readable validation image directory is required.")
    label_root = _label_root(image_root)
    candidates: list[tuple[Path, list[Any]]] = []
    for image in _image_files(image_root):
        label = _matching_label(image, image_root, label_root)
        boxes, errors = parse_label_file(label, len(config.class_names))
        if boxes and not errors:
            candidates.append((image, boxes))

    uncovered = set(range(len(config.class_names)))
    selected: list[tuple[Path, list[Any]]] = []
    while candidates and len(selected) < maximum:
        best = max(
            candidates,
            key=lambda item: (len({box[0] for box in item[1]} & uncovered), -len(selected)),
        )
        candidates.remove(best)
        if selected and not ({box[0] for box in best[1]} & uncovered):
            break
        selected.append(best)
        uncovered -= {box[0] for box in best[1]}
    for candidate in candidates:
        if len(selected) >= maximum:
            break
        selected.append(candidate)
    if not selected:
        raise ValueError("No valid annotated validation images were found.")
    return selected


def render_validation_predictions(
    weights: Path = DEFAULT_WEIGHTS,
    data_yaml: Path = DEFAULT_DATA_YAML,
    output: Path = DEFAULT_OUTPUT,
    *,
    maximum: int = 6,
    image_size: int = 512,
    device: str = "cpu",
    confidence: float = 0.25,
) -> dict[str, Any]:
    """Create one comparison figure; never read images from the test split."""

    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {weights}")
    config = load_dataset_yaml(data_yaml)
    selected = select_validation_examples(config, maximum)
    model = YOLO(str(weights), task="detect")
    predictions = model.predict(
        source=[str(item[0]) for item in selected], imgsz=image_size, device=device,
        conf=confidence, save=False, verbose=False,
    )

    columns = min(3, len(selected))
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False)
    colors = plt.cm.tab10.colors
    prediction_count = 0
    covered_ground_truth = set()
    for axis, (image_path, ground_truth), result in zip(axes.flat, selected, predictions):
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Validation image became unreadable: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        axis.imshow(rgb)
        for class_id, (x_center, y_center, box_width, box_height) in ground_truth:
            covered_ground_truth.add(class_id)
            left = (x_center - box_width / 2) * width
            top = (y_center - box_height / 2) * height
            axis.add_patch(Rectangle(
                (left, top), box_width * width, box_height * height,
                fill=False, edgecolor="white", linewidth=1.5, linestyle="--",
            ))
            axis.text(
                max(0, left), max(8, top), f"GT: {config.class_names[class_id]}",
                color="black", fontsize=7, verticalalignment="bottom",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 1, "edgecolor": "none"},
            )
        if result.boxes is not None:
            for coordinates, class_value, confidence_value in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            ):
                class_id = int(class_value)
                x1, y1, x2, y2 = (float(value) for value in coordinates)
                color = colors[class_id % len(colors)]
                axis.add_patch(Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor=color, linewidth=2.0,
                ))
                axis.text(
                    max(0, x1), min(height - 2, max(8, y1)),
                    f"PRED: {config.class_names[class_id]} {confidence_value:.2f}",
                    color="white", fontsize=7, verticalalignment="bottom",
                    bbox={"facecolor": color, "alpha": 0.9, "pad": 1, "edgecolor": "none"},
                )
                prediction_count += 1
        axis.set_title(f"validation: {image_path.name}", fontsize=8)
        axis.axis("off")
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    figure.suptitle("Validation predictions (solid) vs ground truth (white dashed)")
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return {
        "output": str(output.resolve()),
        "validation_image_count": len(selected),
        "prediction_count": prediction_count,
        "ground_truth_class_ids_covered": sorted(covered_ground_truth),
        "test_images_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    try:
        report = render_validation_predictions(
            args.weights, args.data, args.output, maximum=args.maximum,
            image_size=args.image_size, device=args.device, confidence=args.confidence,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Prediction visualization blocked: {error}\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
