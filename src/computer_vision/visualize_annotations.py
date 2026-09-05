"""Render representative ground-truth YOLO annotations as one contact sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

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
DEFAULT_OUTPUT = ROOT / "outputs/computer_vision/figures/day24_dataset_samples.png"


def select_representative_images(config, maximum: int = 9):
    """Greedily select readable images covering as many actual classes as possible."""

    candidates = []
    for split, image_root in config.split_paths.items():
        if image_root is None:
            continue
        label_root = _label_root(image_root)
        for image in _image_files(image_root):
            label = _matching_label(image, image_root, label_root)
            if not label.is_file():
                continue
            boxes, errors = parse_label_file(label, len(config.class_names))
            if boxes and not errors:
                candidates.append((image, split, boxes))
    uncovered = set(range(len(config.class_names)))
    selected = []
    while candidates and len(selected) < maximum:
        best = max(candidates, key=lambda item: len({box[0] for box in item[2]} & uncovered))
        candidates.remove(best)
        if selected and not ({box[0] for box in best[2]} & uncovered):
            break
        if cv2.imread(str(best[0])) is None:
            continue
        selected.append(best)
        uncovered -= {box[0] for box in best[2]}
    if len(selected) < maximum:
        for candidate in candidates:
            if len(selected) >= maximum:
                break
            if cv2.imread(str(candidate[0])) is not None:
                selected.append(candidate)
    return selected


def render_contact_sheet(data_yaml: Path, output: Path, maximum: int = 9) -> Path:
    config = load_dataset_yaml(data_yaml)
    selected = select_representative_images(config, maximum)
    if not selected:
        raise ValueError("No readable images with valid annotations are available to visualize.")
    columns = min(3, len(selected))
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    colors = plt.cm.tab10.colors
    for axis, (image_path, split, boxes) in zip(axes.flat, selected):
        bgr = cv2.imread(str(image_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        axis.imshow(rgb)
        for class_id, (x_center, y_center, box_width, box_height) in boxes:
            left = (x_center - box_width / 2) * width
            top = (y_center - box_height / 2) * height
            color = colors[class_id % len(colors)]
            axis.add_patch(Rectangle(
                (left, top), box_width * width, box_height * height,
                fill=False, edgecolor=color, linewidth=2,
            ))
            axis.text(
                max(0, left), max(8, top), config.class_names[class_id],
                color="white", fontsize=8, verticalalignment="bottom",
                bbox={"facecolor": color, "alpha": 0.85, "pad": 2, "edgecolor": "none"},
            )
        axis.set_title(f"{split}: {image_path.name}", fontsize=9)
        axis.axis("off")
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    figure.suptitle("Day 24 dataset samples — ground-truth YOLO annotations", fontsize=14)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_yaml", nargs="?", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum", type=int, default=9)
    args = parser.parse_args()
    try:
        result = render_contact_sheet(args.data_yaml, args.output, args.maximum)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Visualization blocked: {error}\n")
    print(f"Saved ground-truth contact sheet: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
