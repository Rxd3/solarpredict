"""Create audit visuals and records for the known Day 24 dataset issues."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from src.computer_vision.inspect_vision_dataset import (
    DEFAULT_DATA_YAML,
    load_dataset_yaml,
    parse_label_file,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "outputs/computer_vision/figures"
EMPTY_REVIEW_CSV = ROOT / "outputs/computer_vision/empty_label_review.csv"
DUPLICATE_REVIEW_FIGURE = FIGURE_DIR / "day24_duplicate_review.png"
TRAIN_STEM = "Physical-damaged-48-_jpg.rf.270e3124c478a269b63d42bb9037807f"
TEST_STEM = "Bird-184-_jpg.rf.4929f26fc5a5f175af527f49cefe25b3"
QUARANTINE_ROOT = ROOT / "data/vision/quarantine/cross_split_duplicate"


def _candidate(active: Path, quarantined: Path) -> Path:
    if active.is_file():
        return active
    if quarantined.is_file():
        return quarantined
    raise FileNotFoundError(f"Neither active nor quarantined review file exists: {active}")


def _draw_boxes(axis, rgb, boxes, class_names, title: str) -> None:
    height, width = rgb.shape[:2]
    axis.imshow(rgb)
    colors = plt.cm.tab10.colors
    for class_id, (x_center, y_center, box_width, box_height) in boxes:
        left = (x_center - box_width / 2) * width
        top = (y_center - box_height / 2) * height
        color = colors[class_id % len(colors)]
        axis.add_patch(Rectangle(
            (left, top), box_width * width, box_height * height,
            fill=False, edgecolor=color, linewidth=2,
        ))
        axis.text(
            max(0, left), max(8, top), class_names[class_id], color="white", fontsize=8,
            verticalalignment="bottom",
            bbox={"facecolor": color, "alpha": 0.85, "pad": 2, "edgecolor": "none"},
        )
    axis.set_title(title, fontsize=10)
    axis.axis("off")


def render_duplicate_review(output: Path = DUPLICATE_REVIEW_FIGURE) -> Path:
    """Render the identical train/test pixels with their differing labels."""

    config = load_dataset_yaml(DEFAULT_DATA_YAML)
    train_image = ROOT / f"data/vision/solar_panel_defects/train/images/{TRAIN_STEM}.jpg"
    train_label = ROOT / f"data/vision/solar_panel_defects/train/labels/{TRAIN_STEM}.txt"
    test_image = _candidate(
        ROOT / f"data/vision/solar_panel_defects/test/images/{TEST_STEM}.jpg",
        QUARANTINE_ROOT / f"images/{TEST_STEM}.jpg",
    )
    test_label = _candidate(
        ROOT / f"data/vision/solar_panel_defects/test/labels/{TEST_STEM}.txt",
        QUARANTINE_ROOT / f"labels/{TEST_STEM}.txt",
    )
    train_boxes, train_errors = parse_label_file(train_label, len(config.class_names))
    test_boxes, test_errors = parse_label_file(test_label, len(config.class_names))
    if train_errors or test_errors:
        raise ValueError(f"Cannot render invalid duplicate labels: {train_errors + test_errors}")
    train_bgr = cv2.imread(str(train_image))
    test_bgr = cv2.imread(str(test_image))
    if train_bgr is None or test_bgr is None:
        raise ValueError("Duplicate review image is unreadable.")
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    _draw_boxes(
        axes[0], cv2.cvtColor(train_bgr, cv2.COLOR_BGR2RGB), train_boxes,
        config.class_names, f"TRAIN — retained ({len(train_boxes)} boxes)",
    )
    _draw_boxes(
        axes[1], cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB), test_boxes,
        config.class_names, f"TEST — quarantined ({len(test_boxes)} boxes)",
    )
    figure.suptitle("Day 24 cross-split duplicate review — identical image, differing ground truth")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def find_empty_label_images() -> pd.DataFrame:
    """Inventory active images whose matching label file has zero bytes/content."""

    config = load_dataset_yaml(DEFAULT_DATA_YAML)
    rows = []
    for split, image_root in config.split_paths.items():
        if image_root is None:
            continue
        label_root = image_root.parent / "labels"
        for label in sorted(label_root.rglob("*.txt")):
            if label.read_text(encoding="utf-8-sig").strip():
                continue
            relative = label.relative_to(label_root).with_suffix("")
            images = [
                path for path in image_root.glob(f"{relative.as_posix()}.*")
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
            ]
            if len(images) != 1:
                raise ValueError(f"Expected one image for empty label {label}; found {len(images)}.")
            image_path = images[0]
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Empty-label review image is unreadable: {image_path}")
            height, width = image.shape[:2]
            rows.append({
                "split": split,
                "image_filename": image_path.name,
                "image_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "empty_label": True,
                "review_status": "NEEDS_MANUAL_REVIEW",
            })
    return pd.DataFrame(rows)


def render_empty_label_sheets(
    review: pd.DataFrame,
    output_dir: Path = FIGURE_DIR,
    *,
    tiles_per_sheet: int = 9,
) -> list[Path]:
    """Display every empty-label image without inventing boxes."""

    outputs = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for sheet_index, start in enumerate(range(0, len(review), tiles_per_sheet), start=1):
        group = review.iloc[start:start + tiles_per_sheet]
        columns = 3
        rows = (len(group) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(15, 4.7 * rows), squeeze=False)
        for axis, (_, record) in zip(axes.flat, group.iterrows()):
            image = cv2.imread(str(record.image_path))
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axis.set_title(f"{record.split}: {record.image_filename}", fontsize=7)
            axis.axis("off")
        for axis in axes.flat[len(group):]:
            axis.axis("off")
        figure.suptitle(
            f"Day 24 empty-label review {sheet_index} — no boxes added",
            fontsize=14,
        )
        figure.tight_layout()
        destination = output_dir / f"day24_empty_labels_review_{sheet_index}.png"
        figure.savefig(destination, dpi=140, bbox_inches="tight")
        plt.close(figure)
        outputs.append(destination)
    return outputs


def create_review_outputs() -> dict[str, object]:
    duplicate = render_duplicate_review()
    empty = find_empty_label_images()
    EMPTY_REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    empty.to_csv(EMPTY_REVIEW_CSV, index=False, lineterminator="\n")
    sheets = render_empty_label_sheets(empty)
    return {
        "duplicate_review": str(duplicate),
        "empty_label_csv": str(EMPTY_REVIEW_CSV),
        "empty_label_count": len(empty),
        "likely_background": int(empty.review_status.eq("LIKELY_BACKGROUND").sum()),
        "needs_manual_review": int(empty.review_status.eq("NEEDS_MANUAL_REVIEW").sum()),
        "empty_label_sheets": [str(path) for path in sheets],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    outputs = create_review_outputs()
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
