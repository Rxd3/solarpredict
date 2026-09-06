"""Create a clearly labelled engineering demo video from validation images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.computer_vision.inspect_vision_dataset import DEFAULT_DATA_YAML, load_dataset_yaml
from src.computer_vision.visualize_predictions import select_validation_examples


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/computer_vision/video/demo_input.mp4"
DEFAULT_METADATA = ROOT / "outputs/computer_vision/video/demo_input_metadata.json"
DEMO_DESCRIPTION = "Demo sequence created from validation images for video-pipeline testing."


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y:y + resized_height, x:x + resized_width] = resized
    return canvas


def create_demo_video(
    output_path: str | Path = DEFAULT_OUTPUT,
    metadata_path: str | Path = DEFAULT_METADATA,
    *,
    data_yaml: str | Path = DEFAULT_DATA_YAML,
    image_count: int = 6,
    repeats_per_image: int = 2,
    fps: float = 2.0,
    width: int = 640,
    height: int = 480,
) -> dict[str, Any]:
    """Create a deterministic non-drone demo input without altering source images."""

    if image_count < 1 or repeats_per_image < 1 or fps <= 0 or width < 1 or height < 1:
        raise ValueError("Demo video dimensions, image count, repeats, and FPS must be positive.")
    dataset = load_dataset_yaml(Path(data_yaml))
    validation_root = dataset.split_paths["val"]
    if validation_root is None:
        raise ValueError("Validation split is unavailable.")
    images = [
        image_path
        for image_path, _ in select_validation_examples(dataset, maximum=image_count)
    ]
    if not images:
        raise ValueError("No validation images are available for the demo sequence.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise OSError(f"OpenCV could not create demo video: {destination}")
    try:
        for image_path in images:
            source = cv2.imread(str(image_path))
            if source is None:
                raise ValueError(f"Validation image is unreadable: {image_path}")
            frame = _letterbox(source, width, height)
            cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
            cv2.putText(
                frame,
                "DEMO - VALIDATION IMAGE SEQUENCE",
                (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )
            for _ in range(repeats_per_image):
                writer.write(frame)
    finally:
        writer.release()

    metadata = {
        "schema_version": 1,
        "description": DEMO_DESCRIPTION,
        "is_real_drone_footage": False,
        "source_split": "validation",
        "source_images": [str(path.resolve()) for path in images],
        "image_count": len(images),
        "repeats_per_image": repeats_per_image,
        "total_frames": len(images) * repeats_per_image,
        "fps": fps,
        "width": width,
        "height": height,
        "output_video": str(destination.resolve()),
    }
    metadata_destination = Path(metadata_path)
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_destination.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--image-count", type=int, default=6)
    parser.add_argument("--repeats-per-image", type=int, default=2)
    args = parser.parse_args()
    try:
        report = create_demo_video(
            args.output,
            args.metadata,
            image_count=args.image_count,
            repeats_per_image=args.repeats_per_image,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"Demo creation blocked: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
