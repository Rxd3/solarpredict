"""Inspect an image dataset without training or running a model."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Recursively inspect image counts, readability, and dimensions."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Root directory of the image dataset to inspect.",
    )
    return parser.parse_args()


def find_image_files(dataset_dir: Path) -> list[Path]:
    """Return supported image files below *dataset_dir* in stable order."""
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def directory_label(image_path: Path, dataset_dir: Path) -> str:
    """Return a portable relative label for an image's containing directory."""
    relative_parent = image_path.parent.relative_to(dataset_dir)
    return "." if str(relative_parent) == "." else relative_parent.as_posix()


def inspect_images(dataset_dir: Path) -> None:
    """Print image counts and validate that OpenCV can read every candidate file."""
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "OpenCV is not installed. Activate the project virtual environment "
            "and run: python -m pip install -r requirements.txt"
        ) from error

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Image dataset directory does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory but received: {dataset_dir}")

    image_files = find_image_files(dataset_dir)
    counts = Counter(directory_label(path, dataset_dir) for path in image_files)

    print(f"Dataset directory: {dataset_dir.resolve()}")
    print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    print(f"Total image files: {len(image_files)}")

    print("\nImages per directory/class:")
    if counts:
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count}")
    else:
        print("  (no supported image files found)")

    unreadable_files: list[Path] = []
    sample: tuple[Path, int, int, int] | None = None

    # Reading each file is necessary to detect corrupt or unsupported content.
    for image_path in image_files:
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        except (cv2.error, OSError):
            image = None

        if image is None or image.size == 0:
            unreadable_files.append(image_path)
            continue

        if sample is None:
            height, width = image.shape[:2]
            channels = 1 if image.ndim == 2 else image.shape[2]
            sample = (image_path, width, height, channels)

    readable_count = len(image_files) - len(unreadable_files)
    print(f"\nReadable image files: {readable_count}")
    print(f"Unreadable image files: {len(unreadable_files)}")

    if sample is not None:
        sample_path, width, height, channels = sample
        print("\nSample image:")
        print(f"  Path: {sample_path.relative_to(dataset_dir).as_posix()}")
        print(f"  Dimensions: {width} x {height} pixels (width x height)")
        print(f"  Channels: {channels}")
    elif image_files:
        print("\nSample image: unavailable because no candidate file was readable.")
    else:
        print("\nSample image: unavailable because the dataset contains no images.")

    if unreadable_files:
        print("\nUnreadable files:")
        for image_path in unreadable_files:
            print(f"  - {image_path.relative_to(dataset_dir).as_posix()}")


def main() -> int:
    """Run the command-line utility and return a process exit code."""
    args = parse_args()
    try:
        inspect_images(args.dataset_dir)
    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Error: unable to scan image dataset: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
