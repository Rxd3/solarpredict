"""Move reviewed empty-label image/label pairs to a recoverable quarantine.

The default mode validates and previews the operation. Files are moved only when
``--execute`` is supplied. No annotation content is created or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW = ROOT / "outputs/computer_vision/empty_label_review.csv"
DATASET_ROOT = ROOT / "data/vision/solar_panel_defects"
QUARANTINE_ROOT = ROOT / "data/vision/quarantine/empty_labels"
DEFAULT_MANIFEST = QUARANTINE_ROOT / "manifest.csv"
REASON = (
    "Visible solar panels with empty annotation; cannot safely treat as background "
    "because clean is an object class."
)
SPLIT_DIRECTORIES = {"train": "train", "val": "valid", "test": "test"}
MANIFEST_FIELDS = (
    "original_split",
    "original_image_path",
    "original_label_path",
    "quarantine_image_path",
    "quarantine_label_path",
    "image_sha256",
    "label_sha256",
    "reason",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _ensure_within(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(parent.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes the allowed directory: {resolved}") from error
    return resolved


def build_quarantine_plan(review_path: str | Path = DEFAULT_REVIEW) -> list[dict[str, Any]]:
    """Validate the manual-review CSV and return an exact, non-mutating move plan."""

    review = Path(review_path).resolve()
    if not review.is_file():
        raise FileNotFoundError(f"Empty-label review CSV not found: {review}")
    with review.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("The empty-label review CSV contains no records.")

    plan: list[dict[str, Any]] = []
    seen_images: set[Path] = set()
    for row_number, row in enumerate(rows, start=2):
        split = row.get("split", "")
        if split not in SPLIT_DIRECTORIES:
            raise ValueError(f"Review row {row_number} has unsupported split {split!r}.")
        if row.get("review_status") != "NEEDS_MANUAL_REVIEW":
            raise ValueError(f"Review row {row_number} was not marked NEEDS_MANUAL_REVIEW.")
        if str(row.get("empty_label", "")).lower() != "true":
            raise ValueError(f"Review row {row_number} was not recorded as an empty label.")

        image = _ensure_within(Path(row["image_path"]), DATASET_ROOT)
        split_directory = SPLIT_DIRECTORIES[split]
        expected_image_root = (DATASET_ROOT / split_directory / "images").resolve()
        _ensure_within(image, expected_image_root)
        label = expected_image_root.parent / "labels" / f"{image.stem}.txt"
        label = _ensure_within(label, DATASET_ROOT)
        if image in seen_images:
            raise ValueError(f"Review CSV contains a duplicate image row: {image}")
        seen_images.add(image)
        if not image.is_file():
            raise FileNotFoundError(f"Reviewed active image not found: {image}")
        if not label.is_file():
            raise FileNotFoundError(f"Matching active label not found: {label}")
        if label.read_text(encoding="utf-8-sig").strip():
            raise ValueError(f"Reviewed label is no longer empty: {label}")

        destination_root = QUARANTINE_ROOT / split_directory
        destination_image = destination_root / "images" / image.name
        destination_label = destination_root / "labels" / label.name
        if destination_image.exists() or destination_label.exists():
            raise FileExistsError(f"Quarantine destination already exists for: {image.name}")
        plan.append({
            "original_split": split,
            "original_image_path": _relative(image),
            "original_label_path": _relative(label),
            "quarantine_image_path": _relative(destination_image),
            "quarantine_label_path": _relative(destination_label),
            "image_sha256": _sha256(image),
            "label_sha256": _sha256(label),
            "reason": REASON,
        })
    return plan


def execute_quarantine(
    plan: list[dict[str, Any]], manifest_path: str | Path = DEFAULT_MANIFEST
) -> Path:
    """Move a prevalidated plan, rolling moved files back if any move fails."""

    if not plan:
        raise ValueError("Refusing to execute an empty quarantine plan.")
    manifest = _ensure_within(Path(manifest_path), QUARANTINE_ROOT)
    if manifest.exists():
        raise FileExistsError(f"Quarantine manifest already exists: {manifest}")

    moved: list[tuple[Path, Path]] = []
    try:
        for row in plan:
            for source_key, destination_key in (
                ("original_image_path", "quarantine_image_path"),
                ("original_label_path", "quarantine_label_path"),
            ):
                source = _ensure_within(ROOT / row[source_key], DATASET_ROOT)
                destination = _ensure_within(ROOT / row[destination_key], QUARANTINE_ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(source))
        raise

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(plan)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--execute", action="store_true", help="Move the validated pairs and write the manifest."
    )
    args = parser.parse_args()
    try:
        plan = build_quarantine_plan(args.review)
        split_counts = {
            split: sum(row["original_split"] == split for row in plan)
            for split in SPLIT_DIRECTORIES
        }
        result = {
            "mode": "execute" if args.execute else "dry_run",
            "pair_count": len(plan),
            "split_counts": split_counts,
            "reason": REASON,
        }
        if args.execute:
            result["manifest"] = str(execute_quarantine(plan, args.manifest))
        print(json.dumps(result, indent=2))
    except (FileNotFoundError, FileExistsError, OSError, UnicodeError, ValueError) as error:
        parser.exit(2, f"Empty-label quarantine blocked: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
