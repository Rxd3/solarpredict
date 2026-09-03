"""Smoke tests for the command-line dataset inspection utilities."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_SCRIPT = PROJECT_ROOT / "src" / "data_processing" / "inspect_dataset.py"
IMAGES_SCRIPT = PROJECT_ROOT / "src" / "data_processing" / "inspect_images.py"


def run_script(script: Path, *arguments: object) -> subprocess.CompletedProcess[str]:
    """Run a project script using the active test interpreter."""
    return subprocess.run(
        [sys.executable, str(script), *(str(argument) for argument in arguments)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_inspect_dataset_reports_expected_sections(tmp_path: Path) -> None:
    csv_path = tmp_path / "operational_sample.csv"
    csv_path.write_text(
        "timestamp,power_kw,status\n"
        "2026-01-01T12:00:00,8.5,normal\n"
        "2026-01-01T12:05:00,,missing_reading\n",
        encoding="utf-8",
    )

    result = run_script(DATASET_SCRIPT, csv_path, "--head", 1)

    assert result.returncode == 0, result.stderr
    assert "Shape: 2 rows x 3 columns" in result.stdout
    assert "Column names:" in result.stdout
    assert "Data types:" in result.stdout
    assert "Missing values:" in result.stdout
    assert "Descriptive statistics:" in result.stdout
    assert "power_kw" in result.stdout


def test_inspect_images_counts_classes_and_unreadable_files(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    dusty_dir = tmp_path / "dusty"
    clean_dir.mkdir()
    dusty_dir.mkdir()

    sample_image = np.zeros((12, 18, 3), dtype=np.uint8)
    assert cv2.imwrite(str(clean_dir / "panel.jpg"), sample_image)
    (dusty_dir / "broken.png").write_text("not an image", encoding="utf-8")

    result = run_script(IMAGES_SCRIPT, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Total image files: 2" in result.stdout
    assert "clean: 1" in result.stdout
    assert "dusty: 1" in result.stdout
    assert "Dimensions: 18 x 12 pixels" in result.stdout
    assert "Unreadable image files: 1" in result.stdout
    assert "dusty/broken.png" in result.stdout
