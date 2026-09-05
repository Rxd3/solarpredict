"""Tests for Day 24 dataset validation and non-training YOLO setup."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import src.computer_vision.train_detector as training
from src.computer_vision.evaluate_detector import evaluate_detector
from src.computer_vision.inspect_vision_dataset import (
    DEFAULT_DATA_YAML,
    DEFAULT_SUMMARY,
    inspect_dataset,
    load_dataset_yaml,
    validate_yolo_line,
)
from src.computer_vision.visualize_annotations import render_contact_sheet


ROOT = Path(__file__).resolve().parents[1]
FROZEN_HASHES = {
    "models/comparisons/no_time_standard_scaler.joblib": "b8b04db85a3bae4a0187b8ede66e859e811b7ee85b995976c9dc808b67b9aefb",
    "models/comparisons/no_time_isolation_forest.joblib": "5eb151856bb1d0776587140bb2eb201ba450f40293d540a90db44b3599ef963c",
    "config/frozen_anomaly_detector.yaml": "825cdc02e000352611de77bc19fce1e550f0dd8c7bd8a791e5cc2f1da79dd8b3",
    "config/frozen_anomaly_threshold.yaml": "019d251482160e64da89e33c7276c6aad864696bfa907a087a7d7c300b0139b7",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_image(path: Path, value: int, shape: tuple[int, int] = (24, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((shape[0], shape[1], 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    for split in ("train", "valid", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
    _write_image(tmp_path / "train/images/one.jpg", 20)
    _write_image(tmp_path / "train/images/unlabelled.png", 40, (20, 20))
    _write_image(tmp_path / "valid/images/two.jpg", 60, (40, 50))
    shutil.copyfile(tmp_path / "train/images/one.jpg", tmp_path / "test/images/one.jpg")
    (tmp_path / "train/labels/one.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n1 0.25 0.25 0.1 0.1\n", encoding="utf-8"
    )
    (tmp_path / "valid/labels/two.txt").write_text(
        "1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (tmp_path / "test/labels/one.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    (tmp_path / "test/labels/orphan.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
    )
    data = {
        "path": ".", "train": "train/images", "val": "valid/images",
        "test": "test/images", "nc": 2, "names": {0: "alpha", 1: "beta"},
    }
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


def test_data_yaml_loading_and_actual_class_mapping(tiny_dataset: Path) -> None:
    config = load_dataset_yaml(tiny_dataset)
    assert config.dataset_root == tiny_dataset.parent.resolve()
    assert config.class_names == ("alpha", "beta")
    assert config.split_paths["train"] == (tiny_dataset.parent / "train/images").resolve()
    assert config.split_paths["val"] == (tiny_dataset.parent / "valid/images").resolve()
    assert config.split_paths["test"] == (tiny_dataset.parent / "test/images").resolve()


def test_inventory_totals_matching_distribution_and_duplicates(tiny_dataset: Path) -> None:
    summary = inspect_dataset(load_dataset_yaml(tiny_dataset))
    assert summary["class_count"] == 2
    assert {key: summary["splits"][key]["images"] for key in ("train", "val", "test")} == {
        "train": 2, "val": 1, "test": 1,
    }
    assert summary["total_images"] == 4
    assert summary["total_annotation_files"] == 4
    assert summary["total_bounding_boxes"] == 5
    assert [item["bounding_box_count"] for item in summary["classes"]] == [3, 2]
    assert [item["image_count"] for item in summary["classes"]] == [2, 2]
    assert summary["class_distribution_review"]["majority_classes"] == ["alpha"]
    assert summary["class_distribution_review"]["minority_classes"] == ["beta"]
    assert summary["class_distribution_review"]["possible_imbalance"]
    assert summary["images_without_labels"]["count"] == 1
    assert summary["labels_without_images"]["count"] == 1
    assert summary["invalid_labels"]["count"] == 0
    assert summary["unreadable_images"]["count"] == 0
    assert len(summary["duplicates"]["duplicate_filenames_across_splits"]) == 1
    assert len(summary["duplicates"]["exact_duplicate_images_across_splits"]) == 1
    assert summary["image_resolutions"]["readable_image_count"] == 4


@pytest.mark.parametrize(
    "line, message",
    [
        ("0 0.5 0.5 0.2", "exactly 5"),
        ("x 0.5 0.5 0.2 0.2", "numeric"),
        ("2 0.5 0.5 0.2 0.2", "outside"),
        ("0 1.1 0.5 0.2 0.2", "center"),
        ("0 0.5 0.5 0 0.2", "width and height"),
        ("0 0.5 0.5 inf 0.2", "finite"),
    ],
)
def test_yolo_label_validation_rejects_bad_lines(line: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_yolo_line(line, class_count=2)


def test_yolo_label_validation_accepts_normalized_box() -> None:
    class_id, box = validate_yolo_line("1 0.25 0.75 0.1 0.2", class_count=2)
    assert class_id == 1
    assert box == (0.25, 0.75, 0.1, 0.2)


def test_ground_truth_contact_sheet_is_generated(tiny_dataset: Path, tmp_path: Path) -> None:
    output = render_contact_sheet(tiny_dataset, tmp_path / "samples.png")
    assert output.is_file()
    assert output.stat().st_size > 0


def test_training_dry_run_resolves_without_calling_train(
    tiny_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummyYOLO:
        train_called = False

        def __init__(self, checkpoint: str, task: str) -> None:
            assert checkpoint == "yolo11n.pt"
            assert task == "detect"

        def train(self, **_kwargs):
            DummyYOLO.train_called = True
            raise AssertionError("dry-run must never train")

    monkeypatch.setattr(training, "YOLO", DummyYOLO)
    config = {
        "schema_version": 1, "status": "prepared_not_trained", "task": "detect",
        "pretrained_model": "yolo11n.pt", "dataset_yaml": str(tiny_dataset),
        "image_size": 640, "epochs": 50, "batch_size": 8, "patience": 10,
        "device": "cpu", "workers": 0, "random_seed": 42, "deterministic": True,
        "output_project": str(tmp_path / "training"), "run_name": "dry_run", "exist_ok": False,
    }
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    weights_before = sorted(ROOT.rglob("best.pt")) + sorted(ROOT.rglob("last.pt"))
    report, model = training.validate_training_setup(config_path)
    weights_after = sorted(ROOT.rglob("best.pt")) + sorted(ROOT.rglob("last.pt"))
    assert model is not None
    assert report["checks"]["ready_for_training"]
    assert report["checks"]["training_started"] is False
    assert not DummyYOLO.train_called
    assert weights_before == weights_after


def test_evaluation_requires_real_trained_weights(tiny_dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Complete a future YOLO training"):
        evaluate_detector(tmp_path / "missing-best.pt", tiny_dataset)


def test_real_project_summary_matches_downloaded_dataset() -> None:
    summary = json.loads(DEFAULT_SUMMARY.read_text(encoding="utf-8"))
    assert summary["status"] == "inspection_complete"
    assert summary["class_names"] == [
        "bird-drop", "clean", "dusty", "electrical-damage",
        "physical-damage", "snow-covered",
    ]
    assert summary["class_count"] == 6
    assert summary["total_images"] == 821
    assert summary["total_annotation_files"] == 821
    assert summary["total_bounding_boxes"] == 5760
    assert sum(split["images"] for split in summary["splits"].values()) == 821
    assert sum(split["labels"] for split in summary["splits"].values()) == 821
    assert sum(split["boxes"] for split in summary["splits"].values()) == 5760
    assert sum(item["bounding_box_count"] for item in summary["classes"]) == 5760
    assert summary["invalid_labels"]["count"] == 0
    assert summary["images_without_labels"]["count"] == 0
    assert summary["labels_without_images"]["count"] == 0
    assert summary["unreadable_images"]["count"] == 0
    assert summary["images_with_empty_label_files"]["count"] == 25
    assert len(summary["duplicates"]["exact_duplicate_images_across_splits"]) == 1
    assert summary["environment"]["cuda_available"] is False


def test_real_data_yaml_paths_and_contact_sheet_exist() -> None:
    config = load_dataset_yaml(DEFAULT_DATA_YAML)
    assert config.class_names == (
        "bird-drop", "clean", "dusty", "electrical-damage",
        "physical-damage", "snow-covered",
    )
    assert all(config.split_paths[key].is_dir() for key in ("train", "val", "test"))
    figure = ROOT / "outputs/computer_vision/figures/day24_dataset_samples.png"
    assert figure.is_file()
    assert cv2.imread(str(figure)) is not None


def test_real_dry_run_succeeds_without_trained_weights() -> None:
    weights_before = sorted(ROOT.rglob("best.pt")) + sorted(ROOT.rglob("last.pt"))
    report, model = training.validate_training_setup(training.DEFAULT_CONFIG)
    weights_after = sorted(ROOT.rglob("best.pt")) + sorted(ROOT.rglob("last.pt"))
    assert model is not None
    assert report["checks"]["ready_for_training"]
    assert report["checks"]["test_path_exists"] is True
    assert report["checks"]["training_started"] is False
    assert report["errors"] == []
    assert weights_before == weights_after == []


def test_frozen_operational_artifacts_are_unchanged() -> None:
    assert {relative: _sha256(ROOT / relative) for relative in FROZEN_HASHES} == FROZEN_HASHES
