"""Tests for the Day 24 dataset gate and Day 25 YOLO prototype."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from ultralytics import YOLO

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
from src.computer_vision.visualize_predictions import select_validation_examples


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
        "1 0.5 0.5 0.2 0.2\n0 0.3 0.3 0.1 0.1\n", encoding="utf-8"
    )
    (tmp_path / "test/labels/one.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n1 0.4 0.4 0.1 0.1\n", encoding="utf-8"
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
    assert summary["total_bounding_boxes"] == 7
    assert [item["bounding_box_count"] for item in summary["classes"]] == [4, 3]
    assert [item["image_count"] for item in summary["classes"]] == [3, 3]
    assert summary["class_distribution_review"]["majority_classes"] == ["alpha"]
    assert summary["class_distribution_review"]["minority_classes"] == ["beta"]
    assert not summary["class_distribution_review"]["possible_imbalance"]
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
    (tiny_dataset.parent / "train/images/unlabelled.png").unlink()
    (tiny_dataset.parent / "test/labels/orphan.txt").unlink()
    _write_image(tiny_dataset.parent / "test/images/one.jpg", 80)
    (tiny_dataset.parent / "test/images/one.jpg").rename(
        tiny_dataset.parent / "test/images/test_one.jpg"
    )
    (tiny_dataset.parent / "test/labels/one.txt").rename(
        tiny_dataset.parent / "test/labels/test_one.txt"
    )
    config = {
        "schema_version": 1, "status": "prepared_not_trained", "task": "detect",
        "pretrained_model": "yolo11n.pt", "dataset_yaml": str(tiny_dataset),
        "image_size": 640, "epochs": 50, "batch_size": 8, "patience": 10,
        "device": "cpu", "workers": 0, "random_seed": 42, "deterministic": True,
        "save": True, "save_period": 5,
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
    assert summary["status"] == "inspection_complete_after_quarantine"
    assert summary["class_names"] == [
        "bird-drop", "clean", "dusty", "electrical-damage",
        "physical-damage", "snow-covered",
    ]
    assert summary["class_count"] == 6
    assert {key: summary["splits"][key]["images"] for key in ("train", "val", "test")} == {
        "train": 582, "val": 115, "test": 98,
    }
    assert summary["total_images"] == 795
    assert summary["total_annotation_files"] == 795
    assert summary["total_bounding_boxes"] == 5751
    assert sum(split["images"] for split in summary["splits"].values()) == 795
    assert sum(split["labels"] for split in summary["splits"].values()) == 795
    assert sum(split["boxes"] for split in summary["splits"].values()) == 5751
    assert sum(item["bounding_box_count"] for item in summary["classes"]) == 5751
    assert [item["bounding_box_count"] for item in summary["classes"]] == [
        1509, 1107, 1460, 207, 169, 1299,
    ]
    assert summary["invalid_labels"]["count"] == 0
    assert summary["images_without_labels"]["count"] == 0
    assert summary["labels_without_images"]["count"] == 0
    assert summary["unreadable_images"]["count"] == 0
    assert summary["images_with_empty_label_files"]["count"] == 0
    assert len(summary["duplicates"]["exact_duplicate_images_across_splits"]) == 0
    assert len(summary["duplicates"]["duplicate_filenames_across_splits"]) == 0
    assert summary["dataset_cleanup"]["cross_split_duplicate"]["removed_box_count"] == 9
    assert summary["dataset_cleanup"]["empty_label_quarantine"]["pair_count"] == 25
    assert summary["dataset_cleanup"]["empty_label_quarantine"][
        "all_manifest_paths_and_hashes_verified"
    ]
    assert summary["empty_label_review"]["likely_background"] == 0
    assert summary["empty_label_review"]["needs_manual_review"] == 25
    assert summary["empty_label_review"]["active_empty_labels_remaining"] == 0
    assert summary["all_classes_present_in_every_split"]
    assert all(
        value["all_classes_present"]
        for value in summary["class_coverage_by_split"].values()
    )
    assert not summary["training_quality_gate"]["blocked"]
    assert not any(summary["training_quality_gate"]["checks"].values())
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


def test_quarantine_and_review_artifacts_are_complete() -> None:
    stem = "Bird-184-_jpg.rf.4929f26fc5a5f175af527f49cefe25b3"
    active_image = ROOT / f"data/vision/solar_panel_defects/test/images/{stem}.jpg"
    active_label = ROOT / f"data/vision/solar_panel_defects/test/labels/{stem}.txt"
    quarantine = ROOT / "data/vision/quarantine/cross_split_duplicate"
    image = quarantine / f"images/{stem}.jpg"
    label = quarantine / f"labels/{stem}.txt"
    assert not active_image.exists() and not active_label.exists()
    assert _sha256(image) == "c6c7d608c8e1ed598fa95fe95a54df7ed5eeaeb5f73fc6bd823d9cab4a2361bc"
    assert _sha256(label) == "5ce3c28c8f1fb24f08e62661a6290add1f144b6643c9f5debeb0f86196c8900f"
    assert (quarantine / "README.md").is_file()
    review = pd.read_csv(ROOT / "outputs/computer_vision/empty_label_review.csv")
    assert len(review) == 25
    assert review.empty_label.eq(True).all()
    assert review.review_status.eq("NEEDS_MANUAL_REVIEW").all()
    assert len(list((ROOT / "outputs/computer_vision/figures").glob(
        "day24_empty_labels_review_*.png"
    ))) == 3
    assert (ROOT / "outputs/computer_vision/figures/day24_duplicate_review.png").is_file()

    empty_quarantine = ROOT / "data/vision/quarantine/empty_labels"
    manifest = pd.read_csv(empty_quarantine / "manifest.csv")
    assert len(manifest) == 25
    assert manifest.original_split.value_counts().to_dict() == {
        "train": 18, "val": 5, "test": 2,
    }
    assert manifest.reason.eq(
        "Visible solar panels with empty annotation; cannot safely treat as background "
        "because clean is an object class."
    ).all()
    for row in manifest.itertuples(index=False):
        original_image = ROOT / row.original_image_path
        original_label = ROOT / row.original_label_path
        quarantined_image = ROOT / row.quarantine_image_path
        quarantined_label = ROOT / row.quarantine_label_path
        assert not original_image.exists() and not original_label.exists()
        assert quarantined_image.is_file() and quarantined_label.is_file()
        assert _sha256(quarantined_image) == row.image_sha256
        assert _sha256(quarantined_label) == row.label_sha256
        assert quarantined_label.read_text(encoding="utf-8-sig").strip() == ""
    assert (empty_quarantine / "README.md").is_file()


def test_real_dry_run_succeeds_without_changing_trained_weights() -> None:
    checkpoint_paths = sorted(ROOT.rglob("best.pt")) + sorted(ROOT.rglob("last.pt"))
    weights_before = {str(path): _sha256(path) for path in checkpoint_paths}
    report, model = training.validate_training_setup(training.DEFAULT_CONFIG)
    checkpoint_paths_after = sorted(ROOT.rglob("best.pt")) + sorted(ROOT.rglob("last.pt"))
    weights_after = {str(path): _sha256(path) for path in checkpoint_paths_after}
    assert model is not None
    assert report["checks"]["ready_for_training"]
    assert report["checks"]["test_path_exists"] is True
    assert report["checks"]["split_image_counts"] == {"train": 582, "val": 115, "test": 98}
    assert report["checks"]["total_active_images"] == 795
    assert report["checks"]["class_count"] == 6
    assert report["checks"]["invalid_label_count"] == 0
    assert report["checks"]["empty_label_count"] == 0
    assert report["checks"]["image_without_label_count"] == 0
    assert report["checks"]["label_without_image_count"] == 0
    assert report["checks"]["unreadable_image_count"] == 0
    assert report["checks"]["cross_split_duplicate_image_groups"] == 0
    assert report["checks"]["cross_split_duplicate_filename_count"] == 0
    assert report["checks"]["all_classes_present_in_every_split"]
    assert report["checks"]["checkpoint_saving_enabled"]
    assert report["checks"]["checkpoint_save_period_epochs"] == 5
    assert report["checks"]["training_started"] is False
    assert report["errors"] == []
    assert weights_before == weights_after


def test_final_day25_configuration_and_checkpoint_artifacts() -> None:
    config = training.load_training_config(training.DEFAULT_CONFIG)
    assert {
        "pretrained_model": config["pretrained_model"],
        "epochs": config["epochs"],
        "image_size": config["image_size"],
        "batch_size": config["batch_size"],
        "patience": config["patience"],
        "device": config["device"],
        "workers": config["workers"],
        "random_seed": config["random_seed"],
    } == {
        "pretrained_model": "yolo11n.pt", "epochs": 30, "image_size": 512,
        "batch_size": 8, "patience": 8, "device": "cpu", "workers": 0,
        "random_seed": 42,
    }
    assert config["save"] is True and config["save_period"] == 5
    run = ROOT / config["output_project"] / config["run_name"]
    best = run / "weights/best.pt"
    last = run / "weights/last.pt"
    selected = ROOT / "models/computer_vision/solar_panel_detector_best.pt"
    assert best.is_file() and last.is_file() and selected.is_file()
    assert _sha256(best) == _sha256(selected)


def test_training_metadata_uses_validation_not_test() -> None:
    metadata = json.loads((
        ROOT / "models/computer_vision/solar_panel_detector_metadata.json"
    ).read_text(encoding="utf-8"))
    summary = json.loads((
        ROOT / "outputs/computer_vision/training_summary.json"
    ).read_text(encoding="utf-8"))
    validation = json.loads((
        ROOT / "outputs/computer_vision/validation_metrics.json"
    ).read_text(encoding="utf-8"))
    config = training.load_training_config(training.DEFAULT_CONFIG)
    assert metadata["training_configuration"] == config
    assert metadata["class_names"] == [
        "bird-drop", "clean", "dusty", "electrical-damage",
        "physical-damage", "snow-covered",
    ]
    assert metadata["model_selection_split"] == "val"
    assert metadata["test_split_used_for_training_or_model_selection"] is False
    assert summary["test_split_used_for_training_or_model_selection"] is False
    assert validation["evaluated_split"] == "val"
    assert validation["test_split_used"] is False
    assert metadata["epochs_completed"] <= metadata["epochs_requested"] == 30
    assert 1 <= metadata["best_epoch"] <= metadata["epochs_completed"]
    assert metadata["wall_clock_training_duration_seconds"] >= metadata[
        "training_duration_seconds"
    ]
    for checkpoint in metadata["checkpoints"].values():
        path = Path(checkpoint["path"])
        assert path.is_file()
        assert _sha256(path) == checkpoint["sha256"]


def test_selected_checkpoint_loads_and_validation_inference_is_bounded() -> None:
    selected = ROOT / "models/computer_vision/solar_panel_detector_best.pt"
    model = YOLO(str(selected), task="detect")
    config = load_dataset_yaml(DEFAULT_DATA_YAML)
    examples = select_validation_examples(config, maximum=3)
    results = model.predict(
        source=[str(item[0]) for item in examples], imgsz=512, device="cpu",
        conf=0.001, verbose=False,
    )
    assert len(results) == len(examples)
    detection_count = 0
    validation_root = config.split_paths["val"].resolve()
    for result in results:
        assert Path(result.path).resolve().is_relative_to(validation_root)
        height, width = result.orig_shape
        if result.boxes is None:
            continue
        for coordinates, class_value, confidence in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
        ):
            x1, y1, x2, y2 = coordinates
            assert 0 <= int(class_value) < 6
            assert 0.0 <= confidence <= 1.0
            assert 0.0 <= x1 <= x2 <= width + 1e-3
            assert 0.0 <= y1 <= y2 <= height + 1e-3
            detection_count += 1
    assert detection_count > 0
    figure = ROOT / "outputs/computer_vision/figures/day25_validation_predictions.png"
    assert figure.is_file() and cv2.imread(str(figure)) is not None


def test_frozen_operational_artifacts_are_unchanged() -> None:
    assert {relative: _sha256(ROOT / relative) for relative in FROZEN_HASHES} == FROZEN_HASHES
