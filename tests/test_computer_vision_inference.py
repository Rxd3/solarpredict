"""Contracts for frozen computer-vision evaluation and image/video inference."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import src.computer_vision.evaluate_detector as evaluation
from src.computer_vision.inference import (
    CLASS_NAMES,
    draw_detections,
    load_detector,
    predict_frame,
    predict_image,
    summarize_conditions,
)
from src.computer_vision.process_video import DETECTION_LOG_COLUMNS, process_video


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/computer_vision/solar_panel_detector_best.pt"
CHECKPOINT_SHA256 = "b7ad8d7fca947527b30fb24f8c11e9135fbe62aa1e7f44e8a54100933141cd89"
OPERATIONAL_ARTIFACT_HASHES = {
    "models/frozen_anomaly_detector_metadata.json":
        "04c323a6fd8cc4fa05660940162b80e12f493a9e88a6e0a4c62dd84fc607b4a8",
    "models/frozen_anomaly_threshold_metadata.json":
        "eb5d7ea51827faaa1620b8d7ef5f42da8294dfd30a8a281c5ccfbc3f77c5eec9",
    "outputs/anomaly_module_final_status.json":
        "7280443d99fa18b6bc5162881c1b528b166957da8a99cfd77978b69df0b51fae",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validation_image() -> Path:
    return sorted((ROOT / "data/vision/solar_panel_defects/valid/images").glob("*"))[0]


def test_selected_checkpoint_hash_and_exact_classes() -> None:
    assert _sha256(CHECKPOINT) == CHECKPOINT_SHA256
    assert CLASS_NAMES == (
        "bird-drop", "clean", "dusty", "electrical-damage",
        "physical-damage", "snow-covered",
    )
    model = load_detector(CHECKPOINT)
    assert tuple(model.names[index] for index in sorted(model.names)) == CLASS_NAMES


def test_real_image_and_frame_inference_return_bounded_schema() -> None:
    model = load_detector(CHECKPOINT)
    image_path = _validation_image()
    source = cv2.imread(str(image_path))
    source_before = source.copy()
    image_result = predict_image(image_path, model=model)
    frame_result = predict_frame(source, model=model)
    assert np.array_equal(source, source_before)
    required = {
        "image_width", "image_height", "detection_count", "classes_detected",
        "highest_confidence", "detections", "condition_summary",
    }
    assert required <= image_result.keys() and required <= frame_result.keys()
    assert image_result["image_path"] == str(image_path.resolve())
    assert image_result["image_width"] == frame_result["image_width"] == source.shape[1]
    assert image_result["image_height"] == frame_result["image_height"] == source.shape[0]
    for detection in frame_result["detections"]:
        assert set(detection) == {
            "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"
        }
        assert 0 <= detection["class_id"] < len(CLASS_NAMES)
        assert detection["class_name"] == CLASS_NAMES[detection["class_id"]]
        assert 0.0 <= detection["confidence"] <= 1.0
        assert 0.0 <= detection["x1"] <= detection["x2"] <= source.shape[1]
        assert 0.0 <= detection["y1"] <= detection["y2"] <= source.shape[0]
    annotated = draw_detections(source, frame_result["detections"])
    assert annotated.shape == source.shape
    assert not np.shares_memory(annotated, source)


def _detection(name: str, confidence: float = 0.7) -> dict[str, object]:
    class_id = CLASS_NAMES.index(name)
    return {
        "class_id": class_id, "class_name": name, "confidence": confidence,
        "x1": 1.0, "y1": 2.0, "x2": 10.0, "y2": 12.0,
    }


def test_condition_summary_logic_is_cautious() -> None:
    empty = summarize_conditions([])
    assert empty["overall_status"] == "NO_DETECTION"
    assert "healthy" not in json.dumps(empty).lower()
    assert "fault-free" not in json.dumps(empty).lower()
    assert summarize_conditions([_detection("clean")])["overall_status"] == "CLEAN"
    mixed = summarize_conditions([_detection("clean"), _detection("dusty", 0.8)])
    assert mixed["overall_status"] == "ATTENTION"
    assert mixed["counts_by_class"]["dusty"] == 1
    assert mixed["highest_confidence_by_class"]["dusty"] == pytest.approx(0.8)


def _write_test_video(path: Path, frame_count: int = 5, fps: float = 5.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 24))
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    writer.release()


def _dummy_prediction(frame: np.ndarray, **_kwargs: object) -> dict[str, object]:
    detections = [_detection("clean", 0.75)]
    return {
        "image_width": frame.shape[1], "image_height": frame.shape[0],
        "detection_count": 1, "classes_detected": ["clean"],
        "highest_confidence": 0.75, "detections": detections,
        "condition_summary": summarize_conditions(detections),
    }


def test_video_output_stride_timestamps_and_log_schema(tmp_path: Path) -> None:
    source = tmp_path / "input.avi"
    output = tmp_path / "annotated.avi"
    detections = tmp_path / "detections.csv"
    frames = tmp_path / "frames.json"
    summary_path = tmp_path / "summary.json"
    _write_test_video(source)
    summary = process_video(
        source, output, detections, frames, summary_path,
        frame_stride=2, detector=object(), prediction_function=_dummy_prediction,
    )
    assert summary["total_frames"] == summary["output_frames_written"] == 5
    assert summary["analyzed_frames"] == 3
    assert summary["frame_stride"] == 2
    assert summary["total_detections"] == 3
    capture = cv2.VideoCapture(str(output))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    capture.release()
    frame_payload = json.loads(frames.read_text(encoding="utf-8"))
    assert len(frame_payload["frames"]) == 5
    assert [record["analyzed"] for record in frame_payload["frames"]] == [
        True, False, True, False, True
    ]
    assert [record["video_time_seconds"] for record in frame_payload["frames"]] == pytest.approx(
        [0.0, 0.2, 0.4, 0.6, 0.8]
    )
    with detections.open(newline="", encoding="utf-8") as source_csv:
        reader = csv.DictReader(source_csv)
        assert tuple(reader.fieldnames or ()) == DETECTION_LOG_COLUMNS
        assert len(list(reader)) == 3


def test_test_evaluation_is_frozen_once_and_contains_no_training_call() -> None:
    config = yaml.safe_load((
        ROOT / "config/computer_vision_test_evaluation.yaml"
    ).read_text(encoding="utf-8"))
    report = json.loads((
        ROOT / "outputs/computer_vision/test_evaluation.json"
    ).read_text(encoding="utf-8"))
    assert config["status"] == "frozen_before_evaluation"
    assert config["evaluation_policy"]["permitted_runs"] == 1
    assert config["evaluation_policy"]["tune_from_test_results"] is False
    assert config["confidence_threshold"] is None and config["iou_threshold"] is None
    assert report["evaluated_split"] == "test" and report["test_split_used"] is True
    assert report["prediction_summary"]["image_count"] == 98
    assert report["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert ".train(" not in inspect.getsource(evaluation.evaluate_detector)
    assert ".fit(" not in inspect.getsource(evaluation.evaluate_detector)


def test_operational_anomaly_artifacts_remain_unchanged() -> None:
    assert {
        relative: _sha256(ROOT / relative) for relative in OPERATIONAL_ARTIFACT_HASHES
    } == OPERATIONAL_ARTIFACT_HASHES
