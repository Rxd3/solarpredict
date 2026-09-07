"""Dashboard adapters for frozen image and video inspection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.computer_vision.inference import CLASS_NAMES, load_detector
from src.computer_vision.process_video import DETECTION_LOG_COLUMNS
from src.dashboard.data_service import build_dashboard_data
from src.dashboard.vision_service import (
    FROZEN_CHECKPOINT_SHA256,
    VisionUploadError,
    _temporary_input_path,
    analyze_image_upload,
    decode_image_upload,
    image_detection_table,
    inspect_video_upload,
    process_video_upload,
    sanitized_upload_name,
    verify_frozen_checkpoint,
    video_condition_summary,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_HASHES = {
    "config/frozen_anomaly_detector.yaml":
        "825cdc02e000352611de77bc19fce1e550f0dd8c7bd8a791e5cc2f1da79dd8b3",
    "config/frozen_anomaly_threshold.yaml":
        "019d251482160e64da89e33c7276c6aad864696bfa907a087a7d7c300b0139b7",
    "models/comparisons/no_time_standard_scaler.joblib":
        "b8b04db85a3bae4a0187b8ede66e859e811b7ee85b995976c9dc808b67b9aefb",
    "models/comparisons/no_time_isolation_forest.joblib":
        "5eb151856bb1d0776587140bb2eb201ba450f40293d540a90db44b3599ef963c",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def detector():
    return load_detector()


@pytest.fixture(scope="module")
def real_image_path() -> Path:
    return sorted((ROOT / "data/vision/solar_panel_defects/test/images").glob("*"))[0]


def _write_video(path: Path, source_image: Path, frames: int = 4, fps: float = 4.0) -> None:
    image = cv2.imread(str(source_image))
    assert image is not None
    resized = cv2.resize(image, (96, 64))
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (96, 64)
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = resized.copy()
        frame[0, 0] = index
        writer.write(frame)
    writer.release()


def test_checkpoint_and_upload_name_safety(tmp_path: Path) -> None:
    assert verify_frozen_checkpoint() == FROZEN_CHECKPOINT_SHA256
    assert _sha256(ROOT / "models/computer_vision/solar_panel_detector_best.pt") == (
        FROZEN_CHECKPOINT_SHA256
    )
    assert sanitized_upload_name(r"..\..\unsafe panel.jpg", (".jpg",)) == "unsafe_panel.jpg"
    temporary = _temporary_input_path(tmp_path.resolve(), ".mp4")
    assert temporary.parent == tmp_path.resolve()
    assert temporary.name.startswith("input_") and temporary.suffix == ".mp4"


def test_image_decode_rejects_corrupt_and_unsupported_content(real_image_path: Path) -> None:
    decoded = decode_image_upload(real_image_path.read_bytes(), real_image_path.name)
    assert decoded["width"] > 0 and decoded["height"] > 0
    assert decoded["frame"].shape[:2] == (decoded["height"], decoded["width"])
    with pytest.raises(VisionUploadError, match="readable"):
        decode_image_upload(b"not an image", "corrupt.jpg")
    with pytest.raises(VisionUploadError, match="Unsupported"):
        decode_image_upload(real_image_path.read_bytes(), "image.gif")


def test_real_image_inference_schema_and_table(detector, real_image_path: Path) -> None:
    result = analyze_image_upload(
        real_image_path.read_bytes(), real_image_path.name, detector=detector
    )
    prediction = result["prediction"]
    assert result["upload_sha256"] == _sha256(real_image_path)
    assert result["annotated_png"].startswith(b"\x89PNG")
    assert prediction["condition_summary"]["overall_status"] in {
        "CLEAN", "ATTENTION", "NO_DETECTION"
    }
    table = image_detection_table(prediction)
    assert table.columns.tolist() == [
        "Detection", "Class", "Confidence", "x1", "y1", "x2", "y2"
    ]
    assert len(table) == prediction["detection_count"]
    if not table.empty:
        assert table["Confidence"].between(0, 1).all()


def test_video_condition_summary_has_distinct_cautious_states() -> None:
    empty = {"detections_by_class": {name: 0 for name in CLASS_NAMES}}
    no_detection = video_condition_summary(empty)
    assert no_detection["overall_status"] == "NO_DETECTION"
    assert "healthy" in no_detection["message"] and "certify" in no_detection["message"]
    clean = {"detections_by_class": {name: int(name == "clean") for name in CLASS_NAMES}}
    assert video_condition_summary(clean)["overall_status"] == "CLEAN"
    dusty = {"detections_by_class": {name: int(name == "dusty") for name in CLASS_NAMES}}
    assert video_condition_summary(dusty)["overall_status"] == "ATTENTION"


def test_video_upload_rejects_corrupt_content() -> None:
    with pytest.raises(VisionUploadError, match="OpenCV can open"):
        inspect_video_upload(b"not a video", "corrupt.mp4")


def test_real_video_dashboard_integration_and_stride(
    tmp_path: Path, detector, real_image_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    _write_video(source, real_image_path)
    result = process_video_upload(
        source.read_bytes(),
        r"..\..\uploaded drone.mp4",
        detector=detector,
        confidence=0.25,
        frame_stride=2,
    )
    summary = result["summary"]
    assert summary["total_frames"] == summary["output_frames_written"] == 4
    assert summary["analyzed_frames"] == 2
    assert summary["frame_stride"] == 2
    assert summary["temporary_files_cleaned_after_processing"] is True
    assert summary["browser_compatible_h264_created"] is True
    assert result["browser_playback_ready"] is True
    assert result["annotated_video_bytes"]
    assert tuple(result["detections"].columns) == DETECTION_LOG_COLUMNS
    assert all("uploaded_drone" not in name for name in result["download_names"].values())
    assert all("demo_" not in name for name in result["download_names"].values())
    public_summary = json.loads(result["summary_json_bytes"])
    assert "solarpredict_video_run_" not in json.dumps(public_summary)


def test_frozen_modules_and_operational_zero_alert_result_remain_unchanged() -> None:
    data = build_dashboard_data(verify_frozen_scores=True)
    assert len(data) == 337 and data["status"].eq("NORMAL").all()
    assert {relative: _sha256(ROOT / relative) for relative in FROZEN_HASHES} == FROZEN_HASHES
    sources = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "src/dashboard/app.py",
            "src/dashboard/vision_service.py",
            "src/dashboard/components.py",
        )
    )
    assert ".train(" not in sources and ".fit(" not in sources
