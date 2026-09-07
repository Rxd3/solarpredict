"""Safe dashboard adapters for the existing frozen image/video inference APIs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

from src.computer_vision.inference import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIDENCE,
    draw_detections,
    predict_frame,
)
from src.computer_vision.process_video import DETECTION_LOG_COLUMNS, process_video


FROZEN_CHECKPOINT_SHA256 = (
    "b7ad8d7fca947527b30fb24f8c11e9135fbe62aa1e7f44e8a54100933141cd89"
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov")
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024


class VisionUploadError(ValueError):
    """Raised when uploaded media cannot satisfy the inspection contract."""


def sha256_bytes(content: bytes) -> str:
    """Return a stable upload identity for session-result invalidation."""

    return hashlib.sha256(content).hexdigest()


def verify_frozen_checkpoint(path: str | Path = DEFAULT_CHECKPOINT) -> str:
    """Reject missing or changed weights before dashboard inference."""

    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen computer-vision checkpoint not found: {checkpoint}")
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual != FROZEN_CHECKPOINT_SHA256:
        raise VisionUploadError(
            "Frozen computer-vision checkpoint hash does not match the selected model."
        )
    return actual


def sanitized_upload_name(filename: str, allowed_suffixes: tuple[str, ...]) -> str:
    """Reduce an untrusted client filename to a display-only safe basename."""

    basename = Path(str(filename)).name
    suffix = Path(basename).suffix.lower()
    if suffix not in allowed_suffixes:
        choices = ", ".join(value.removeprefix(".").upper() for value in allowed_suffixes)
        raise VisionUploadError(f"Unsupported file extension. Accepted formats: {choices}.")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(basename).stem).strip("._-")
    return f"{stem or 'upload'}{suffix}"


def _validate_size(content: bytes, maximum: int, media_type: str) -> None:
    if not content:
        raise VisionUploadError(f"Uploaded {media_type} is empty.")
    if len(content) > maximum:
        raise VisionUploadError(
            f"Uploaded {media_type} exceeds the {maximum // (1024 * 1024)} MB prototype limit."
        )


def decode_image_upload(content: bytes, filename: str) -> dict[str, Any]:
    """Decode and validate actual image bytes instead of trusting the extension."""

    safe_name = sanitized_upload_name(filename, IMAGE_SUFFIXES)
    _validate_size(content, MAX_IMAGE_BYTES, "image")
    encoded = np.frombuffer(content, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise VisionUploadError("Uploaded content is not a readable JPG/JPEG/PNG image.")
    height, width = frame.shape[:2]
    return {
        "filename": safe_name,
        "sha256": sha256_bytes(content),
        "width": int(width),
        "height": int(height),
        "frame": frame,
    }


def analyze_image_upload(
    content: bytes,
    filename: str,
    *,
    detector: Any,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Run real frozen inference and return memory-resident display/download data."""

    upload = decode_image_upload(content, filename)
    prediction = predict_frame(
        upload["frame"], model=detector, confidence=float(confidence), device="cpu"
    )
    annotated = draw_detections(upload["frame"], prediction["detections"])
    success, encoded = cv2.imencode(".png", annotated)
    if not success:
        raise OSError("OpenCV could not encode the annotated image result.")
    return {
        "filename": upload["filename"],
        "upload_sha256": upload["sha256"],
        "confidence_threshold": float(confidence),
        "image_width": upload["width"],
        "image_height": upload["height"],
        "prediction": prediction,
        "annotated_png": encoded.tobytes(),
        "annotated_filename": f"annotated_{Path(upload['filename']).stem}.png",
    }


def image_detection_table(prediction: dict[str, Any]) -> pd.DataFrame:
    """Return an ordered, presentation-safe table of actual detections."""

    columns = ["Detection", "Class", "Confidence", "x1", "y1", "x2", "y2"]
    rows = []
    ordered = sorted(
        prediction.get("detections", []),
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )
    for number, detection in enumerate(ordered, start=1):
        rows.append(
            {
                "Detection": number,
                "Class": detection["class_name"],
                "Confidence": float(detection["confidence"]),
                "x1": float(detection["x1"]),
                "y1": float(detection["y1"]),
                "x2": float(detection["x2"]),
                "y2": float(detection["y2"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _temporary_input_path(workspace: Path, suffix: str) -> Path:
    """Generate an input path that cannot escape its temporary workspace."""

    candidate = (workspace / f"input_{secrets.token_hex(8)}{suffix}").resolve()
    if candidate.parent != workspace.resolve():
        raise RuntimeError("Generated temporary input path escaped its workspace.")
    return candidate


def _inspect_open_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VisionUploadError("Uploaded content is not a video OpenCV can open.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not math.isfinite(fps) or fps <= 0:
            raise VisionUploadError("Uploaded video has missing or invalid FPS metadata.")
        if width <= 0 or height <= 0:
            raise VisionUploadError("Uploaded video has invalid frame dimensions.")
        readable, first_frame = capture.read()
        if not readable or first_frame is None or first_frame.size == 0:
            raise VisionUploadError("Uploaded video contains no readable frames.")
        if frame_count <= 0:
            raise VisionUploadError("Uploaded video reports zero frames.")
        return {
            "fps": fps,
            "reported_frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": frame_count / fps,
        }
    finally:
        capture.release()


def inspect_video_upload(content: bytes, filename: str) -> dict[str, Any]:
    """Validate video extension, bytes, container metadata, and a real frame read."""

    safe_name = sanitized_upload_name(filename, VIDEO_SUFFIXES)
    _validate_size(content, MAX_VIDEO_BYTES, "video")
    with tempfile.TemporaryDirectory(prefix="solarpredict_video_check_") as directory:
        workspace = Path(directory).resolve()
        path = _temporary_input_path(workspace, Path(safe_name).suffix.lower())
        path.write_bytes(content)
        metadata = _inspect_open_video(path)
    return {
        "filename": safe_name,
        "sha256": sha256_bytes(content),
        "byte_count": len(content),
        **metadata,
    }


def _validate_generated_video(path: Path, expected_frames: int) -> None:
    metadata = _inspect_open_video(path)
    if metadata["reported_frame_count"] != expected_frames:
        raise OSError(
            "Annotated output frame count does not match the processor summary: "
            f"{metadata['reported_frame_count']} versus {expected_frames}."
        )


def _make_browser_video(source: Path, destination: Path) -> tuple[bool, str | None]:
    """Create H.264/yuv420p output; retain original MP4V when conversion fails."""

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.SubprocessError):
        return False, "Browser-compatible video conversion could not be completed."
    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        return False, "FFmpeg did not create a valid browser-compatible video."
    return True, None


def video_condition_summary(summary: dict[str, Any]) -> dict[str, str]:
    """Map aggregate model outputs to cautious visual-condition terminology."""

    counts = summary["detections_by_class"]
    detected = [name for name in CLASS_NAMES if int(counts.get(name, 0)) > 0]
    if not detected:
        return {
            "overall_status": "NO_DETECTION",
            "message": (
                "No visual conditions were detected by the model. This does not "
                "certify that panels are healthy or fault-free."
            ),
        }
    if any(name != "clean" for name in detected):
        return {
            "overall_status": "ATTENTION",
            "message": "Visual conditions requiring review were detected by the model.",
        }
    return {
        "overall_status": "CLEAN",
        "message": (
            "The model produced only clean-class detections. This is not a physical-"
            "health certification."
        ),
    }


def process_video_upload(
    content: bytes,
    filename: str,
    *,
    detector: Any,
    confidence: float = DEFAULT_CONFIDENCE,
    frame_stride: int = 1,
) -> dict[str, Any]:
    """Process one upload in an isolated temporary workspace and return bytes."""

    if not 0.05 <= float(confidence) <= 0.95:
        raise VisionUploadError("Video confidence must be between 0.05 and 0.95.")
    if not 1 <= int(frame_stride) <= 10:
        raise VisionUploadError("Frame stride must be between 1 and 10.")
    upload = inspect_video_upload(content, filename)
    run_id = secrets.token_hex(6)
    with tempfile.TemporaryDirectory(prefix="solarpredict_video_run_") as directory:
        workspace = Path(directory).resolve()
        input_path = _temporary_input_path(
            workspace, Path(upload["filename"]).suffix.lower()
        )
        original_output = workspace / f"annotated_{run_id}_opencv.mp4"
        browser_output = workspace / f"annotated_{run_id}_browser.mp4"
        detection_log = workspace / f"detections_{run_id}.csv"
        frame_log = workspace / f"frames_{run_id}.json"
        summary_path = workspace / f"summary_{run_id}.json"
        input_path.write_bytes(content)
        summary = process_video(
            input_path,
            original_output,
            detection_log,
            frame_log,
            summary_path,
            confidence=float(confidence),
            frame_stride=int(frame_stride),
            detector=detector,
            device="cpu",
        )
        _validate_generated_video(original_output, int(summary["output_frames_written"]))
        browser_ready, conversion_error = _make_browser_video(
            original_output, browser_output
        )
        playback_path = browser_output if browser_ready else original_output
        _validate_generated_video(playback_path, int(summary["output_frames_written"]))

        detection_bytes = detection_log.read_bytes()
        detection_frame = pd.read_csv(BytesIO(detection_bytes))
        if tuple(detection_frame.columns) != DETECTION_LOG_COLUMNS:
            raise OSError("Generated detection CSV has an unexpected schema.")
        frame_payload = json.loads(frame_log.read_text(encoding="utf-8"))
        if len(frame_payload.get("frames", [])) != int(summary["total_frames"]):
            raise OSError("Generated frame log does not cover every source frame.")

        public_summary = {
            **summary,
            "input_video": upload["filename"],
            "annotated_video": f"annotated_inspection_{run_id}.mp4",
            "detection_log_csv": f"inspection_detections_{run_id}.csv",
            "frame_log_json": f"inspection_frames_{run_id}.json",
            "temporary_files_cleaned_after_processing": True,
            "browser_compatible_h264_created": browser_ready,
            "browser_conversion_error": conversion_error,
        }
        summary_bytes = (json.dumps(public_summary, indent=2) + "\n").encode("utf-8")
        return {
            "run_id": run_id,
            "filename": upload["filename"],
            "upload_sha256": upload["sha256"],
            "input_metadata": upload,
            "confidence_threshold": float(confidence),
            "frame_stride": int(frame_stride),
            "summary": public_summary,
            "condition_summary": video_condition_summary(public_summary),
            "detections": detection_frame,
            "annotated_video_bytes": playback_path.read_bytes(),
            "annotated_original_bytes": original_output.read_bytes(),
            "browser_playback_ready": browser_ready,
            "browser_conversion_error": conversion_error,
            "detection_csv_bytes": detection_bytes,
            "frame_json_bytes": frame_log.read_bytes(),
            "summary_json_bytes": summary_bytes,
            "download_names": {
                "annotated_video": f"annotated_inspection_{run_id}.mp4",
                "detection_csv": f"inspection_detections_{run_id}.csv",
                "frame_json": f"inspection_frames_{run_id}.json",
                "summary_json": f"inspection_summary_{run_id}.json",
            },
        }
