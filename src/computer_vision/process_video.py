"""Sequential image/video inference with annotated video and structured logs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.computer_vision.inference import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIDENCE,
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_SIZE,
    draw_detections,
    load_detector,
    predict_frame,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_DIR = ROOT / "outputs/computer_vision/video"
DETECTION_LOG_COLUMNS = (
    "frame_index",
    "video_time_seconds",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
)


def _writer_codec(path: Path) -> str:
    return "mp4v" if path.suffix.lower() in {".mp4", ".m4v", ".mov"} else "MJPG"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def process_video(
    input_path: str | Path,
    output_video_path: str | Path,
    detection_log_path: str | Path,
    frame_log_path: str | Path,
    summary_path: str | Path,
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    confidence: float = DEFAULT_CONFIDENCE,
    frame_stride: int = 1,
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: str = DEFAULT_DEVICE,
    detector: Any | None = None,
    prediction_function: Callable[..., dict[str, Any]] = predict_frame,
) -> dict[str, Any]:
    """Process a video, writing every input frame and analyzing the selected stride.

    Frames excluded by ``frame_stride`` are still written to the output and logged
    explicitly as not analyzed; they are never silently discarded.
    """

    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1.")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1 inclusive.")

    output_video = Path(output_video_path)
    detection_log = Path(detection_log_path)
    frame_log = Path(frame_log_path)
    summary_output = Path(summary_path)
    for destination in (output_video, detection_log, frame_log, summary_output):
        destination.parent.mkdir(parents=True, exist_ok=True)
    if output_video.resolve() == source:
        raise ValueError("Annotated output path must differ from the input path.")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    warnings: list[str] = []
    if fps <= 0 or not np.isfinite(fps):
        fps = 30.0
        warnings.append("Source FPS was unavailable; output uses a documented 30 FPS fallback.")
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Video reports invalid frame dimensions.")

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*_writer_codec(output_video)),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"OpenCV could not create output video: {output_video}")

    if detector is None and prediction_function is predict_frame:
        detector = load_detector(checkpoint)

    started = time.perf_counter()
    inference_seconds = 0.0
    total_frames = 0
    analyzed_frames = 0
    detections_by_class: Counter[str] = Counter()
    frames_by_class: Counter[str] = Counter()
    max_confidence: dict[str, float | None] = {name: None for name in CLASS_NAMES}
    detection_rows: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_index = total_frames
            video_time = frame_index / fps
            analyzed = frame_index % frame_stride == 0
            if analyzed:
                inference_started = time.perf_counter()
                prediction = prediction_function(
                    frame,
                    model=detector,
                    confidence=confidence,
                    image_size=image_size,
                    device=device,
                )
                inference_seconds += time.perf_counter() - inference_started
                analyzed_frames += 1
                detections = prediction["detections"]
                annotated = draw_detections(frame, detections)
                present_classes = set()
                for detection in detections:
                    name = str(detection["class_name"])
                    present_classes.add(name)
                    detections_by_class[name] += 1
                    value = float(detection["confidence"])
                    max_confidence[name] = (
                        value if max_confidence[name] is None
                        else max(max_confidence[name], value)
                    )
                    detection_rows.append(
                        {
                            "frame_index": frame_index,
                            "video_time_seconds": video_time,
                            **{key: detection[key] for key in DETECTION_LOG_COLUMNS[2:]},
                        }
                    )
                frames_by_class.update(present_classes)
                frame_records.append(
                    {
                        "frame_index": frame_index,
                        "video_time_seconds": video_time,
                        "analyzed": True,
                        "detection_count": prediction["detection_count"],
                        "condition_summary": prediction["condition_summary"],
                        "detections": detections,
                    }
                )
            else:
                annotated = frame.copy()
                cv2.putText(
                    annotated,
                    f"Not analyzed (frame_stride={frame_stride})",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 215, 255),
                    2,
                    cv2.LINE_AA,
                )
                frame_records.append(
                    {
                        "frame_index": frame_index,
                        "video_time_seconds": video_time,
                        "analyzed": False,
                        "detection_count": None,
                        "condition_summary": None,
                        "detections": [],
                    }
                )
            writer.write(annotated)
            total_frames += 1
    finally:
        capture.release()
        writer.release()

    processing_seconds = time.perf_counter() - started
    if total_frames == 0:
        raise ValueError("Input video contained no readable frames.")

    with detection_log.open("w", newline="", encoding="utf-8") as destination:
        csv_writer = csv.DictWriter(destination, fieldnames=DETECTION_LOG_COLUMNS)
        csv_writer.writeheader()
        csv_writer.writerows(detection_rows)

    frame_payload = {
        "schema_version": 1,
        "input_video": str(source),
        "frame_stride": frame_stride,
        "records_include_all_source_frames": True,
        "frames": frame_records,
    }
    _write_json(frame_log, frame_payload)

    summary = {
        "schema_version": 1,
        "input_video": str(source),
        "input_video_name": source.name,
        "annotated_video": str(output_video.resolve()),
        "detection_log_csv": str(detection_log.resolve()),
        "frame_log_json": str(frame_log.resolve()),
        "duration_seconds": total_frames / fps,
        "source_fps": fps,
        "output_fps": fps,
        "frame_width": width,
        "frame_height": height,
        "total_frames": total_frames,
        "output_frames_written": total_frames,
        "analyzed_frames": analyzed_frames,
        "frame_stride": frame_stride,
        "confidence_threshold": float(confidence),
        "confidence_purpose": "deployment/viewing; not selected from test metrics",
        "total_detections": len(detection_rows),
        "detections_by_class": {
            name: detections_by_class.get(name, 0) for name in CLASS_NAMES
        },
        "frames_containing_each_class": {
            name: frames_by_class.get(name, 0) for name in CLASS_NAMES
        },
        "maximum_confidence_by_class": max_confidence,
        "processing_duration_seconds": processing_seconds,
        "inference_duration_seconds": inference_seconds,
        "average_inference_seconds_per_analyzed_frame": (
            inference_seconds / analyzed_frames if analyzed_frames else None
        ),
        "approximate_inference_fps": (
            analyzed_frames / inference_seconds if inference_seconds > 0 else None
        ),
        "approximate_end_to_end_processing_fps": total_frames / processing_seconds,
        "warnings": warnings,
    }
    _write_json(summary_output, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_video", type=Path)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--detection-log", type=Path, required=True)
    parser.add_argument("--frame-log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args()
    try:
        summary = process_video(
            args.input_video,
            args.output_video,
            args.detection_log,
            args.frame_log,
            args.summary,
            checkpoint=args.checkpoint,
            confidence=args.confidence,
            frame_stride=args.frame_stride,
            image_size=args.image_size,
            device=args.device,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.exit(2, f"Video processing blocked: {error}\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
