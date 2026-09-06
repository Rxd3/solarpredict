"""Read-only image and frame inference for the frozen solar-panel detector."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "models/computer_vision/solar_panel_detector_best.pt"
CLASS_NAMES = (
    "bird-drop",
    "clean",
    "dusty",
    "electrical-damage",
    "physical-damage",
    "snow-covered",
)
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 512
DEFAULT_DEVICE = "cpu"


def _ordered_model_names(names: Any) -> tuple[str, ...]:
    if isinstance(names, dict):
        return tuple(str(names[index]) for index in sorted(names))
    return tuple(str(name) for name in names)


@lru_cache(maxsize=4)
def _load_detector_cached(checkpoint: str) -> YOLO:
    model = YOLO(checkpoint, task="detect")
    actual_names = _ordered_model_names(model.names)
    if actual_names != CLASS_NAMES:
        raise ValueError(
            "Checkpoint class mapping does not match the frozen six-class contract: "
            f"{actual_names!r}."
        )
    return model


def load_detector(checkpoint: str | Path = DEFAULT_CHECKPOINT) -> YOLO:
    """Load and cache the frozen detector after validating its class mapping."""

    resolved = Path(checkpoint).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Frozen detector checkpoint not found: {resolved}")
    return _load_detector_cached(str(resolved))


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy.ndarray in OpenCV BGR format.")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("frame must be a non-empty HxWx3 OpenCV BGR image.")


def _validate_confidence(confidence: float) -> float:
    value = float(confidence)
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be between 0 and 1 inclusive.")
    return value


def _serialize_result(result: Any, width: int, height: int) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return detections
    coordinates = result.boxes.xyxy.cpu().tolist()
    class_ids = result.boxes.cls.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    for raw_box, raw_class_id, raw_confidence in zip(
        coordinates, class_ids, confidences
    ):
        class_id = int(raw_class_id)
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"Detector returned unknown class id {class_id}.")
        raw_x1, raw_y1, raw_x2, raw_y2 = (float(value) for value in raw_box)
        x1, x2 = sorted((raw_x1, raw_x2))
        y1, y2 = sorted((raw_y1, raw_y2))
        # Clip tiny numerical overshoots so every API box is safe for rendering.
        x1 = min(max(x1, 0.0), float(width))
        x2 = min(max(x2, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        y2 = min(max(y2, 0.0), float(height))
        detections.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "confidence": min(max(float(raw_confidence), 0.0), 1.0),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return detections


def summarize_conditions(detections: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Convert detections into a cautious dashboard-ready condition summary."""

    detection_list = list(detections)
    counts = {name: 0 for name in CLASS_NAMES}
    maxima: dict[str, float | None] = {name: None for name in CLASS_NAMES}
    for detection in detection_list:
        name = str(detection["class_name"])
        if name not in counts:
            raise ValueError(f"Unknown condition in detection: {name!r}.")
        confidence = float(detection["confidence"])
        counts[name] += 1
        maxima[name] = confidence if maxima[name] is None else max(maxima[name], confidence)

    detected = [name for name in CLASS_NAMES if counts[name] > 0]
    if not detected:
        status = "NO_DETECTION"
    elif any(name != "clean" for name in detected):
        status = "ATTENTION"
    else:
        status = "CLEAN"
    return {
        "overall_status": status,
        "detected_conditions": detected,
        "counts_by_class": counts,
        "highest_confidence_by_class": maxima,
    }


def predict_frame(
    frame: np.ndarray,
    *,
    model: YOLO | None = None,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    confidence: float = DEFAULT_CONFIDENCE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: str = DEFAULT_DEVICE,
) -> dict[str, Any]:
    """Run inference on one OpenCV frame without modifying the input array."""

    _validate_frame(frame)
    confidence = _validate_confidence(confidence)
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    detector = model if model is not None else load_detector(checkpoint)
    results = detector.predict(
        source=frame,
        conf=confidence,
        imgsz=int(image_size),
        device=device,
        save=False,
        verbose=False,
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one inference result, received {len(results)}.")
    height, width = frame.shape[:2]
    detections = _serialize_result(results[0], width, height)
    classes_detected = [name for name in CLASS_NAMES if any(
        detection["class_name"] == name for detection in detections
    )]
    prediction = {
        "image_width": int(width),
        "image_height": int(height),
        "detection_count": len(detections),
        "classes_detected": classes_detected,
        "highest_confidence": max(
            (detection["confidence"] for detection in detections), default=None
        ),
        "detections": detections,
    }
    prediction["condition_summary"] = summarize_conditions(detections)
    return prediction


def predict_image(
    image_path: str | Path,
    *,
    model: YOLO | None = None,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    confidence: float = DEFAULT_CONFIDENCE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: str = DEFAULT_DEVICE,
) -> dict[str, Any]:
    """Read one image path and return structured detections; source is unchanged."""

    resolved = Path(image_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")
    frame = cv2.imread(str(resolved))
    if frame is None:
        raise ValueError(f"OpenCV could not read image: {resolved}")
    prediction = predict_frame(
        frame,
        model=model,
        checkpoint=checkpoint,
        confidence=confidence,
        image_size=image_size,
        device=device,
    )
    prediction["image_path"] = str(resolved)
    return prediction


def predict_images(
    image_paths: Iterable[str | Path],
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    confidence: float = DEFAULT_CONFIDENCE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: str = DEFAULT_DEVICE,
) -> list[dict[str, Any]]:
    """Predict multiple paths while loading the detector only once."""

    model = load_detector(checkpoint)
    return [
        predict_image(
            path,
            model=model,
            confidence=confidence,
            image_size=image_size,
            device=device,
        )
        for path in image_paths
    ]


def draw_detections(
    frame: np.ndarray, detections: Iterable[dict[str, Any]]
) -> np.ndarray:
    """Return an annotated copy of a frame, leaving the source array unchanged."""

    _validate_frame(frame)
    annotated = frame.copy()
    palette = (
        (0, 165, 255),
        (0, 180, 0),
        (0, 215, 255),
        (0, 0, 255),
        (180, 0, 255),
        (255, 180, 0),
    )
    for detection in detections:
        class_id = int(detection["class_id"])
        color = palette[class_id % len(palette)]
        x1, y1, x2, y2 = (
            int(round(float(detection[key]))) for key in ("x1", "y1", "x2", "y2")
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{detection['class_name']} {float(detection['confidence']):.2f}"
        text_y = max(18, y1 - 6)
        cv2.putText(
            annotated,
            label,
            (max(0, x1), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def save_annotated_image(
    frame: np.ndarray,
    detections: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Draw detections and save a new image file without touching its source."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), draw_detections(frame, detections)):
        raise OSError(f"OpenCV could not write annotated image: {destination}")
    return destination.resolve()
