"""Future trained-detector evaluation interface; no results exist on Day 24."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.computer_vision.inspect_vision_dataset import DEFAULT_DATA_YAML, load_dataset_yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models/computer_vision/yolo11n_solar_panel_defects/best.pt"
DEFAULT_PROJECT = ROOT / "outputs/computer_vision/evaluation"


def evaluate_detector(
    weights: Path,
    data_yaml: Path = DEFAULT_DATA_YAML,
    *,
    split: str = "test",
) -> Any:
    """Evaluate real trained weights when they exist; never fabricate metrics."""

    if not weights.is_file():
        raise FileNotFoundError(
            f"Trained weights not found: {weights}. Complete a future YOLO training stage first."
        )
    dataset = load_dataset_yaml(data_yaml)
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be 'val' or 'test'.")
    if dataset.split_paths[split] is None:
        raise ValueError(f"Dataset data.yaml does not configure a '{split}' split.")
    model = YOLO(str(weights), task="detect")
    return model.val(
        data=str(dataset.yaml_path), split=split, project=str(DEFAULT_PROJECT),
        name=f"{split}_evaluation", save_json=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    try:
        evaluate_detector(args.weights, args.data, split=args.split)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Evaluation blocked: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
