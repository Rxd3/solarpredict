"""Tests for the conservative operational preprocessing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_processing.load_operational_data import SELECTED_NUMERIC_COLUMNS
from src.data_processing.preprocess_operational_data import (
    OUTPUT_TIMESTAMP_FORMAT,
    preprocess_operational_data,
    validate_preprocessed_data,
)


def operational_rows() -> list[dict[str, object]]:
    """Return a complete-schema fixture with order, duplicates, and bad values."""
    base: dict[str, object] = {
        "Air_Temp": 20.0,
        "Relative_Humidity": 40.0,
        "Wind_Speed": 1.0,
        "Wind_Direction": 180.0,
        "Solar_Radiation": 100.0,
        "RTD_1": 30.0,
        "RTD_2": 31.0,
        "RTD_3": 32.0,
        "RTD_4": 33.0,
        "RTD_5": 34.0,
        "Array_Voltage": 50.0,
        "Array_Current": 5.0,
        "Power_Generated": 250.0,
    }
    later = {"Timestamp": "01-05-2022 00:04", **base}
    earlier = {"Timestamp": "01-05-2022 00:00", **base, "Power_Generated": 240.0}
    middle = {
        "Timestamp": "01-05-2022 00:02",
        **base,
        "Air_Temp": None,
        "Wind_Direction": 361.0,
        "Solar_Radiation": -0.2,
        "Power_Generated": 245.0,
    }
    return [later, earlier, earlier.copy(), middle]


def write_fixture(path: Path) -> bytes:
    """Write the fixture and return its original bytes."""
    dataframe = pd.DataFrame(operational_rows())
    dataframe = dataframe[["Timestamp", *SELECTED_NUMERIC_COLUMNS]]
    dataframe.to_csv(path, index=False)
    return path.read_bytes()


def test_preprocessing_preserves_schema_missing_values_and_raw_file(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    summary_path = tmp_path / "summary.json"
    raw_bytes = write_fixture(raw_path)

    report = preprocess_operational_data(
        raw_path,
        output_csv=processed_path,
        summary_json=summary_path,
    )
    processed = pd.read_csv(processed_path)

    assert raw_path.read_bytes() == raw_bytes
    assert processed.columns.tolist() == ["Timestamp", *SELECTED_NUMERIC_COLUMNS]
    assert len(processed) == 3
    assert processed["Air_Temp"].isna().sum() == 1
    assert report["rows_before_preprocessing"] == 4
    assert report["rows_after_preprocessing"] == 3
    assert report["duplicate_rows_found"] == 1
    assert report["duplicate_rows_removed"] == 1
    assert report["total_missing_values_before"] == 1
    assert report["total_missing_values_after"] == 1
    assert report["raw_file_unchanged"] is True
    assert report["validation"]["passed"] is True
    assert json.loads(summary_path.read_text(encoding="utf-8"))[
        "validation"
    ]["passed"] is True


def test_preprocessing_sorts_timestamps_and_keeps_suspicious_values(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    write_fixture(raw_path)

    report = preprocess_operational_data(
        raw_path,
        output_csv=processed_path,
        summary_json=tmp_path / "summary.json",
    )
    processed = pd.read_csv(processed_path)
    timestamps = pd.to_datetime(
        processed["Timestamp"], format=OUTPUT_TIMESTAMP_FORMAT
    )

    assert timestamps.is_monotonic_increasing
    assert processed.loc[1, "Solar_Radiation"] == -0.2
    assert processed.loc[1, "Wind_Direction"] == 361.0
    assert report["source_order_was_changed_by_sorting"] is True
    assert any("Solar_Radiation" in warning for warning in report["processing_warnings"])
    assert any("Wind_Direction" in warning for warning in report["processing_warnings"])


def test_invalid_numeric_text_becomes_missing_without_dropping_row(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    rows = operational_rows()
    rows[0]["Air_Temp"] = "invalid"
    dataframe = pd.DataFrame(rows)[["Timestamp", *SELECTED_NUMERIC_COLUMNS]]
    dataframe.to_csv(raw_path, index=False)

    report = preprocess_operational_data(
        raw_path,
        output_csv=processed_path,
        summary_json=tmp_path / "summary.json",
    )
    processed = pd.read_csv(processed_path)

    assert len(processed) == 3
    assert processed["Air_Temp"].isna().sum() == 2
    assert report["invalid_numeric_values_coerced_to_missing"]["Air_Temp"] == 1


def test_validation_detects_a_missing_required_column(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    write_fixture(raw_path)
    processed = pd.read_csv(raw_path).drop(columns=["Power_Generated"])
    processed["Timestamp"] = pd.to_datetime(
        processed["Timestamp"], format="%d-%m-%Y %H:%M"
    ).dt.strftime(OUTPUT_TIMESTAMP_FORMAT)
    processed.to_csv(processed_path, index=False)

    validation = validate_preprocessed_data(raw_path, processed_path)

    assert validation["passed"] is False
    assert "Power_Generated" in validation["missing_required_columns"]
