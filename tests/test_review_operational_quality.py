"""Tests for the read-only Day 7 operational quality review."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_processing.load_operational_data import SELECTED_NUMERIC_COLUMNS
from src.data_processing.review_operational_quality import (
    analyze_correlations,
    analyze_negative_solar_radiation,
    analyze_temporal_coverage,
    analyze_wind_direction,
    load_review_dataframe,
    review_operational_quality,
)


def review_fixture() -> pd.DataFrame:
    """Build a complete-schema fixture with two descriptive light cycles."""
    timestamps = pd.date_range("2022-05-01 00:00:00", periods=8, freq="2min")
    air_temperature = [20.0, 21.0, 24.0, 27.0, 22.0, 21.0, 25.0, 26.0]
    voltage = [2.0, 2.1, 10.0, 20.0, 2.2, 2.0, 15.0, 18.0]
    current = [5.40, 5.39, 5.38, 5.37, 5.40, 5.40, 5.36, 5.35]
    dataframe = pd.DataFrame(
        {
            "Timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "Air_Temp": air_temperature,
            "Relative_Humidity": [70.0, 68.0, 50.0, 40.0, 65.0, 67.0, 45.0, 42.0],
            "Wind_Speed": [0.1, 0.2, 0.4, 0.6, 0.1, 0.2, 0.5, 0.4],
            "Wind_Direction": [100.0, 365.0, 200.0, 370.0, 371.0, 250.0, 360.0, 40.0],
            "Solar_Radiation": [-0.5, -0.1, 10.0, 20.0, -0.2, -0.4, 15.0, 20.0],
            "RTD_1": [value + 5.0 for value in air_temperature],
            "RTD_2": [value + 5.5 for value in air_temperature],
            "RTD_3": [value + 6.0 for value in air_temperature],
            "RTD_4": [value + 6.5 for value in air_temperature],
            "RTD_5": [value + 7.0 for value in air_temperature],
            "Array_Voltage": voltage,
            "Array_Current": current,
            "Power_Generated": [
                voltage_value * current_value
                for voltage_value, current_value in zip(voltage, current, strict=True)
            ],
        }
    )
    return dataframe[["Timestamp", *SELECTED_NUMERIC_COLUMNS]]


def write_review_fixture(path: Path) -> bytes:
    """Write the fixture and return its original bytes."""
    review_fixture().to_csv(path, index=False)
    return path.read_bytes()


def test_review_loader_uses_verified_schema_without_modifying_file(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "operational_cleaned.csv"
    original = write_review_fixture(csv_path)

    dataframe = load_review_dataframe(csv_path)

    assert csv_path.read_bytes() == original
    assert dataframe.columns.tolist() == ["Timestamp", *SELECTED_NUMERIC_COLUMNS]
    assert pd.api.types.is_datetime64_any_dtype(dataframe["Timestamp"])
    assert dataframe["Timestamp"].dt.tz is None
    assert len(dataframe) == 8


def test_negative_radiation_review_reports_statistics_and_clusters() -> None:
    dataframe = review_fixture()
    dataframe["Timestamp"] = pd.to_datetime(dataframe["Timestamp"])

    result = analyze_negative_solar_radiation(dataframe)

    assert result["count"] == 4
    assert result["percentage_of_all_observations"] == 50.0
    assert result["minimum"] == -0.5
    assert result["median"] == -0.30000000000000004
    assert result["maximum"] == -0.1
    assert len(result["timestamps"]) == 4
    assert result["broad_time_cluster_count"] == 2
    assert result["contiguous_negative_run_count"] == 2
    assert result["longest_contiguous_negative_run_observations"] == 2
    assert result["data_action"].startswith("Retained unchanged")


def test_wind_direction_review_includes_neighbors_and_sequence_pattern() -> None:
    dataframe = review_fixture()
    dataframe["Timestamp"] = pd.to_datetime(dataframe["Timestamp"])

    result = analyze_wind_direction(dataframe)

    assert result["count"] == 3
    assert result["values"] == [365.0, 370.0, 371.0]
    assert result["maximum"] == 371.0
    assert result["sequence_count"] == 2
    assert result["isolated_observation_count"] == 1
    assert result["repeated_sequence_count"] == 1
    first_case = result["cases_with_immediate_neighbors"][0]
    assert first_case["previous_observation"]["Wind_Direction"] == 100.0
    assert first_case["next_observation"]["Wind_Direction"] == 200.0


def test_temporal_review_counts_radiation_derived_periods() -> None:
    dataframe = review_fixture()
    dataframe["Timestamp"] = pd.to_datetime(dataframe["Timestamp"])

    result = analyze_temporal_coverage(dataframe)

    assert result["calendar_date_count"] == 1
    assert result["median_sampling_interval_seconds"] == 120.0
    assert result["daylight_like_period_count"] == 2
    assert result["night_like_period_count"] == 2
    assert result["robust_daily_evaluation_supported"] is False


def test_complete_review_writes_strict_json_and_preserves_input(tmp_path: Path) -> None:
    csv_path = tmp_path / "operational_cleaned.csv"
    output_path = tmp_path / "quality_review.json"
    original = write_review_fixture(csv_path)

    report = review_operational_quality(csv_path, output_json=output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    correlations = analyze_correlations(load_review_dataframe(csv_path))

    assert csv_path.read_bytes() == original
    assert saved["data_modified"] is False
    assert saved["rows"] == 8
    assert saved["columns"] == 14
    assert report["column_names"] == ["Timestamp", *SELECTED_NUMERIC_COLUMNS]
    assert correlations["array_power_identity_check"][
        "maximum_absolute_difference_from_power_generated"
    ] < 1e-12
    assert not any(column.startswith("rolling_") for column in report["column_names"])
