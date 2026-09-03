"""Tests for the bounded Day 8 basic feature-engineering pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.engineer_basic_features import (
    ENGINEERED_FEATURES,
    ORIGINAL_COLUMNS,
    add_basic_features,
    engineer_basic_feature_file,
)


def basic_feature_fixture() -> pd.DataFrame:
    """Return a complete verified-schema fixture with known calculations."""
    timestamp = pd.to_datetime(
        [
            "2022-05-01 00:00:00",
            "2022-05-01 06:30:00",
            "2022-05-01 12:00:00",
            "2022-05-01 18:45:00",
        ]
    )
    return pd.DataFrame(
        {
            "Timestamp": timestamp,
            "Air_Temp": [20.0, 21.0, 22.0, 23.0],
            "Relative_Humidity": [60.0, 55.0, 50.0, 45.0],
            "Wind_Speed": [0.0, 1.0, 0.5, 0.2],
            "Wind_Direction": [365.0, 10.0, 370.0, 200.0],
            "Solar_Radiation": [-1.0, 5.0, 5.000001, 100.0],
            "RTD_1": [1.0, 2.0, 3.0, 4.0],
            "RTD_2": [2.0, 3.0, 4.0, 5.0],
            "RTD_3": [3.0, 4.0, 5.0, 6.0],
            "RTD_4": [4.0, 5.0, 6.0, 7.0],
            "RTD_5": [5.0, 6.0, 7.0, 8.0],
            "Array_Voltage": [2.0, 3.0, 4.0, 5.0],
            "Array_Current": [5.0, 6.0, 7.0, 8.0],
            "Power_Generated": [10.0, 18.5, 28.0, 39.0],
        }
    )[[*ORIGINAL_COLUMNS]]


def test_time_and_cyclical_features_use_correct_periods() -> None:
    featured = add_basic_features(basic_feature_fixture())

    assert featured["hour"].tolist() == [0, 6, 12, 18]
    assert featured["minute"].tolist() == [0, 30, 0, 45]
    assert featured["minute_of_day"].tolist() == [0, 390, 720, 1125]
    assert featured["elapsed_minutes_from_start"].tolist() == [0.0, 390.0, 720.0, 1125.0]
    np.testing.assert_allclose(
        featured["hour_sin"],
        np.sin(2.0 * np.pi * featured["hour"] / 24.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        featured["hour_cos"],
        np.cos(2.0 * np.pi * featured["hour"] / 24.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        featured["minute_of_day_sin"],
        np.sin(2.0 * np.pi * featured["minute_of_day"] / 1440.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        featured["minute_of_day_cos"],
        np.cos(2.0 * np.pi * featured["minute_of_day"] / 1440.0),
        atol=1e-12,
    )


def test_rtd_features_are_population_summaries() -> None:
    featured = add_basic_features(basic_feature_fixture())

    assert featured.loc[0, "rtd_mean"] == 3.0
    assert featured.loc[0, "rtd_min"] == 1.0
    assert featured.loc[0, "rtd_max"] == 5.0
    assert featured.loc[0, "rtd_range"] == 4.0
    assert featured.loc[0, "rtd_std"] == np.std([1, 2, 3, 4, 5], ddof=0)
    assert (featured["rtd_min"] <= featured["rtd_mean"]).all()
    assert (featured["rtd_mean"] <= featured["rtd_max"]).all()
    assert (featured["rtd_range"] >= 0).all()


def test_electrical_consistency_features_match_documented_formulas() -> None:
    featured = add_basic_features(basic_feature_fixture())

    assert featured["electrical_power_product"].tolist() == [10.0, 18.0, 28.0, 40.0]
    assert featured["power_consistency_residual"].tolist() == [0.0, 0.5, 0.0, -1.0]
    assert featured["power_consistency_abs_error"].tolist() == [0.0, 0.5, 0.0, 1.0]


def test_low_light_flag_keeps_radiation_and_wind_direction_unchanged() -> None:
    source = basic_feature_fixture()
    featured = add_basic_features(source)

    assert featured["low_light_context"].tolist() == [1, 1, 0, 0]
    pd.testing.assert_series_equal(
        featured["Solar_Radiation"], source["Solar_Radiation"]
    )
    pd.testing.assert_series_equal(
        featured["Wind_Direction"], source["Wind_Direction"]
    )
    assert "wind_direction_sin" not in featured.columns
    assert "wind_direction_cos" not in featured.columns


def test_file_pipeline_preserves_original_data_and_writes_summary(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "operational_cleaned.csv"
    output_path = tmp_path / "operational_features_basic.csv"
    summary_path = tmp_path / "basic_feature_summary.json"
    source = basic_feature_fixture()
    source.to_csv(input_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    original_bytes = input_path.read_bytes()

    report = engineer_basic_feature_file(
        input_path,
        output_csv=output_path,
        summary_json=summary_path,
    )
    output = pd.read_csv(output_path)
    original_from_disk = pd.read_csv(input_path)
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert input_path.read_bytes() == original_bytes
    assert output.shape == (4, 31)
    assert output.columns.tolist() == [*ORIGINAL_COLUMNS, *ENGINEERED_FEATURES]
    pd.testing.assert_frame_equal(
        output.loc[:, list(ORIGINAL_COLUMNS)], original_from_disk, check_exact=True
    )
    assert report["engineered_feature_count"] == 17
    assert report["validation"]["passed"] is True
    assert saved_summary["validation"]["passed"] is True
    assert saved_summary["validation"]["engineered_missing_values"] == 0
    assert not any("rolling" in column for column in output.columns)
