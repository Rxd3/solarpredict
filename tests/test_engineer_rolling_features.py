"""Tests for full-window, trailing-only Day 9 rolling features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.engineer_basic_features import (
    ORIGINAL_COLUMNS,
    add_basic_features,
)
from src.data_processing.engineer_rolling_features import (
    INPUT_COLUMNS,
    ROLLING_FEATURES,
    add_rolling_features,
    engineer_rolling_feature_file,
)


def rolling_fixture(rows: int = 35) -> pd.DataFrame:
    """Build a 31-column basic-feature fixture with two-minute sampling."""
    values = np.arange(1, rows + 1, dtype=float)
    current = np.full(rows, 5.0)
    dataframe = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2022-05-01", periods=rows, freq="2min"),
            "Air_Temp": 20.0 + values,
            "Relative_Humidity": 80.0 - values,
            "Wind_Speed": values / 10.0,
            "Wind_Direction": (values * 10.0) % 360.0,
            "Solar_Radiation": values * 2.0,
            "RTD_1": 30.0 + values,
            "RTD_2": 31.0 + values,
            "RTD_3": 32.0 + values,
            "RTD_4": 33.0 + values,
            "RTD_5": 34.0 + values,
            "Array_Voltage": values / current,
            "Array_Current": current,
            "Power_Generated": values,
        }
    )[[*ORIGINAL_COLUMNS]]
    return add_basic_features(dataframe)


def test_ten_minute_features_use_five_complete_samples() -> None:
    featured = add_rolling_features(rolling_fixture())

    assert featured["power_generated_roll_mean_10m"].iloc[:4].isna().all()
    assert featured.loc[4, "power_generated_roll_mean_10m"] == 3.0
    assert featured.loc[4, "power_generated_roll_min_10m"] == 1.0
    assert featured.loc[4, "power_generated_roll_max_10m"] == 5.0
    assert featured.loc[4, "power_generated_roll_std_10m"] == np.std(
        [1, 2, 3, 4, 5], ddof=0
    )


def test_thirty_minute_features_use_fifteen_complete_samples() -> None:
    featured = add_rolling_features(rolling_fixture())

    assert featured["power_generated_roll_mean_30m"].iloc[:14].isna().all()
    assert featured.loc[14, "power_generated_roll_mean_30m"] == 8.0
    assert featured.loc[14, "solar_radiation_roll_min_30m"] == 2.0
    assert featured.loc[14, "solar_radiation_roll_max_30m"] == 30.0
    assert featured.loc[14, "air_temp_roll_mean_30m"] == 28.0
    assert featured.loc[14, "rtd_mean_roll_mean_30m"] == 40.0


def test_sixty_minute_features_use_thirty_complete_samples() -> None:
    featured = add_rolling_features(rolling_fixture())

    assert featured["power_generated_roll_mean_60m"].iloc[:29].isna().all()
    assert featured.loc[29, "power_generated_roll_mean_60m"] == 15.5
    assert featured.loc[29, "power_generated_roll_min_60m"] == 1.0
    assert featured.loc[29, "power_generated_roll_max_60m"] == 30.0
    assert featured.loc[29, "relative_humidity_roll_mean_60m"] == 64.5


def test_rolling_features_do_not_use_future_observations() -> None:
    source = rolling_fixture()
    changed_future = source.copy(deep=True)
    changed_future.loc[5, "Power_Generated"] = 10000.0

    original_features = add_rolling_features(source)
    changed_features = add_rolling_features(changed_future)

    assert (
        original_features.loc[4, "power_generated_roll_mean_10m"]
        == changed_features.loc[4, "power_generated_roll_mean_10m"]
    )
    assert (
        original_features.loc[5, "power_generated_roll_mean_10m"]
        != changed_features.loc[5, "power_generated_roll_mean_10m"]
    )


def test_rows_columns_and_min_mean_max_relationships_are_preserved() -> None:
    source = rolling_fixture()
    featured = add_rolling_features(source)

    assert featured.shape == (35, 61)
    assert featured.columns.tolist() == [*INPUT_COLUMNS, *ROLLING_FEATURES]
    pd.testing.assert_frame_equal(featured.loc[:, list(INPUT_COLUMNS)], source)
    for prefix in ("power_generated", "solar_radiation"):
        for minutes in (10, 30, 60):
            minimum = featured[f"{prefix}_roll_min_{minutes}m"]
            mean = featured[f"{prefix}_roll_mean_{minutes}m"]
            maximum = featured[f"{prefix}_roll_max_{minutes}m"]
            valid = minimum.notna()
            assert (minimum[valid] <= mean[valid]).all()
            assert (mean[valid] <= maximum[valid]).all()


def test_file_pipeline_writes_valid_summary_and_single_figure(tmp_path: Path) -> None:
    input_path = tmp_path / "operational_features_basic.csv"
    output_path = tmp_path / "operational_features_rolling.csv"
    summary_path = tmp_path / "rolling_feature_summary.json"
    figure_path = tmp_path / "day9_power_rolling_mean.png"
    source = rolling_fixture()
    source.to_csv(input_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    original_bytes = input_path.read_bytes()

    report = engineer_rolling_feature_file(
        input_path,
        output_csv=output_path,
        summary_json=summary_path,
        figure_path=figure_path,
    )
    output = pd.read_csv(output_path)
    input_from_disk = pd.read_csv(input_path)
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert input_path.read_bytes() == original_bytes
    assert output.shape == (35, 61)
    pd.testing.assert_frame_equal(
        output.loc[:, list(INPUT_COLUMNS)], input_from_disk, check_exact=True
    )
    assert report["rolling_feature_count"] == 30
    assert report["nan_counts_by_window"]["10m"][
        "expected_initial_nan_count_per_feature"
    ] == 4
    assert report["nan_counts_by_window"]["30m"][
        "expected_initial_nan_count_per_feature"
    ] == 14
    assert report["nan_counts_by_window"]["60m"][
        "expected_initial_nan_count_per_feature"
    ] == 29
    assert saved_summary["validation"]["passed"] is True
    assert figure_path.is_file() and figure_path.stat().st_size > 0
