"""Tests for causal Day 10 change and rate-of-change features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.engineer_basic_features import (
    ORIGINAL_COLUMNS,
    add_basic_features,
)
from src.data_processing.engineer_change_features import (
    CHANGE_FEATURES,
    INPUT_COLUMNS,
    add_change_features,
    engineer_change_feature_file,
    safe_relative_change,
)
from src.data_processing.engineer_rolling_features import add_rolling_features


def change_fixture(rows: int = 35) -> pd.DataFrame:
    """Create a verified 61-column Day 9-style fixture."""
    values = np.arange(1, rows + 1, dtype=float)
    current = np.full(rows, 5.0)
    operational = pd.DataFrame(
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
    return add_rolling_features(add_basic_features(operational))


def test_first_differences_and_rates_use_previous_row_and_two_minutes() -> None:
    featured = add_change_features(change_fixture())

    assert pd.isna(featured.loc[0, "power_generated_diff"])
    assert pd.isna(featured.loc[0, "power_generated_rate_per_min"])
    assert featured.loc[1, "power_generated_diff"] == 1.0
    assert featured.loc[1, "power_generated_rate_per_min"] == 0.5
    assert featured.loc[1, "solar_radiation_diff"] == 2.0
    assert featured.loc[1, "solar_radiation_rate_per_min"] == 1.0
    assert featured.loc[1, "rtd_mean_diff"] == 1.0
    assert featured.loc[1, "rtd_mean_rate_per_min"] == 0.5


def test_safe_relative_change_masks_zero_and_near_zero_denominators() -> None:
    values = pd.Series([0.0, 0.001, 2.0, 4.0])

    result = safe_relative_change(values, minimum_absolute_denominator=1.0)

    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == 1.0
    assert not np.isinf(result.to_numpy()).any()


def test_project_relative_change_thresholds_are_applied() -> None:
    source = change_fixture()
    source.loc[0, "Power_Generated"] = 0.5
    source.loc[0, "Solar_Radiation"] = 4.999
    featured = add_change_features(source)

    assert pd.isna(featured.loc[1, "power_generated_relative_change"])
    assert pd.isna(featured.loc[1, "solar_radiation_relative_change"])
    assert not np.isinf(
        featured.loc[:, list(CHANGE_FEATURES)].to_numpy(dtype=float)
    ).any()


def test_rolling_baseline_deviations_match_documented_formulas() -> None:
    featured = add_change_features(change_fixture())

    # At row 14, the 30-minute window contains values 1 through 15.
    assert featured.loc[14, "power_generated_roll_mean_30m"] == 8.0
    assert featured.loc[14, "power_deviation_from_30m_mean"] == 7.0
    assert featured.loc[14, "power_deviation_ratio_30m"] == 7.0 / 8.0
    assert featured.loc[14, "solar_radiation_roll_mean_30m"] == 16.0
    assert featured.loc[14, "solar_deviation_from_30m_mean"] == 14.0
    assert featured.loc[14, "solar_deviation_ratio_30m"] == 14.0 / 16.0
    assert featured["power_deviation_from_30m_mean"].iloc[:14].isna().all()


def test_change_features_do_not_use_a_future_observation() -> None:
    source = change_fixture()
    changed_future = source.copy(deep=True)
    changed_future.loc[21, "Power_Generated"] = 10000.0
    changed_future.loc[21, "Solar_Radiation"] = 10000.0

    original_features = add_change_features(source)
    changed_features = add_change_features(changed_future)

    pd.testing.assert_series_equal(
        original_features.loc[20, list(CHANGE_FEATURES)],
        changed_features.loc[20, list(CHANGE_FEATURES)],
    )
    assert (
        original_features.loc[21, "power_generated_diff"]
        != changed_features.loc[21, "power_generated_diff"]
    )


def test_file_pipeline_preserves_day9_fields_and_writes_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "operational_features_rolling.csv"
    output_path = tmp_path / "operational_features_change.csv"
    summary_path = tmp_path / "change_feature_summary.json"
    source = change_fixture()
    source.to_csv(input_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    original_bytes = input_path.read_bytes()

    report = engineer_change_feature_file(
        input_path,
        output_csv=output_path,
        summary_json=summary_path,
    )
    output = pd.read_csv(output_path)
    input_from_disk = pd.read_csv(input_path)
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert input_path.read_bytes() == original_bytes
    assert output.shape == (35, 81)
    assert output.columns.tolist() == [*INPUT_COLUMNS, *CHANGE_FEATURES]
    pd.testing.assert_frame_equal(
        output.loc[:, list(INPUT_COLUMNS)], input_from_disk, check_exact=True
    )
    assert report["new_feature_count"] == 20
    assert report["validation"]["passed"] is True
    assert saved_summary["validation"]["passed"] is True
    assert not np.isinf(output.loc[:, list(CHANGE_FEATURES)].to_numpy()).any()
