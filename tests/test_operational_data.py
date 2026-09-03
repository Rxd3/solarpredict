"""Tests for conservative operational-data loading and inspection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_processing.inspect_dataset import build_inspection_report
from src.data_processing.load_operational_data import (
    build_timestamp_profile,
    discover_csv_files,
    load_operational_csv,
)


def write_operational_fixture(path: Path) -> None:
    """Write a tiny fixture with one duplicate and out-of-order timestamps."""
    path.write_text(
        "Timestamp,Air_Temp,Power_Generated\n"
        "01-05-2022 00:04,22.0,101.0\n"
        "01-05-2022 00:00,20.0,99.0\n"
        "01-05-2022 00:00,20.0,99.0\n"
        "01-05-2022 00:02,not-numeric,100.0\n",
        encoding="utf-8",
    )


def test_loader_sorts_deduplicates_and_coerces_safely(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    write_operational_fixture(csv_path)

    dataframe = load_operational_csv(
        csv_path,
        numeric_columns=["Air_Temp", "Power_Generated"],
    )

    assert len(dataframe) == 3
    assert dataframe["Timestamp"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(dataframe["Timestamp"].dtype)
    assert dataframe["Timestamp"].dt.tz is None
    assert dataframe["Air_Temp"].isna().sum() == 1
    assert dataframe.attrs["duplicate_rows_found"] == 1
    assert dataframe.attrs["duplicate_rows_removed"] == 1


def test_loader_can_preserve_duplicate_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    write_operational_fixture(csv_path)

    dataframe = load_operational_csv(
        csv_path,
        numeric_columns=["Air_Temp", "Power_Generated"],
        drop_duplicate_rows=False,
        sort_chronologically=False,
    )

    assert len(dataframe) == 4
    assert dataframe.attrs["duplicate_rows_removed"] == 0


def test_inspection_report_includes_duplicates_and_sampling(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    write_operational_fixture(csv_path)

    report = build_inspection_report(csv_path)
    timestamp = report["timestamp_profile"]

    assert report["rows"] == 4
    assert report["columns"] == 3
    assert report["duplicate_rows"] == 1
    assert report["missing_values"] == {
        "Timestamp": 0,
        "Air_Temp": 0,
        "Power_Generated": 0,
    }
    assert timestamp["median_interval_seconds"] == 120.0
    assert timestamp["duplicate_timestamps"] == 1
    assert timestamp["source_order_monotonic_increasing"] is False


def test_timestamp_profile_reports_an_irregular_gap() -> None:
    timestamps = pd.Series(
        ["01-05-2022 00:00", "01-05-2022 00:02", "01-05-2022 00:08"]
    )

    profile = build_timestamp_profile(timestamps)

    assert profile["median_interval_seconds"] == 240.0
    assert profile["irregular_gap_count"] == 2


def test_csv_discovery_is_recursive_and_sorted(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    files = discover_csv_files(tmp_path)

    assert [path.name for path in files] == ["a.csv", "b.csv"]
