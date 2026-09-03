"""Reproduce the Day 7 operational data-quality and feature-planning evidence.

This module is deliberately read-only with respect to the dataset. It reviews
the cleaned CSV, reports unusual measurements and relationships, and does not
create features, change values, label anomalies, or train a model.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__:
    from .load_operational_data import (
        DEFAULT_TIMESTAMP_COLUMN,
        SELECTED_NUMERIC_COLUMNS,
        load_operational_csv,
    )
else:
    from load_operational_data import (  # type: ignore[no-redef]
        DEFAULT_TIMESTAMP_COLUMN,
        SELECTED_NUMERIC_COLUMNS,
        load_operational_csv,
    )


DEFAULT_INPUT_PATH = Path("data/processed/operational_cleaned.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/operational_quality_review.json")
PROCESSED_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMEZONE_NOTE = "Not specified by dataset source; timestamps kept timezone-naive."
LOW_LIGHT_THRESHOLD = 5.0
STRONG_CORRELATION_THRESHOLD = 0.90
REQUIRED_COLUMNS = (DEFAULT_TIMESTAMP_COLUMN, *SELECTED_NUMERIC_COLUMNS)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Review the cleaned operational dataset without modifying it."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Cleaned CSV path (default: {DEFAULT_INPUT_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"JSON output path (default: {DEFAULT_OUTPUT_PATH.as_posix()}).",
    )
    return parser.parse_args()


def _relative_or_absolute(path: Path) -> str:
    """Return a project-relative path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _json_number(value: object) -> float | int | None:
    """Convert a scalar number to a strict JSON-compatible value."""
    if value is None or pd.isna(value):
        return None
    converted = float(value)
    if not math.isfinite(converted):
        return None
    if converted.is_integer() and isinstance(value, int):
        return int(value)
    return converted


def _timestamp_text(value: object) -> str | None:
    """Serialize a valid timestamp without assigning a timezone."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat(sep=" ")


def _require_columns(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Raise a clear error if the verified schema is incomplete."""
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError("Required columns are missing: " + ", ".join(missing))


def load_review_dataframe(csv_path: str | Path) -> pd.DataFrame:
    """Load the cleaned dataset using the existing conservative loader.

    Source order and duplicate rows are retained because this stage is a review,
    not another preprocessing pass.
    """
    dataframe = load_operational_csv(
        csv_path,
        timestamp_column=DEFAULT_TIMESTAMP_COLUMN,
        timestamp_format=PROCESSED_TIMESTAMP_FORMAT,
        numeric_columns=SELECTED_NUMERIC_COLUMNS,
        sort_chronologically=False,
        drop_duplicate_rows=False,
    )
    _require_columns(dataframe, REQUIRED_COLUMNS)
    if dataframe.columns.tolist() != list(REQUIRED_COLUMNS):
        raise ValueError(
            "The cleaned dataset does not match the verified 14-column order."
        )
    if dataframe[DEFAULT_TIMESTAMP_COLUMN].isna().any():
        failures = int(dataframe[DEFAULT_TIMESTAMP_COLUMN].isna().sum())
        raise ValueError(f"The cleaned dataset has {failures} timestamp parse failures.")
    return dataframe


def summarize_schema(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Return observed dtype and descriptive statistics for every column."""
    _require_columns(dataframe, REQUIRED_COLUMNS)
    summary: list[dict[str, object]] = []
    for column in dataframe.columns:
        values = dataframe[column]
        if column == DEFAULT_TIMESTAMP_COLUMN:
            summary.append(
                {
                    "column": column,
                    "dtype": str(values.dtype),
                    "minimum": _timestamp_text(values.min()),
                    "maximum": _timestamp_text(values.max()),
                    "mean": None,
                    "median": None,
                }
            )
        else:
            summary.append(
                {
                    "column": column,
                    "dtype": str(values.dtype),
                    "minimum": _json_number(values.min()),
                    "maximum": _json_number(values.max()),
                    "mean": _json_number(values.mean()),
                    "median": _json_number(values.median()),
                }
            )
    return summary


def _measurement_summary(
    dataframe: pd.DataFrame, columns: tuple[str, ...]
) -> dict[str, dict[str, float | int | None]]:
    """Summarize selected numeric measurements."""
    available = tuple(column for column in columns if column in dataframe.columns)
    result: dict[str, dict[str, float | int | None]] = {}
    for column in available:
        values = dataframe[column]
        result[column] = {
            "count": int(values.notna().sum()),
            "minimum": _json_number(values.min()),
            "maximum": _json_number(values.max()),
            "mean": _json_number(values.mean()),
            "median": _json_number(values.median()),
        }
    return result


def _period_records(
    dataframe: pd.DataFrame,
    state: pd.Series,
    *,
    true_label: str,
    false_label: str,
) -> list[dict[str, object]]:
    """Describe contiguous true/false periods in source-row order."""
    if dataframe.empty:
        return []
    groups = state.ne(state.shift()).cumsum()
    periods: list[dict[str, object]] = []
    for _, period in dataframe.groupby(groups, sort=False):
        is_true = bool(state.loc[period.index[0]])
        periods.append(
            {
                "classification": true_label if is_true else false_label,
                "start": _timestamp_text(period[DEFAULT_TIMESTAMP_COLUMN].iloc[0]),
                "end": _timestamp_text(period[DEFAULT_TIMESTAMP_COLUMN].iloc[-1]),
                "observations": int(len(period)),
                "solar_radiation_minimum": _json_number(
                    period["Solar_Radiation"].min()
                ),
                "solar_radiation_maximum": _json_number(
                    period["Solar_Radiation"].max()
                ),
                "touches_dataset_boundary": bool(
                    period.index[0] == dataframe.index[0]
                    or period.index[-1] == dataframe.index[-1]
                ),
            }
        )
    return periods


def analyze_temporal_coverage(
    dataframe: pd.DataFrame,
    *,
    low_light_threshold: float = LOW_LIGHT_THRESHOLD,
) -> dict[str, object]:
    """Describe dates, duration, cadence, and radiation-derived light periods."""
    _require_columns(dataframe, (DEFAULT_TIMESTAMP_COLUMN, "Solar_Radiation"))
    timestamps = dataframe[DEFAULT_TIMESTAMP_COLUMN]
    ordered = timestamps.sort_values().reset_index(drop=True)
    positive_intervals = ordered.diff().dropna()
    positive_intervals = positive_intervals[positive_intervals > pd.Timedelta(0)]
    median_interval = positive_intervals.median() if not positive_intervals.empty else pd.NaT

    dates = timestamps.dt.date
    counts = dates.value_counts().sort_index()
    complete_dates: list[str] = []
    if not pd.isna(median_interval) and median_interval > pd.Timedelta(0):
        expected_per_day = int(pd.Timedelta(days=1) / median_interval)
        for date_value, count in counts.items():
            rows = dataframe.loc[dates == date_value, DEFAULT_TIMESTAMP_COLUMN]
            if (
                int(count) == expected_per_day
                and rows.min().time() == datetime.min.time()
                and rows.max().time()
                == (pd.Timestamp(date_value) + pd.Timedelta(days=1) - median_interval).time()
            ):
                complete_dates.append(str(date_value))

    daylight_state = dataframe["Solar_Radiation"] > low_light_threshold
    periods = _period_records(
        dataframe,
        daylight_state,
        true_label="daylight-like",
        false_label="night-like",
    )
    daylight_periods = [
        period for period in periods if period["classification"] == "daylight-like"
    ]
    night_periods = [
        period for period in periods if period["classification"] == "night-like"
    ]

    start = timestamps.min()
    end = timestamps.max()
    return {
        "start": _timestamp_text(start),
        "end": _timestamp_text(end),
        "duration_hours": _json_number((end - start).total_seconds() / 3600),
        "timezone": TIMEZONE_NOTE,
        "calendar_date_count": int(dates.nunique()),
        "calendar_dates": [str(value) for value in counts.index],
        "observations_per_calendar_date": {
            str(date): int(count) for date, count in counts.items()
        },
        "complete_calendar_dates": complete_dates,
        "median_sampling_interval_seconds": (
            _json_number(median_interval.total_seconds())
            if not pd.isna(median_interval)
            else None
        ),
        "light_period_method": (
            f"Descriptive threshold only: Solar_Radiation > {low_light_threshold:g} "
            "is daylight-like; values at or below the threshold are night-like. "
            "This is not a source-provided label or an anomaly rule."
        ),
        "daylight_like_period_count": int(len(daylight_periods)),
        "night_like_period_count": int(len(night_periods)),
        "light_periods": periods,
        "robust_daily_evaluation_supported": False,
        "limitation": (
            "Only one complete calendar date and partial boundary dates are "
            "available; this is insufficient for robust repeated-daily-cycle, "
            "seasonal, or long-term model evaluation."
        ),
    }


def analyze_negative_solar_radiation(
    dataframe: pd.DataFrame,
    *,
    low_light_threshold: float = LOW_LIGHT_THRESHOLD,
) -> dict[str, object]:
    """Review negative radiation readings without labeling or modifying them."""
    _require_columns(
        dataframe,
        (
            DEFAULT_TIMESTAMP_COLUMN,
            "Solar_Radiation",
            "Power_Generated",
            "Array_Voltage",
            "Array_Current",
        ),
    )
    negative = dataframe.loc[dataframe["Solar_Radiation"] < 0].copy()
    count = int(len(negative))
    total = int(len(dataframe))

    if negative.empty:
        return {
            "condition": "Solar_Radiation < 0",
            "count": 0,
            "percentage_of_all_observations": 0.0,
            "timestamps": [],
            "broad_time_clusters": [],
        }

    # Negative readings fluctuate around zero, so broad clustering is based on
    # contiguous low-light periods rather than treating every sign change as a
    # separate night. The threshold is explicitly descriptive, not a label.
    low_light = dataframe["Solar_Radiation"] <= low_light_threshold
    period_groups = low_light.ne(low_light.shift()).cumsum()
    broad_clusters: list[dict[str, object]] = []
    for _, period in dataframe.groupby(period_groups, sort=False):
        if not bool(low_light.loc[period.index[0]]):
            continue
        negative_in_period = period.loc[period["Solar_Radiation"] < 0]
        if negative_in_period.empty:
            continue
        broad_clusters.append(
            {
                "low_light_period_start": _timestamp_text(
                    period[DEFAULT_TIMESTAMP_COLUMN].iloc[0]
                ),
                "low_light_period_end": _timestamp_text(
                    period[DEFAULT_TIMESTAMP_COLUMN].iloc[-1]
                ),
                "first_negative_timestamp": _timestamp_text(
                    negative_in_period[DEFAULT_TIMESTAMP_COLUMN].iloc[0]
                ),
                "last_negative_timestamp": _timestamp_text(
                    negative_in_period[DEFAULT_TIMESTAMP_COLUMN].iloc[-1]
                ),
                "negative_observations": int(len(negative_in_period)),
            }
        )

    negative_positions = dataframe.index[dataframe["Solar_Radiation"] < 0].to_series()
    contiguous_groups = negative_positions.diff().ne(1).cumsum()
    contiguous_sizes = negative_positions.groupby(contiguous_groups).size()

    return {
        "condition": "Solar_Radiation < 0",
        "count": count,
        "percentage_of_all_observations": _json_number(count / total * 100),
        "minimum": _json_number(negative["Solar_Radiation"].min()),
        "median": _json_number(negative["Solar_Radiation"].median()),
        "maximum": _json_number(negative["Solar_Radiation"].max()),
        "first_timestamp": _timestamp_text(
            negative[DEFAULT_TIMESTAMP_COLUMN].iloc[0]
        ),
        "last_timestamp": _timestamp_text(
            negative[DEFAULT_TIMESTAMP_COLUMN].iloc[-1]
        ),
        "timestamps": [
            _timestamp_text(value)
            for value in negative[DEFAULT_TIMESTAMP_COLUMN].tolist()
        ],
        "counts_by_calendar_date": {
            str(date): int(value)
            for date, value in negative.groupby(
                negative[DEFAULT_TIMESTAMP_COLUMN].dt.date
            ).size().items()
        },
        "clock_hours_present": sorted(
            int(value) for value in negative[DEFAULT_TIMESTAMP_COLUMN].dt.hour.unique()
        ),
        "low_light_threshold": low_light_threshold,
        "within_descriptive_low_light_periods_count": int(
            low_light.loc[negative.index].sum()
        ),
        "within_descriptive_low_light_periods_percentage": _json_number(
            low_light.loc[negative.index].mean() * 100
        ),
        "broad_time_cluster_count": int(len(broad_clusters)),
        "broad_time_clusters": broad_clusters,
        "contiguous_negative_run_count": int(len(contiguous_sizes)),
        "longest_contiguous_negative_run_observations": int(contiguous_sizes.max()),
        "measurements_during_negative_radiation": _measurement_summary(
            negative,
            (
                "Solar_Radiation",
                "Power_Generated",
                "Array_Voltage",
                "Array_Current",
            ),
        ),
        "interpretation": (
            "The readings are concentrated in two evening-to-morning low-light "
            "windows and are mostly associated with low electrical output. "
            "Transition-edge readings include higher output. Sensor offset, "
            "nighttime noise, or another encoding behavior are possible "
            "interpretations, but none is confirmed by the available metadata."
        ),
        "data_action": "Retained unchanged; not classified as sensor errors or anomalies.",
    }


def _row_record(row: pd.Series | None) -> dict[str, object] | None:
    """Serialize every field in one neighboring row."""
    if row is None:
        return None
    record: dict[str, object] = {}
    for column, value in row.items():
        if column == DEFAULT_TIMESTAMP_COLUMN:
            record[column] = _timestamp_text(value)
        elif pd.isna(value):
            record[column] = None
        else:
            record[column] = _json_number(value)
    return record


def analyze_wind_direction(dataframe: pd.DataFrame) -> dict[str, object]:
    """Review wind-direction values above 360 with immediate neighbors."""
    _require_columns(dataframe, (DEFAULT_TIMESTAMP_COLUMN, "Wind_Direction"))
    mask = dataframe["Wind_Direction"] > 360
    unusual_positions = list(dataframe.index[mask])
    cases: list[dict[str, object]] = []
    for position in unusual_positions:
        integer_position = dataframe.index.get_loc(position)
        previous = (
            dataframe.iloc[integer_position - 1] if integer_position > 0 else None
        )
        current = dataframe.iloc[integer_position]
        following = (
            dataframe.iloc[integer_position + 1]
            if integer_position + 1 < len(dataframe)
            else None
        )
        cases.append(
            {
                "timestamp": _timestamp_text(current[DEFAULT_TIMESTAMP_COLUMN]),
                "wind_direction": _json_number(current["Wind_Direction"]),
                "previous_observation": _row_record(previous),
                "current_observation": _row_record(current),
                "next_observation": _row_record(following),
            }
        )

    runs: list[dict[str, object]] = []
    if unusual_positions:
        positions = pd.Series(
            [dataframe.index.get_loc(position) for position in unusual_positions]
        )
        group_ids = positions.diff().ne(1).cumsum()
        for _, position_group in positions.groupby(group_ids):
            group_rows = dataframe.iloc[position_group.tolist()]
            runs.append(
                {
                    "start": _timestamp_text(
                        group_rows[DEFAULT_TIMESTAMP_COLUMN].iloc[0]
                    ),
                    "end": _timestamp_text(
                        group_rows[DEFAULT_TIMESTAMP_COLUMN].iloc[-1]
                    ),
                    "observations": int(len(group_rows)),
                    "values": [
                        _json_number(value)
                        for value in group_rows["Wind_Direction"].tolist()
                    ],
                }
            )

    return {
        "condition": "Wind_Direction > 360",
        "count": int(mask.sum()),
        "maximum": (
            _json_number(dataframe.loc[mask, "Wind_Direction"].max())
            if mask.any()
            else None
        ),
        "values": [
            _json_number(value)
            for value in dataframe.loc[mask, "Wind_Direction"].tolist()
        ],
        "cases_with_immediate_neighbors": cases,
        "sequence_count": int(len(runs)),
        "isolated_observation_count": int(
            sum(run["observations"] == 1 for run in runs)
        ),
        "repeated_sequence_count": int(
            sum(run["observations"] > 1 for run in runs)
        ),
        "sequences": runs,
        "interpretation": (
            "Most readings are isolated; one pair is consecutive. The encoding "
            "cannot be determined conclusively without confirmation from the "
            "dataset source."
        ),
        "data_action": "Retained unchanged; not wrapped, clipped, removed, or labeled.",
    }


def analyze_correlations(
    dataframe: pd.DataFrame,
    *,
    strong_threshold: float = STRONG_CORRELATION_THRESHOLD,
) -> dict[str, object]:
    """Calculate descriptive Pearson relationships among actual measurements."""
    numeric = dataframe.loc[:, list(SELECTED_NUMERIC_COLUMNS)]
    correlation = numeric.corr(method="pearson")

    matrix: dict[str, dict[str, float | int | None]] = {}
    for row_name in correlation.index:
        matrix[row_name] = {
            column: _json_number(correlation.loc[row_name, column])
            for column in correlation.columns
        }

    def relationships_with(column: str) -> dict[str, float | int | None]:
        values = correlation[column].drop(labels=[column]).sort_values(ascending=False)
        return {name: _json_number(value) for name, value in values.items()}

    pairs: list[dict[str, object]] = []
    names = list(correlation.columns)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= strong_threshold:
                pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "pearson_correlation": _json_number(value),
                        "absolute_correlation": _json_number(abs(float(value))),
                    }
                )
    pairs.sort(key=lambda pair: float(pair["absolute_correlation"] or 0), reverse=True)

    voltage_current_product = dataframe["Array_Voltage"] * dataframe["Array_Current"]
    product_error = dataframe["Power_Generated"] - voltage_current_product
    electrical_identity = {
        "expression": "Array_Voltage * Array_Current",
        "pearson_correlation_with_power_generated": _json_number(
            voltage_current_product.corr(dataframe["Power_Generated"])
        ),
        "maximum_absolute_difference_from_power_generated": _json_number(
            product_error.abs().max()
        ),
        "median_absolute_difference_from_power_generated": _json_number(
            product_error.abs().median()
        ),
        "mean_signed_difference_from_power_generated": _json_number(
            product_error.mean()
        ),
        "planning_warning": (
            "This near-deterministic relationship may be useful for consistency "
            "checks but can cause target leakage or redundant weighting, depending "
            "on the later anomaly-detection objective."
        ),
    }

    return {
        "method": "Pearson correlation on the 13 numeric columns",
        "scope_warning": (
            "Relationships are descriptive for this short, autocorrelated time "
            "window and do not establish causality or justify deleting variables."
        ),
        "correlation_matrix": matrix,
        "with_power_generated": relationships_with("Power_Generated"),
        "with_solar_radiation": relationships_with("Solar_Radiation"),
        "strong_absolute_threshold": strong_threshold,
        "strong_pairs": pairs,
        "array_power_identity_check": electrical_identity,
    }


def build_quality_review(dataframe: pd.DataFrame, source_path: str | Path) -> dict[str, object]:
    """Build the complete machine-readable Day 7 review."""
    _require_columns(dataframe, REQUIRED_COLUMNS)
    return {
        "review_stage": "Day 7 operational data quality review and feature planning",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": _relative_or_absolute(Path(source_path)),
        "data_modified": False,
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "column_names": dataframe.columns.tolist(),
        "timezone": TIMEZONE_NOTE,
        "schema_statistics": summarize_schema(dataframe),
        "negative_solar_radiation": analyze_negative_solar_radiation(dataframe),
        "wind_direction_above_360": analyze_wind_direction(dataframe),
        "temporal_coverage": analyze_temporal_coverage(dataframe),
        "relationships": analyze_correlations(dataframe),
        "limitations": [
            "The file spans 33.6 hours and contains only one complete calendar date.",
            "The source does not specify the timezone.",
            "Several measurement meanings or units remain unconfirmed.",
            "No quality flag in this review is an anomaly label.",
        ],
    }


def write_json_report(report: dict[str, object], destination: str | Path) -> None:
    """Atomically write a strict, UTF-8 JSON report."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".json.tmp",
            dir=output.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(report, temporary_file, indent=2, ensure_ascii=False, allow_nan=False)
            temporary_file.write("\n")
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def review_operational_quality(
    csv_path: str | Path = DEFAULT_INPUT_PATH,
    *,
    output_json: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    """Load, review, and save the operational quality report."""
    dataframe = load_review_dataframe(csv_path)
    report = build_quality_review(dataframe, csv_path)
    write_json_report(report, output_json)
    return report


def main() -> int:
    """Run the command-line review and print a compact summary."""
    arguments = parse_args()
    try:
        report = review_operational_quality(
            arguments.csv_path, output_json=arguments.output
        )
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}")
        return 1

    negative = report["negative_solar_radiation"]
    wind = report["wind_direction_above_360"]
    temporal = report["temporal_coverage"]
    print(f"Reviewed: {report['source_file']}")
    print(f"Shape: {report['rows']} rows x {report['columns']} columns")
    print(
        "Negative Solar_Radiation: "
        f"{negative['count']} ({negative['percentage_of_all_observations']:.3f}%)"
    )
    print(f"Wind_Direction > 360: {wind['count']}")
    print(
        "Coverage: "
        f"{temporal['start']} to {temporal['end']} "
        f"({temporal['duration_hours']:.1f} hours)"
    )
    print(f"Report: {_relative_or_absolute(arguments.output)}")
    print("No data was modified and no features or anomaly labels were created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
