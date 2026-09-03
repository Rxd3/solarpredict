"""Analyze operational-data quality and generate schema-aware EDA figures.

The script is read-only with respect to source CSV files. Quality flags are
reported, never silently removed or corrected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from .load_operational_data import (
        DEFAULT_TIMESTAMP_FORMAT,
        SELECTED_NUMERIC_COLUMNS,
        build_timestamp_profile,
        discover_csv_files,
        infer_numeric_columns,
        infer_timestamp_column,
        load_operational_csv,
        parse_timestamp_series,
    )
else:
    from load_operational_data import (  # type: ignore[no-redef]
        DEFAULT_TIMESTAMP_FORMAT,
        SELECTED_NUMERIC_COLUMNS,
        build_timestamp_profile,
        discover_csv_files,
        infer_numeric_columns,
        infer_timestamp_column,
        load_operational_csv,
        parse_timestamp_series,
    )


DATASET_NAME = "Solar Power Dataset"
DATASET_SOURCE = (
    "https://www.kaggle.com/datasets/s1nister/solar-power-generation-dataset"
)
DATASET_LICENSE = "CC0: Public Domain"
TIMEZONE_NOTE = "Not specified by dataset source; timestamps kept timezone-naive."

# Rules are deliberately limited to ranges with a clear physical interpretation
# and source-confirmed units. A flag is not an instruction to delete a record.
PHYSICAL_RANGE_RULES: dict[
    str, tuple[str, Callable[[pd.Series], pd.Series]]
] = {
    "Air_Temp": (
        "below absolute zero (-273.15 degrees Celsius)",
        lambda values: values < -273.15,
    ),
    "Relative_Humidity": (
        "outside 0 to 100 %RH",
        lambda values: (values < 0) | (values > 100),
    ),
    "Wind_Speed": ("below 0 m/s", lambda values: values < 0),
    "Wind_Direction": (
        "outside 0 to 360 degrees",
        lambda values: (values < 0) | (values > 360),
    ),
    "Solar_Radiation": ("below 0 W/m^2", lambda values: values < 0),
    "Power_Generated": ("below 0 W", lambda values: values < 0),
}

CONFIRMED_UNITS = {
    "Air_Temp": "degrees Celsius",
    "Relative_Humidity": "%RH",
    "Wind_Speed": "m/s",
    "Wind_Direction": "degrees",
    "Solar_Radiation": "W/m^2",
    "Power_Generated": "W",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a machine-readable quality report and exploratory figures "
            "for operational CSV data."
        )
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        type=Path,
        default=Path("data/operational"),
        help="CSV file or directory to analyze (default: data/operational).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/operational_data_quality.json"),
        help="Quality JSON path.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures/operational"),
        help="Directory for EDA figures.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Generate the quality JSON without figures.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Calculate a source-file digest without changing the file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: object) -> float | None:
    """Return a JSON-safe float or ``None`` for missing/non-finite values."""
    if pd.isna(value):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def analyze_csv(csv_path: str | Path) -> dict[str, object]:
    """Build a non-destructive quality report for one CSV file."""
    path = Path(csv_path)
    resolved_path = path.resolve()
    try:
        report_path = resolved_path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        report_path = str(resolved_path)
    inferred = pd.read_csv(path)
    text = pd.read_csv(path, dtype="string")
    timestamp_column = infer_timestamp_column(inferred.columns)

    inferred_numeric = infer_numeric_columns(
        inferred, excluded_columns=(timestamp_column,)
    )
    expected_numeric_columns = [
        column for column in SELECTED_NUMERIC_COLUMNS if column in inferred.columns
    ]
    for column in inferred_numeric:
        if column not in expected_numeric_columns:
            expected_numeric_columns.append(column)

    invalid_numeric: dict[str, dict[str, object]] = {}
    numeric_values: dict[str, pd.Series] = {}
    for column in expected_numeric_columns:
        original = text[column]
        present = original.notna() & original.str.strip().ne("")
        converted = pd.to_numeric(original, errors="coerce")
        invalid = present & converted.isna()
        numeric_values[column] = converted
        invalid_numeric[column] = {
            "count": int(invalid.sum()),
            "examples": original[invalid].drop_duplicates().head(10).tolist(),
        }

    timestamp_profile = build_timestamp_profile(
        inferred[timestamp_column], timestamp_format=DEFAULT_TIMESTAMP_FORMAT
    )

    physical_flags: dict[str, dict[str, object]] = {}
    for column, (rule, predicate) in PHYSICAL_RANGE_RULES.items():
        if column not in numeric_values:
            physical_flags[column] = {
                "rule": rule,
                "evaluated": False,
                "count": None,
                "note": "Column not present.",
            }
            continue

        values = numeric_values[column]
        mask = predicate(values) & values.notna()
        physical_flags[column] = {
            "rule": rule,
            "evaluated": True,
            "count": int(mask.sum()),
            "minimum_flagged_value": _finite_float(values[mask].min()),
            "maximum_flagged_value": _finite_float(values[mask].max()),
            "row_indices_sample": [int(index) for index in values[mask].index[:10]],
            "action": "flagged only; source rows retained",
        }

    numeric_statistics: dict[str, dict[str, float | None]] = {}
    for column, values in numeric_values.items():
        numeric_statistics[column] = {
            "count": _finite_float(values.count()),
            "mean": _finite_float(values.mean()),
            "standard_deviation": _finite_float(values.std()),
            "minimum": _finite_float(values.min()),
            "median": _finite_float(values.median()),
            "maximum": _finite_float(values.max()),
        }

    return {
        "filename": path.name,
        "path": report_path,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "rows": int(inferred.shape[0]),
        "columns": int(inferred.shape[1]),
        "column_names": inferred.columns.tolist(),
        "inferred_dtypes": {
            column: str(dtype) for column, dtype in inferred.dtypes.items()
        },
        "missing_values": {
            column: int(count) for column, count in inferred.isna().sum().items()
        },
        "total_missing_values": int(inferred.isna().sum().sum()),
        "duplicate_rows": int(inferred.duplicated().sum()),
        "unique_values_per_column": {
            column: int(inferred[column].nunique(dropna=False))
            for column in inferred.columns
        },
        "timestamp": {
            "column": timestamp_column,
            "format": DEFAULT_TIMESTAMP_FORMAT,
            "timezone": TIMEZONE_NOTE,
            **timestamp_profile,
        },
        "expected_numeric_columns": expected_numeric_columns,
        "invalid_non_numeric_values": invalid_numeric,
        "total_invalid_non_numeric_values": int(
            sum(details["count"] for details in invalid_numeric.values())
        ),
        "physical_range_flags": physical_flags,
        "numeric_statistics": numeric_statistics,
    }


def _column(dataframe: pd.DataFrame, *candidates: str) -> str | None:
    """Find the first present column, ignoring case and punctuation."""
    normalized = {
        "".join(char.lower() for char in column if char.isalnum()): column
        for column in dataframe.columns
    }
    for candidate in candidates:
        key = "".join(char.lower() for char in candidate if char.isalnum())
        if key in normalized:
            return normalized[key]
    return None


def _axis_label(column: str) -> str:
    """Create an axis label with a unit only when source-confirmed."""
    readable = column.replace("_", " ")
    unit = CONFIRMED_UNITS.get(column)
    return f"{readable} ({unit})" if unit else readable


def generate_eda_figures(csv_path: str | Path, figures_dir: str | Path) -> list[str]:
    """Generate plots only for measurements found in the actual CSV schema."""
    path = Path(csv_path)
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_operational_csv(
        path,
        numeric_columns=[
            column for column in SELECTED_NUMERIC_COLUMNS if column in pd.read_csv(path, nrows=0).columns
        ],
        drop_duplicate_rows=False,
    )
    timestamp = _column(dataframe, "Timestamp", "DateTime")
    power = _column(dataframe, "Power_Generated", "Generated_Power", "Power")
    radiation = _column(dataframe, "Solar_Radiation", "Irradiance")
    air_temperature = _column(dataframe, "Air_Temp", "Ambient_Temperature")
    voltage = _column(dataframe, "Array_Voltage", "Voltage")
    current = _column(dataframe, "Array_Current", "Current")

    prefix = f"{path.stem}_"
    saved: list[str] = []

    def save_figure(figure: plt.Figure, suffix: str) -> None:
        destination = output_dir / f"{prefix}{suffix}.png"
        figure.tight_layout()
        figure.savefig(destination, dpi=160, bbox_inches="tight")
        plt.close(figure)
        saved.append(destination.as_posix())

    if timestamp is not None:
        for column, title, suffix, color in (
            (power, "Generated Power over Time", "power_vs_time", "tab:blue"),
            (
                radiation,
                "Solar Radiation over Time",
                "solar_radiation_vs_time",
                "tab:orange",
            ),
            (
                air_temperature,
                "Air Temperature over Time",
                "ambient_temperature_vs_time",
                "tab:red",
            ),
        ):
            if column is None:
                continue
            valid = dataframe[timestamp].notna() & dataframe[column].notna()
            figure, axis = plt.subplots(figsize=(12, 4.5))
            axis.plot(
                dataframe.loc[valid, timestamp],
                dataframe.loc[valid, column],
                color=color,
                linewidth=1,
            )
            axis.set(title=title, xlabel="Timestamp (timezone not specified)", ylabel=_axis_label(column))
            axis.grid(alpha=0.25)
            save_figure(figure, suffix)

    if power is not None and radiation is not None:
        valid = dataframe[power].notna() & dataframe[radiation].notna()
        figure, axis = plt.subplots(figsize=(7, 5.5))
        axis.scatter(
            dataframe.loc[valid, radiation],
            dataframe.loc[valid, power],
            s=14,
            alpha=0.55,
            edgecolors="none",
        )
        axis.set(
            title="Generated Power versus Solar Radiation",
            xlabel=_axis_label(radiation),
            ylabel=_axis_label(power),
        )
        axis.grid(alpha=0.25)
        save_figure(figure, "power_vs_solar_radiation")

    if timestamp is not None and (voltage is not None or current is not None):
        figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        measurements = (
            (voltage, axes[0], "Array Voltage over Time", "tab:purple"),
            (current, axes[1], "Array Current over Time", "tab:green"),
        )
        for column, axis, title, color in measurements:
            if column is None:
                axis.set_visible(False)
                continue
            valid = dataframe[timestamp].notna() & dataframe[column].notna()
            axis.plot(
                dataframe.loc[valid, timestamp],
                dataframe.loc[valid, column],
                color=color,
                linewidth=1,
            )
            axis.set(title=title, ylabel=_axis_label(column))
            axis.grid(alpha=0.25)
        axes[-1].set_xlabel("Timestamp (timezone not specified)")
        save_figure(figure, "voltage_current_vs_time")

    numeric = dataframe.select_dtypes(include="number")
    if numeric.shape[1] >= 2:
        correlations = numeric.corr()
        figure, axis = plt.subplots(figsize=(11, 9))
        image = axis.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_xticks(np.arange(len(correlations.columns)))
        axis.set_yticks(np.arange(len(correlations.columns)))
        axis.set_xticklabels(correlations.columns, rotation=55, ha="right")
        axis.set_yticklabels(correlations.columns)
        axis.set_title("Pearson Correlation Matrix of Numerical Measurements")
        figure.colorbar(image, ax=axis, label="Pearson correlation")
        save_figure(figure, "correlation_matrix")

    if timestamp is not None and power is not None:
        daily = dataframe[[timestamp, power]].dropna().copy()
        if not daily.empty:
            daily["minute_of_day"] = daily[timestamp].dt.hour * 60 + daily[timestamp].dt.minute
            profile = daily.groupby("minute_of_day")[power].mean()
            figure, axis = plt.subplots(figsize=(10, 4.5))
            axis.plot(profile.index / 60, profile.values, color="tab:blue", linewidth=1.5)
            axis.set(
                title="Mean Generated Power by Time of Day",
                xlabel="Time of day (hours; timezone not specified)",
                ylabel=_axis_label(power),
                xlim=(0, 24),
            )
            axis.grid(alpha=0.25)
            save_figure(figure, "daily_power_pattern")

    important = [
        column
        for column in (
            power,
            radiation,
            air_temperature,
            _column(dataframe, "Relative_Humidity"),
            _column(dataframe, "Wind_Speed"),
            voltage,
            current,
        )
        if column is not None
    ]
    if important:
        columns_per_row = 3
        rows = math.ceil(len(important) / columns_per_row)
        figure, axes = plt.subplots(
            rows, columns_per_row, figsize=(13, 3.6 * rows), squeeze=False
        )
        for axis, column in zip(axes.flat, important, strict=False):
            axis.hist(dataframe[column].dropna(), bins=30, color="tab:blue", alpha=0.8)
            axis.set(title=f"Distribution of {column.replace('_', ' ')}", xlabel=_axis_label(column), ylabel="Count")
            axis.grid(axis="y", alpha=0.2)
        for axis in axes.flat[len(important) :]:
            axis.set_visible(False)
        save_figure(figure, "important_feature_distributions")

    return saved


def analyze_dataset(
    dataset_path: str | Path,
    *,
    output_path: str | Path,
    figures_dir: str | Path,
    generate_figures: bool = True,
) -> dict[str, object]:
    """Analyze all CSV files, optionally generate plots, and write JSON."""
    csv_files = discover_csv_files(dataset_path)
    if not csv_files:
        raise ValueError(f"No CSV files found under: {dataset_path}")

    file_reports = [analyze_csv(csv_file) for csv_file in csv_files]
    figure_paths: list[str] = []
    if generate_figures:
        for csv_file in csv_files:
            figure_paths.extend(generate_eda_figures(csv_file, figures_dir))

    report: dict[str, object] = {
        "dataset": {
            "name": DATASET_NAME,
            "source_url": DATASET_SOURCE,
            "license": DATASET_LICENSE,
            "timezone": TIMEZONE_NOTE,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files_are_modified": False,
        "file_count": len(file_reports),
        "files": file_reports,
        "totals": {
            "rows": int(sum(report["rows"] for report in file_reports)),
            "missing_values": int(
                sum(report["total_missing_values"] for report in file_reports)
            ),
            "duplicate_rows": int(
                sum(report["duplicate_rows"] for report in file_reports)
            ),
            "invalid_non_numeric_values": int(
                sum(
                    report["total_invalid_non_numeric_values"]
                    for report in file_reports
                )
            ),
        },
        "figures": figure_paths,
        "quality_policy": (
            "Flags are retained for review; this stage does not delete, impute, "
            "correct, or classify anomalies."
        ),
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, ensure_ascii=False, allow_nan=False)
        output_file.write("\n")
    return report


def main() -> int:
    """Run the quality/EDA pipeline and return a process exit code."""
    args = parse_args()
    try:
        report = analyze_dataset(
            args.dataset_path,
            output_path=args.output,
            figures_dir=args.figures_dir,
            generate_figures=not args.skip_figures,
        )
    except (
        FileNotFoundError,
        PermissionError,
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Error: unable to analyze operational data: {error}", file=sys.stderr)
        return 1

    print(f"Analyzed {report['file_count']} CSV file(s).")
    print(f"Rows: {report['totals']['rows']}")
    print(f"Missing values: {report['totals']['missing_values']}")
    print(f"Duplicate rows: {report['totals']['duplicate_rows']}")
    print(f"Quality report: {args.output}")
    print(f"Figures generated: {len(report['figures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
