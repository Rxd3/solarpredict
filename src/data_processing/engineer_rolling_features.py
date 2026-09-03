"""Create the trailing-only Day 9 operational rolling features.

All prior columns and rows are preserved. Rolling calculations include only the
current row and previous rows, require a complete window, and never backfill the
initial missing values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from .engineer_basic_features import (
        ENGINEERED_FEATURES as BASIC_ENGINEERED_FEATURES,
        ORIGINAL_COLUMNS,
        PROCESSED_TIMESTAMP_FORMAT,
    )
    from .load_operational_data import DEFAULT_TIMESTAMP_COLUMN, parse_timestamp_series
else:
    from engineer_basic_features import (  # type: ignore[no-redef]
        ENGINEERED_FEATURES as BASIC_ENGINEERED_FEATURES,
        ORIGINAL_COLUMNS,
        PROCESSED_TIMESTAMP_FORMAT,
    )
    from load_operational_data import (  # type: ignore[no-redef]
        DEFAULT_TIMESTAMP_COLUMN,
        parse_timestamp_series,
    )


DEFAULT_RAW_PATH = Path("data/operational/solar-power-dataset/solar_data.csv")
DEFAULT_CLEANED_PATH = Path("data/processed/operational_cleaned.csv")
DEFAULT_INPUT_PATH = Path("data/processed/operational_features_basic.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/operational_features_rolling.csv")
DEFAULT_SUMMARY_PATH = Path("outputs/rolling_feature_summary.json")
DEFAULT_FIGURE_PATH = Path(
    "outputs/figures/operational/day9_power_rolling_mean.png"
)
TIMEZONE_NOTE = "Not specified by dataset source; timestamps kept timezone-naive."
INPUT_COLUMNS = (*ORIGINAL_COLUMNS, *BASIC_ENGINEERED_FEATURES)
WINDOW_SAMPLES = {10: 5, 30: 15, 60: 30}
ROLLING_STANDARD_DEVIATION_DDOF = 0


@dataclass(frozen=True)
class RollingFeatureSpec:
    """Declarative definition of one trailing rolling feature."""

    name: str
    source_column: str
    statistic: str
    window_minutes: int
    window_samples: int


def _build_rolling_specs() -> tuple[RollingFeatureSpec, ...]:
    specs: list[RollingFeatureSpec] = []
    primary_signals = (
        ("Power_Generated", "power_generated"),
        ("Solar_Radiation", "solar_radiation"),
    )
    for source_column, prefix in primary_signals:
        for minutes, samples in WINDOW_SAMPLES.items():
            for statistic in ("mean", "std", "min", "max"):
                specs.append(
                    RollingFeatureSpec(
                        name=f"{prefix}_roll_{statistic}_{minutes}m",
                        source_column=source_column,
                        statistic=statistic,
                        window_minutes=minutes,
                        window_samples=samples,
                    )
                )

    for source_column, prefix in (
        ("Air_Temp", "air_temp"),
        ("Relative_Humidity", "relative_humidity"),
    ):
        for minutes in (30, 60):
            specs.append(
                RollingFeatureSpec(
                    name=f"{prefix}_roll_mean_{minutes}m",
                    source_column=source_column,
                    statistic="mean",
                    window_minutes=minutes,
                    window_samples=WINDOW_SAMPLES[minutes],
                )
            )

    specs.extend(
        (
            RollingFeatureSpec(
                name="rtd_mean_roll_mean_30m",
                source_column="rtd_mean",
                statistic="mean",
                window_minutes=30,
                window_samples=WINDOW_SAMPLES[30],
            ),
            RollingFeatureSpec(
                name="rtd_mean_roll_std_30m",
                source_column="rtd_mean",
                statistic="std",
                window_minutes=30,
                window_samples=WINDOW_SAMPLES[30],
            ),
        )
    )
    return tuple(specs)


ROLLING_FEATURE_SPECS = _build_rolling_specs()
ROLLING_FEATURES = tuple(spec.name for spec in ROLLING_FEATURE_SPECS)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create full-window, trailing-only Day 9 rolling features."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Basic feature CSV (default: {DEFAULT_INPUT_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Rolling feature CSV (default: {DEFAULT_OUTPUT_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Summary JSON (default: {DEFAULT_SUMMARY_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=DEFAULT_FIGURE_PATH,
        help=f"Validation figure (default: {DEFAULT_FIGURE_PATH.as_posix()}).",
    )
    return parser.parse_args()


def _relative_or_absolute(path: Path) -> str:
    """Return a project-relative path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without changing it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_number(value: object) -> float | None:
    """Convert a scalar to a finite JSON-compatible float."""
    if value is None or pd.isna(value):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _require_columns(dataframe: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error when an expected input column is missing."""
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError("Required columns are missing: " + ", ".join(missing))


def load_basic_feature_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the verified Day 8 feature CSV without sorting or dropping rows."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature CSV does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a CSV file but received: {path}")

    dataframe = pd.read_csv(path)
    _require_columns(dataframe, INPUT_COLUMNS)
    if dataframe.columns.tolist() != list(INPUT_COLUMNS):
        raise ValueError(
            "The input must match the verified Day 8 31-column schema and order."
        )

    parsed_timestamps = parse_timestamp_series(
        dataframe[DEFAULT_TIMESTAMP_COLUMN],
        timestamp_format=PROCESSED_TIMESTAMP_FORMAT,
    )
    parse_failures = int(parsed_timestamps.isna().sum())
    if parse_failures:
        raise ValueError(f"The input contains {parse_failures} timestamp parse failures.")
    dataframe[DEFAULT_TIMESTAMP_COLUMN] = parsed_timestamps

    for column in dataframe.columns:
        if column == DEFAULT_TIMESTAMP_COLUMN:
            continue
        original_missing = dataframe[column].isna()
        converted = pd.to_numeric(dataframe[column], errors="coerce")
        invalid_count = int((converted.isna() & ~original_missing).sum())
        if invalid_count:
            raise ValueError(
                f"Column {column} contains {invalid_count} unexpected non-numeric values."
            )
        dataframe[column] = converted
    return dataframe


def _calculate_rolling_feature(
    values: pd.Series, spec: RollingFeatureSpec
) -> pd.Series:
    """Calculate one full-window rolling statistic with no future access."""
    rolling = values.rolling(
        window=spec.window_samples,
        min_periods=spec.window_samples,
        center=False,
    )
    if spec.statistic == "mean":
        return rolling.mean()
    if spec.statistic == "std":
        return rolling.std(ddof=ROLLING_STANDARD_DEVIATION_DDOF)
    if spec.statistic == "min":
        return rolling.min()
    if spec.statistic == "max":
        return rolling.max()
    raise ValueError(f"Unsupported rolling statistic: {spec.statistic}")


def add_rolling_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Append exactly the approved trailing rolling features to a copy."""
    _require_columns(dataframe, INPUT_COLUMNS)
    collisions = [name for name in ROLLING_FEATURES if name in dataframe.columns]
    if collisions:
        raise ValueError("Rolling feature columns already exist: " + ", ".join(collisions))

    result = dataframe.copy(deep=True)
    for spec in ROLLING_FEATURE_SPECS:
        result[spec.name] = _calculate_rolling_feature(
            result[spec.source_column], spec
        )

    expected_columns = [*dataframe.columns, *ROLLING_FEATURES]
    if result.columns.tolist() != expected_columns:
        raise RuntimeError("Unexpected rolling feature columns were created.")
    return result


def _format_rolling_value(value: object) -> str:
    """Format a new float for portable CSV round-trip parsing."""
    if pd.isna(value):
        return ""
    return repr(float(value))


def _atomic_write_csv(
    dataframe: pd.DataFrame,
    destination: Path,
    *,
    source_csv: Path,
) -> None:
    """Append features while copying every pre-existing CSV field verbatim."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv.tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.writer(temporary_file, lineterminator="\n")
            with source_csv.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.reader(source)
                try:
                    source_header = next(reader)
                except StopIteration as error:
                    raise ValueError("The Day 8 input CSV is empty.") from error
                if source_header != list(INPUT_COLUMNS):
                    raise ValueError("The Day 8 CSV header changed during processing.")
                writer.writerow([*source_header, *ROLLING_FEATURES])

                source_row_count = 0
                for source_row, feature_values in zip(
                    reader,
                    dataframe.loc[:, list(ROLLING_FEATURES)].itertuples(
                        index=False, name=None
                    ),
                    strict=True,
                ):
                    writer.writerow(
                        [
                            *source_row,
                            *(
                                _format_rolling_value(value)
                                for value in feature_values
                            ),
                        ]
                    )
                    source_row_count += 1
                if source_row_count != len(dataframe):
                    raise ValueError(
                        "The source CSV row count changed during feature writing."
                    )
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(report: dict[str, object], destination: Path) -> None:
    """Atomically write a strict JSON report."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".json.tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                report,
                temporary_file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            temporary_file.write("\n")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_power_validation_figure(
    dataframe: pd.DataFrame, destination: str | Path
) -> None:
    """Plot generated power against its 30-minute trailing mean."""
    required = (
        DEFAULT_TIMESTAMP_COLUMN,
        "Power_Generated",
        "power_generated_roll_mean_30m",
    )
    _require_columns(dataframe, required)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.plot(
        dataframe[DEFAULT_TIMESTAMP_COLUMN],
        dataframe["Power_Generated"],
        color="#427AA1",
        linewidth=1.1,
        alpha=0.65,
        label="Power_Generated",
    )
    axis.plot(
        dataframe[DEFAULT_TIMESTAMP_COLUMN],
        dataframe["power_generated_roll_mean_30m"],
        color="#D1495B",
        linewidth=2.0,
        label="30-minute trailing rolling mean",
    )
    axis.set_title("Generated Power and 30-Minute Trailing Rolling Mean")
    axis.set_xlabel("Timestamp (timezone not specified by source)")
    axis.set_ylabel("Power Generated (W)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    figure.tight_layout()

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png", dir=output.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        figure.savefig(temporary_path, dpi=160, bbox_inches="tight")
        temporary_path.replace(output)
    finally:
        plt.close(figure)
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _expected_rolling_values(source: pd.DataFrame) -> dict[str, pd.Series]:
    """Recompute every approved rolling feature from the unmodified input."""
    return {
        spec.name: _calculate_rolling_feature(source[spec.source_column], spec)
        for spec in ROLLING_FEATURE_SPECS
    }


def validate_rolling_dataset(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    expected_input_sha256: str | None = None,
) -> dict[str, object]:
    """Validate saved rolling features against independently recomputed values."""
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    source = pd.read_csv(input_path)
    output = pd.read_csv(output_path)
    input_hash = sha256_file(input_path)

    rows_preserved = len(source) == len(output)
    original_columns_present = all(column in output.columns for column in source.columns)
    rolling_columns_present = all(column in output.columns for column in ROLLING_FEATURES)
    exact_layout = output.columns.tolist() == [*source.columns, *ROLLING_FEATURES]

    original_values_preserved = False
    original_value_mismatch: str | None = None
    if rows_preserved and original_columns_present:
        try:
            pd.testing.assert_frame_equal(
                source,
                output.loc[:, source.columns],
                check_dtype=True,
                check_exact=True,
                check_names=True,
            )
            original_values_preserved = True
        except AssertionError as error:
            original_value_mismatch = str(error)

    source_numeric = source.copy()
    for column in source_numeric.columns:
        if column != DEFAULT_TIMESTAMP_COLUMN:
            source_numeric[column] = pd.to_numeric(
                source_numeric[column], errors="coerce"
            )
    expected = _expected_rolling_values(source_numeric)
    formula_checks: dict[str, bool] = {}
    nan_checks: dict[str, bool] = {}
    prefix_checks: dict[str, bool] = {}
    for spec in ROLLING_FEATURE_SPECS:
        actual_values = output[spec.name] if spec.name in output else pd.Series(dtype=float)
        expected_values = expected[spec.name]
        formula_checks[spec.name] = bool(
            len(actual_values) == len(expected_values)
            and np.allclose(
                actual_values,
                expected_values,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
        )
        expected_nan_count = spec.window_samples - 1
        nan_checks[spec.name] = bool(
            int(actual_values.isna().sum()) == expected_nan_count
        )
        prefix_checks[spec.name] = bool(
            len(actual_values) >= spec.window_samples
            and actual_values.iloc[:expected_nan_count].isna().all()
            and actual_values.iloc[expected_nan_count:].notna().all()
        )

    min_mean_max_checks: dict[str, bool] = {}
    for prefix in ("power_generated", "solar_radiation"):
        for minutes in WINDOW_SAMPLES:
            minimum = output[f"{prefix}_roll_min_{minutes}m"]
            mean = output[f"{prefix}_roll_mean_{minutes}m"]
            maximum = output[f"{prefix}_roll_max_{minutes}m"]
            valid = minimum.notna() & mean.notna() & maximum.notna()
            min_mean_max_checks[f"{prefix}_{minutes}m"] = bool(
                ((minimum[valid] <= mean[valid]) & (mean[valid] <= maximum[valid])).all()
            )

    checks = {
        "rows_preserved": rows_preserved,
        "all_previous_columns_present": original_columns_present,
        "all_previous_values_preserved": original_values_preserved,
        "all_rolling_columns_present": rolling_columns_present,
        "exact_column_layout": exact_layout,
        "all_rolling_formulas_match_trailing_recalculation": all(
            formula_checks.values()
        ),
        "all_nan_counts_match_full_windows": all(nan_checks.values()),
        "all_nans_are_initial_prefix_only": all(prefix_checks.values()),
        "all_primary_min_mean_max_relationships_valid": all(
            min_mean_max_checks.values()
        ),
        "input_hash_matches_pre_write_hash": (
            True if expected_input_sha256 is None else input_hash == expected_input_sha256
        ),
        "trailing_only_configuration": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "source_rows": int(len(source)),
        "output_rows": int(len(output)),
        "source_columns": int(len(source.columns)),
        "output_columns": int(len(output.columns)),
        "formula_checks_by_feature": formula_checks,
        "nan_count_checks_by_feature": nan_checks,
        "initial_nan_prefix_checks_by_feature": prefix_checks,
        "min_mean_max_checks": min_mean_max_checks,
        "source_sha256": input_hash,
        "original_value_mismatch": original_value_mismatch,
    }


def _feature_statistics(dataframe: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Return missing counts, ranges, and descriptive statistics."""
    result: dict[str, dict[str, object]] = {}
    for feature in ROLLING_FEATURES:
        values = dataframe[feature]
        result[feature] = {
            "count": int(values.notna().sum()),
            "nan_count": int(values.isna().sum()),
            "minimum": _json_number(values.min()),
            "maximum": _json_number(values.max()),
            "mean": _json_number(values.mean()),
            "median": _json_number(values.median()),
            "standard_deviation": _json_number(values.std(ddof=1)),
        }
    return result


def _nan_counts_by_window(dataframe: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Aggregate initial missing-value counts by window duration."""
    result: dict[str, dict[str, object]] = {}
    for minutes, samples in WINDOW_SAMPLES.items():
        names = [
            spec.name
            for spec in ROLLING_FEATURE_SPECS
            if spec.window_minutes == minutes
        ]
        counts = {name: int(dataframe[name].isna().sum()) for name in names}
        result[f"{minutes}m"] = {
            "window_samples": samples,
            "feature_count": len(names),
            "expected_initial_nan_count_per_feature": samples - 1,
            "actual_nan_counts": counts,
            "total_nan_values": int(sum(counts.values())),
        }
    return result


def _protected_paths_for_run(input_path: Path) -> list[Path]:
    """Select source files that must remain byte-for-byte unchanged."""
    paths = [input_path]
    if input_path.resolve() == DEFAULT_INPUT_PATH.resolve():
        paths = [DEFAULT_RAW_PATH, DEFAULT_CLEANED_PATH, input_path]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Protected source file(s) missing: " + ", ".join(missing))
    return paths


def engineer_rolling_feature_file(
    input_csv: str | Path = DEFAULT_INPUT_PATH,
    *,
    output_csv: str | Path = DEFAULT_OUTPUT_PATH,
    summary_json: str | Path = DEFAULT_SUMMARY_PATH,
    figure_path: str | Path = DEFAULT_FIGURE_PATH,
) -> dict[str, object]:
    """Create, validate, plot, and report the Day 9 rolling dataset."""
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    summary_path = Path(summary_json)
    figure_output = Path(figure_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from the Day 8 input path.")

    protected_paths = _protected_paths_for_run(input_path)
    hashes_before = {path: sha256_file(path) for path in protected_paths}
    basic = load_basic_feature_data(input_path)
    rolling = add_rolling_features(basic)
    _atomic_write_csv(rolling, output_path, source_csv=input_path)
    save_power_validation_figure(rolling, figure_output)
    hashes_after = {path: sha256_file(path) for path in protected_paths}

    validation = validate_rolling_dataset(
        input_path,
        output_path,
        expected_input_sha256=hashes_before[input_path],
    )
    protected_hashes: dict[str, dict[str, object]] = {}
    for path in protected_paths:
        key = _relative_or_absolute(path)
        protected_hashes[key] = {
            "sha256_before": hashes_before[path],
            "sha256_after": hashes_after[path],
            "unchanged": hashes_before[path] == hashes_after[path],
        }
    protected_files_unchanged = all(
        evidence["unchanged"] for evidence in protected_hashes.values()
    )
    validation["checks"]["all_protected_source_files_unchanged"] = (
        protected_files_unchanged
    )
    validation["passed"] = bool(
        validation["passed"] and protected_files_unchanged
    )

    saved = pd.read_csv(output_path)
    report: dict[str, object] = {
        "feature_stage": "Day 9 trailing rolling-window feature engineering",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": _relative_or_absolute(input_path),
        "output_file": _relative_or_absolute(output_path),
        "validation_figure": _relative_or_absolute(figure_output),
        "input_dataset_shape": {
            "rows": int(len(basic)),
            "columns": int(len(basic.columns)),
        },
        "output_dataset_shape": {
            "rows": int(len(saved)),
            "columns": int(len(saved.columns)),
        },
        "rolling_feature_count": int(len(ROLLING_FEATURES)),
        "rolling_feature_names": list(ROLLING_FEATURES),
        "rolling_feature_specs": [
            asdict(spec) | {"center": False, "minimum_periods": spec.window_samples}
            for spec in ROLLING_FEATURE_SPECS
        ],
        "window_definitions": {
            f"{minutes}m": {
                "minutes": minutes,
                "samples": samples,
                "sampling_interval_minutes": 2,
                "full_window_required": True,
                "centered": False,
            }
            for minutes, samples in WINDOW_SAMPLES.items()
        },
        "window_direction": (
            "Trailing only: each value uses the current row and previous rows; "
            "future rows are never used."
        ),
        "rolling_standard_deviation_ddof": ROLLING_STANDARD_DEVIATION_DDOF,
        "initial_nan_policy": (
            "Retained in place; no backfill, future fill, or row removal."
        ),
        "nan_counts_by_window": _nan_counts_by_window(saved),
        "rolling_feature_statistics": _feature_statistics(saved),
        "protected_source_files": protected_hashes,
        "validation": validation,
        "excluded_work": [
            "rate-of-change features",
            "percentage-change features",
            "synthetic anomalies",
            "anomaly-detection models",
            "computer vision",
            "video processing",
            "dashboard",
        ],
        "timezone": TIMEZONE_NOTE,
    }
    _atomic_write_json(report, summary_path)

    if not validation["passed"]:
        raise RuntimeError(
            "Rolling-feature validation failed. See "
            + _relative_or_absolute(summary_path)
        )
    return report


def main() -> int:
    """Run rolling feature engineering from the command line."""
    arguments = parse_args()
    try:
        report = engineer_rolling_feature_file(
            arguments.input_csv,
            output_csv=arguments.output,
            summary_json=arguments.summary,
            figure_path=arguments.figure,
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError, pd.errors.ParserError) as error:
        print(f"Error: {error}")
        return 1

    shape = report["output_dataset_shape"]
    print(f"Rolling feature dataset: {report['output_file']}")
    print(f"Shape: {shape['rows']} rows x {shape['columns']} columns")
    print(f"Rolling features: {report['rolling_feature_count']}")
    for window, evidence in report["nan_counts_by_window"].items():
        print(
            f"{window} window: {evidence['expected_initial_nan_count_per_feature']} "
            "initial NaNs per feature"
        )
    print(f"Validation passed: {report['validation']['passed']}")
    print("No future observations, change features, or models were used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
