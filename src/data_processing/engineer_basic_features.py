"""Create the safe, interpretable Day 8 operational features.

The pipeline preserves every cleaned input row and column. It intentionally
does not create rolling, rate-of-change, or wind-direction features and does
not modify negative solar-radiation or wind-direction measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
DEFAULT_OUTPUT_PATH = Path("data/processed/operational_features_basic.csv")
DEFAULT_SUMMARY_PATH = Path("outputs/basic_feature_summary.json")
PROCESSED_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMEZONE_NOTE = "Not specified by dataset source; timestamps kept timezone-naive."
LOW_LIGHT_THRESHOLD = 5.0
ORIGINAL_COLUMNS = (DEFAULT_TIMESTAMP_COLUMN, *SELECTED_NUMERIC_COLUMNS)
RTD_COLUMNS = ("RTD_1", "RTD_2", "RTD_3", "RTD_4", "RTD_5")
CYCLICAL_FEATURES = (
    "hour_sin",
    "hour_cos",
    "minute_of_day_sin",
    "minute_of_day_cos",
)
ENGINEERED_FEATURES = (
    "hour",
    "minute",
    "minute_of_day",
    "elapsed_minutes_from_start",
    *CYCLICAL_FEATURES,
    "rtd_mean",
    "rtd_std",
    "rtd_min",
    "rtd_max",
    "rtd_range",
    "electrical_power_product",
    "power_consistency_residual",
    "power_consistency_abs_error",
    "low_light_context",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create the Day 8 basic operational features."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Cleaned CSV path (default: {DEFAULT_INPUT_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Feature CSV path (default: {DEFAULT_OUTPUT_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Summary JSON path (default: {DEFAULT_SUMMARY_PATH.as_posix()}).",
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
    """Convert a scalar to a strict JSON-compatible finite float."""
    if value is None or pd.isna(value):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _require_columns(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Raise a clear error when a verified input column is absent."""
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError("Required columns are missing: " + ", ".join(missing))


def load_cleaned_operational_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the Day 6 cleaned CSV through the existing conservative loader."""
    dataframe = load_operational_csv(
        csv_path,
        timestamp_column=DEFAULT_TIMESTAMP_COLUMN,
        timestamp_format=PROCESSED_TIMESTAMP_FORMAT,
        numeric_columns=SELECTED_NUMERIC_COLUMNS,
        sort_chronologically=False,
        drop_duplicate_rows=False,
    )
    _require_columns(dataframe, ORIGINAL_COLUMNS)
    if dataframe.columns.tolist() != list(ORIGINAL_COLUMNS):
        raise ValueError(
            "The cleaned input must match the verified 14-column schema and order."
        )
    parse_failures = int(dataframe[DEFAULT_TIMESTAMP_COLUMN].isna().sum())
    if parse_failures:
        raise ValueError(
            f"The cleaned input contains {parse_failures} timestamp parse failures."
        )
    return dataframe


def add_basic_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy containing only the approved Day 8 engineered features."""
    _require_columns(dataframe, ORIGINAL_COLUMNS)
    collisions = [column for column in ENGINEERED_FEATURES if column in dataframe.columns]
    if collisions:
        raise ValueError(
            "Engineered feature columns already exist: " + ", ".join(collisions)
        )

    result = dataframe.copy(deep=True)
    timestamps = result[DEFAULT_TIMESTAMP_COLUMN]
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(
            timestamps, format=PROCESSED_TIMESTAMP_FORMAT, errors="coerce"
        )
    if timestamps.isna().any():
        raise ValueError("Cannot engineer time features from unparseable timestamps.")
    if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError(
            "Timezone-aware timestamps are not expected; the source timezone is unknown."
        )
    result[DEFAULT_TIMESTAMP_COLUMN] = timestamps

    result["hour"] = timestamps.dt.hour.astype("int16")
    result["minute"] = timestamps.dt.minute.astype("int16")
    result["minute_of_day"] = (result["hour"] * 60 + result["minute"]).astype(
        "int16"
    )
    result["elapsed_minutes_from_start"] = (
        timestamps - timestamps.min()
    ).dt.total_seconds() / 60.0

    result["hour_sin"] = np.sin(2.0 * np.pi * result["hour"] / 24.0)
    result["hour_cos"] = np.cos(2.0 * np.pi * result["hour"] / 24.0)
    result["minute_of_day_sin"] = np.sin(
        2.0 * np.pi * result["minute_of_day"] / 1440.0
    )
    result["minute_of_day_cos"] = np.cos(
        2.0 * np.pi * result["minute_of_day"] / 1440.0
    )

    rtd_values = result.loc[:, list(RTD_COLUMNS)]
    result["rtd_mean"] = rtd_values.mean(axis=1)
    # Population standard deviation summarizes the five channels at each time.
    result["rtd_std"] = rtd_values.std(axis=1, ddof=0)
    result["rtd_min"] = rtd_values.min(axis=1)
    result["rtd_max"] = rtd_values.max(axis=1)
    result["rtd_range"] = result["rtd_max"] - result["rtd_min"]

    result["electrical_power_product"] = (
        result["Array_Voltage"] * result["Array_Current"]
    )
    result["power_consistency_residual"] = (
        result["Power_Generated"] - result["electrical_power_product"]
    )
    result["power_consistency_abs_error"] = result[
        "power_consistency_residual"
    ].abs()

    # This project threshold is contextual only; original radiation values stay
    # untouched, including all negative measurements.
    result["low_light_context"] = (
        result["Solar_Radiation"] <= LOW_LIGHT_THRESHOLD
    ).astype("int8")

    expected_columns = [*dataframe.columns, *ENGINEERED_FEATURES]
    if result.columns.tolist() != expected_columns:
        raise RuntimeError("Unexpected feature columns were created.")
    return result


def _atomic_write_csv(dataframe: pd.DataFrame, destination: Path) -> None:
    """Atomically write the feature dataset."""
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
            dataframe.to_csv(
                temporary_file,
                index=False,
                date_format=PROCESSED_TIMESTAMP_FORMAT,
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


def _feature_ranges(dataframe: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Return minimum and maximum values for each engineered feature."""
    return {
        feature: {
            "minimum": _json_number(dataframe[feature].min()),
            "maximum": _json_number(dataframe[feature].max()),
        }
        for feature in ENGINEERED_FEATURES
    }


def _residual_statistics(dataframe: pd.DataFrame) -> dict[str, float | int | None]:
    """Summarize the signed and absolute electrical consistency errors."""
    residual = dataframe["power_consistency_residual"]
    absolute = dataframe["power_consistency_abs_error"]
    return {
        "count": int(residual.notna().sum()),
        "minimum": _json_number(residual.min()),
        "maximum": _json_number(residual.max()),
        "mean": _json_number(residual.mean()),
        "median": _json_number(residual.median()),
        "standard_deviation": _json_number(residual.std(ddof=1)),
        "absolute_error_minimum": _json_number(absolute.min()),
        "absolute_error_maximum": _json_number(absolute.max()),
        "absolute_error_mean": _json_number(absolute.mean()),
        "absolute_error_median": _json_number(absolute.median()),
    }


def validate_feature_dataset(
    input_csv: str | Path,
    feature_csv: str | Path,
    *,
    expected_input_sha256: str | None = None,
) -> dict[str, object]:
    """Validate the saved feature CSV against the untouched cleaned input."""
    input_path = Path(input_csv)
    feature_path = Path(feature_csv)
    source = pd.read_csv(input_path)
    featured = pd.read_csv(feature_path)
    input_hash = sha256_file(input_path)

    original_columns_present = all(
        column in featured.columns for column in source.columns
    )
    engineered_columns_present = all(
        column in featured.columns for column in ENGINEERED_FEATURES
    )
    exact_column_layout = featured.columns.tolist() == [
        *source.columns,
        *ENGINEERED_FEATURES,
    ]
    rows_preserved = len(source) == len(featured)

    original_values_preserved = False
    original_value_mismatch: str | None = None
    if original_columns_present and rows_preserved:
        try:
            pd.testing.assert_frame_equal(
                source,
                featured.loc[:, source.columns],
                check_dtype=True,
                check_exact=True,
                check_names=True,
            )
            original_values_preserved = True
        except AssertionError as error:
            original_value_mismatch = str(error)

    cyclical_ranges_valid = engineered_columns_present and all(
        bool(featured[column].between(-1.0 - 1e-12, 1.0 + 1e-12).all())
        for column in CYCLICAL_FEATURES
    )
    rtd_order_valid = engineered_columns_present and bool(
        (
            (featured["rtd_min"] <= featured["rtd_mean"])
            & (featured["rtd_mean"] <= featured["rtd_max"])
        ).all()
    )
    rtd_range_valid = engineered_columns_present and bool(
        (featured["rtd_range"] >= -1e-12).all()
    )

    if engineered_columns_present:
        expected_product = featured["Array_Voltage"] * featured["Array_Current"]
        expected_residual = featured["Power_Generated"] - expected_product
        electrical_calculations_valid = bool(
            np.allclose(
                featured["electrical_power_product"],
                expected_product,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
            and np.allclose(
                featured["power_consistency_residual"],
                expected_residual,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
            and np.allclose(
                featured["power_consistency_abs_error"],
                expected_residual.abs(),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
        )
        expected_low_light = (featured["Solar_Radiation"] <= LOW_LIGHT_THRESHOLD).astype(
            "int8"
        )
        low_light_flag_valid = bool(
            featured["low_light_context"].equals(expected_low_light.astype("int64"))
        )
    else:
        electrical_calculations_valid = False
        low_light_flag_valid = False

    original_missing_values = int(source.isna().sum().sum())
    engineered_missing_values = (
        int(featured.loc[:, list(ENGINEERED_FEATURES)].isna().sum().sum())
        if engineered_columns_present
        else None
    )
    no_unexpected_engineered_missing_values = engineered_missing_values == 0
    input_matches_expected_hash = (
        True if expected_input_sha256 is None else input_hash == expected_input_sha256
    )
    prohibited_feature_names = [
        column
        for column in featured.columns
        if "rolling" in column.lower()
        or "rate_of_change" in column.lower()
        or column in {"wind_direction_sin", "wind_direction_cos"}
    ]

    checks = {
        "input_hash_matches_pre_write_hash": input_matches_expected_hash,
        "rows_preserved": rows_preserved,
        "all_original_columns_present": original_columns_present,
        "all_original_values_preserved": original_values_preserved,
        "all_engineered_columns_present": engineered_columns_present,
        "exact_column_layout": exact_column_layout,
        "no_unexpected_engineered_missing_values": no_unexpected_engineered_missing_values,
        "cyclical_ranges_within_minus_one_and_one": cyclical_ranges_valid,
        "rtd_minimum_mean_maximum_order_valid": rtd_order_valid,
        "rtd_range_nonnegative": rtd_range_valid,
        "electrical_consistency_calculations_valid": electrical_calculations_valid,
        "low_light_flag_matches_threshold": low_light_flag_valid,
        "no_prohibited_day_8_features": not prohibited_feature_names,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "source_rows": int(len(source)),
        "feature_rows": int(len(featured)),
        "source_columns": int(len(source.columns)),
        "feature_columns": int(len(featured.columns)),
        "source_missing_values": original_missing_values,
        "engineered_missing_values": engineered_missing_values,
        "source_sha256": input_hash,
        "original_value_mismatch": original_value_mismatch,
        "prohibited_feature_names": prohibited_feature_names,
    }


def engineer_basic_feature_file(
    input_csv: str | Path = DEFAULT_INPUT_PATH,
    *,
    output_csv: str | Path = DEFAULT_OUTPUT_PATH,
    summary_json: str | Path = DEFAULT_SUMMARY_PATH,
) -> dict[str, object]:
    """Create, validate, and report the Day 8 feature dataset."""
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    summary_path = Path(summary_json)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from the cleaned input path.")

    input_hash_before = sha256_file(input_path)
    cleaned = load_cleaned_operational_data(input_path)
    featured = add_basic_features(cleaned)
    _atomic_write_csv(featured, output_path)
    input_hash_after = sha256_file(input_path)

    validation = validate_feature_dataset(
        input_path,
        output_path,
        expected_input_sha256=input_hash_before,
    )
    input_unchanged = input_hash_before == input_hash_after
    validation["checks"]["cleaned_input_hash_unchanged"] = input_unchanged
    validation["passed"] = bool(validation["passed"] and input_unchanged)

    saved = pd.read_csv(output_path)
    low_light_count = int(saved["low_light_context"].sum())
    report: dict[str, object] = {
        "feature_stage": "Day 8 initial feature engineering",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": _relative_or_absolute(input_path),
        "output_file": _relative_or_absolute(output_path),
        "input_sha256_before": input_hash_before,
        "input_sha256_after": input_hash_after,
        "input_file_unchanged": input_unchanged,
        "original_column_count": int(len(ORIGINAL_COLUMNS)),
        "original_measurement_feature_count_excluding_timestamp": int(
            len(SELECTED_NUMERIC_COLUMNS)
        ),
        "engineered_feature_count": int(len(ENGINEERED_FEATURES)),
        "final_dataset_shape": {
            "rows": int(len(saved)),
            "columns": int(len(saved.columns)),
        },
        "original_column_names": list(ORIGINAL_COLUMNS),
        "engineered_feature_names": list(ENGINEERED_FEATURES),
        "engineered_feature_ranges": _feature_ranges(saved),
        "low_light_context": {
            "formula": f"1 when Solar_Radiation <= {LOW_LIGHT_THRESHOLD:g}; otherwise 0",
            "threshold": LOW_LIGHT_THRESHOLD,
            "threshold_status": (
                "Exploratory project threshold, not a manufacturer specification."
            ),
            "observation_count": low_light_count,
            "percentage_of_rows": _json_number(low_light_count / len(saved) * 100),
            "negative_solar_radiation_values_modified": False,
        },
        "power_consistency_residual_statistics": _residual_statistics(saved),
        "power_generated_policy": {
            "strategy_selected": False,
            "possible_unsupervised_role": "May be used as an input variable.",
            "possible_prediction_role": "May instead be a prediction target.",
            "electrical_power_product_warning": (
                "Do not automatically use it alongside Power_Generated because "
                "it recreates that measurement almost exactly."
            ),
            "residual_role": "Primarily a diagnostic consistency signal.",
        },
        "timezone": TIMEZONE_NOTE,
        "wind_direction_policy": (
            "Original values retained unchanged; no modulo correction or circular "
            "wind features were created."
        ),
        "excluded_feature_categories": [
            "weekday, month, and season",
            "rolling-window statistics",
            "rate-of-change and first-difference features",
            "wind-direction cyclical features",
        ],
        "validation": validation,
    }
    _atomic_write_json(report, summary_path)

    if not validation["passed"]:
        raise RuntimeError(
            "Feature validation failed. See " + _relative_or_absolute(summary_path)
        )
    return report


def main() -> int:
    """Run feature engineering from the command line."""
    arguments = parse_args()
    try:
        report = engineer_basic_feature_file(
            arguments.input_csv,
            output_csv=arguments.output,
            summary_json=arguments.summary,
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError, pd.errors.ParserError) as error:
        print(f"Error: {error}")
        return 1

    shape = report["final_dataset_shape"]
    low_light = report["low_light_context"]
    print(f"Feature dataset: {report['output_file']}")
    print(f"Shape: {shape['rows']} rows x {shape['columns']} columns")
    print(f"Engineered features: {report['engineered_feature_count']}")
    print(
        "Low-light context: "
        f"{low_light['observation_count']} ({low_light['percentage_of_rows']:.3f}%)"
    )
    print(f"Validation passed: {report['validation']['passed']}")
    print("No rolling, rate-of-change, or wind-direction features were created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
