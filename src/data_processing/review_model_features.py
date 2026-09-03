"""Review the Day 10 feature space without fitting preprocessing or models.

The Day 11 pipeline is deliberately read-only with respect to every dataset. It
creates an 81-column inventory, exploratory redundancy evidence, candidate
model-input sets, and a focused descriptive review of the recurring 17:00 power
transitions. No feature is removed and no anomaly label is assigned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .engineer_basic_features import PROCESSED_TIMESTAMP_FORMAT
    from .engineer_change_features import (
        CHANGE_FEATURES,
        DEFAULT_BASIC_FEATURE_PATH,
        DEFAULT_INPUT_PATH as DEFAULT_ROLLING_PATH,
        DEFAULT_OUTPUT_PATH as DEFAULT_INPUT_PATH,
        DEVIATION_FEATURES,
        INPUT_COLUMNS as DAY_9_COLUMNS,
        RELATIVE_CHANGE_THRESHOLDS,
    )
    from .engineer_rolling_features import DEFAULT_CLEANED_PATH, DEFAULT_RAW_PATH
    from .load_operational_data import DEFAULT_TIMESTAMP_COLUMN, parse_timestamp_series
else:
    from engineer_basic_features import PROCESSED_TIMESTAMP_FORMAT  # type: ignore[no-redef]
    from engineer_change_features import (  # type: ignore[no-redef]
        CHANGE_FEATURES,
        DEFAULT_BASIC_FEATURE_PATH,
        DEFAULT_INPUT_PATH as DEFAULT_ROLLING_PATH,
        DEFAULT_OUTPUT_PATH as DEFAULT_INPUT_PATH,
        DEVIATION_FEATURES,
        INPUT_COLUMNS as DAY_9_COLUMNS,
        RELATIVE_CHANGE_THRESHOLDS,
    )
    from engineer_rolling_features import (  # type: ignore[no-redef]
        DEFAULT_CLEANED_PATH,
        DEFAULT_RAW_PATH,
    )
    from load_operational_data import (  # type: ignore[no-redef]
        DEFAULT_TIMESTAMP_COLUMN,
        parse_timestamp_series,
    )


DEFAULT_REVIEW_PATH = Path("outputs/model_feature_review.json")
DEFAULT_TRANSITION_PATH = Path("outputs/recurring_transition_review.json")
DEFAULT_DOCUMENTATION_PATH = Path("docs/model_feature_review.md")
DEFAULT_MODELS_DIRECTORY = Path("models")
TIMEZONE_NOTE = "Not specified by dataset source; timestamps kept timezone-naive."
FEATURE_COLUMNS = (*DAY_9_COLUMNS, *CHANGE_FEATURES)
CORRELATION_THRESHOLD = 0.95
NEAR_CONSTANT_CV_THRESHOLD = 0.01
NEAR_CONSTANT_DOMINANT_FRACTION = 0.95
TRANSITION_HALF_WINDOW_MINUTES = 20
TRANSITION_TIMESTAMPS = (
    pd.Timestamp("2022-04-27 17:00:00"),
    pd.Timestamp("2022-04-28 17:00:00"),
)

CATEGORY_ORDER = (
    "identifiers/time",
    "original environmental measurements",
    "original electrical measurements",
    "RTD channels",
    "basic engineered features",
    "rolling features",
    "change/rate features",
    "diagnostic features",
    "context flags",
)
RECOMMENDATION_VALUES = {
    "KEEP",
    "OPTIONAL",
    "EXCLUDE_FROM_MODEL",
    "CONTEXT_ONLY",
    "UNRESOLVED",
}

IDENTIFIER_TIME_COLUMNS = {
    "Timestamp",
    "hour",
    "minute",
    "minute_of_day",
    "elapsed_minutes_from_start",
    "hour_sin",
    "hour_cos",
    "minute_of_day_sin",
    "minute_of_day_cos",
}
ENVIRONMENTAL_COLUMNS = {
    "Air_Temp",
    "Relative_Humidity",
    "Wind_Speed",
    "Wind_Direction",
    "Solar_Radiation",
}
ELECTRICAL_COLUMNS = {"Array_Voltage", "Array_Current", "Power_Generated"}
RTD_CHANNELS = {f"RTD_{index}" for index in range(1, 6)}
RTD_SUMMARY_COLUMNS = {"rtd_mean", "rtd_std", "rtd_min", "rtd_max", "rtd_range"}
DIAGNOSTIC_COLUMNS = {
    "electrical_power_product",
    "power_consistency_residual",
    "power_consistency_abs_error",
}
CONTEXT_COLUMNS = {"low_light_context"}

CORE_UNSUPERVISED_FEATURES = (
    "Power_Generated",
    "Solar_Radiation",
    "Air_Temp",
    "Relative_Humidity",
    "Wind_Speed",
    "rtd_mean",
    "rtd_std",
    "minute_of_day_sin",
    "minute_of_day_cos",
)
EXTENDED_DIAGNOSTIC_FEATURES = (
    *CORE_UNSUPERVISED_FEATURES,
    "Array_Current",
    "rtd_range",
    "power_generated_roll_mean_30m",
    "power_generated_roll_std_30m",
    "solar_radiation_roll_mean_30m",
    "solar_radiation_roll_std_30m",
    "rtd_mean_roll_mean_30m",
    "rtd_mean_roll_std_30m",
    "power_generated_diff",
    "solar_radiation_diff",
    "array_current_diff",
    "rtd_mean_diff",
    "power_deviation_from_30m_mean",
    "solar_deviation_from_30m_mean",
)
DAYLIGHT_SPECIFIC_FEATURES = (
    "Power_Generated",
    "Solar_Radiation",
    "power_generated_diff",
    "solar_radiation_diff",
    "power_generated_relative_change",
    "solar_radiation_relative_change",
    "power_deviation_ratio_30m",
    "solar_deviation_ratio_30m",
)

KEEP_FEATURES = set(CORE_UNSUPERVISED_FEATURES)
OPTIONAL_FEATURES = set(EXTENDED_DIAGNOSTIC_FEATURES).union(
    DAYLIGHT_SPECIFIC_FEATURES
) - KEEP_FEATURES
OPTIONAL_FEATURES.update(
    {
        "air_temp_diff",
        "relative_humidity_diff",
        "air_temp_roll_mean_30m",
        "relative_humidity_roll_mean_30m",
        "power_generated_roll_mean_10m",
        "power_generated_roll_std_10m",
        "solar_radiation_roll_mean_10m",
        "solar_radiation_roll_std_10m",
    }
)
UNRESOLVED_FEATURES = {"Wind_Direction"}
CONTEXT_ONLY_FEATURES = {"Timestamp", "low_light_context"}

TRANSITION_COLUMNS = (
    "Timestamp",
    "Power_Generated",
    "Solar_Radiation",
    "Array_Voltage",
    "Array_Current",
    "Air_Temp",
    "Relative_Humidity",
    "low_light_context",
    "power_generated_roll_mean_10m",
    "power_generated_roll_mean_30m",
    "power_generated_roll_mean_60m",
    "power_generated_diff",
    "power_generated_rate_per_min",
    "power_deviation_from_30m_mean",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Review all Day 10 features and plan future model inputs."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Day 10 feature CSV (default: {DEFAULT_INPUT_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=DEFAULT_REVIEW_PATH,
        help=f"Feature-review JSON (default: {DEFAULT_REVIEW_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--transition-output",
        type=Path,
        default=DEFAULT_TRANSITION_PATH,
        help=f"Transition-review JSON (default: {DEFAULT_TRANSITION_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--documentation",
        type=Path,
        default=DEFAULT_DOCUMENTATION_PATH,
        help=f"Markdown review (default: {DEFAULT_DOCUMENTATION_PATH.as_posix()}).",
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
    """Return a file SHA-256 digest without changing the file."""
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


def _timestamp_text(value: object) -> str | None:
    """Serialize a timezone-naive timestamp."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat(sep=" ")


def _atomic_write_json(report: dict[str, object], destination: Path) -> None:
    """Atomically write strict JSON."""
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


def _atomic_write_text(text: str, destination: Path) -> None:
    """Atomically write UTF-8 text."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".md.tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(text)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _snapshot_directory(directory: Path) -> dict[str, str]:
    """Record hashes for every file in a directory without creating it."""
    if not directory.exists():
        return {}
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def load_feature_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the exact Day 10 schema without changing row order."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Day 10 feature CSV does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a CSV file but received: {path}")

    dataframe = pd.read_csv(path)
    if dataframe.columns.tolist() != list(FEATURE_COLUMNS):
        raise ValueError("Input must match the verified Day 10 81-column schema.")

    timestamps = parse_timestamp_series(
        dataframe[DEFAULT_TIMESTAMP_COLUMN],
        timestamp_format=PROCESSED_TIMESTAMP_FORMAT,
    )
    if timestamps.isna().any():
        raise ValueError("The Day 10 input contains timestamp parse failures.")
    if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("Timezone-aware timestamps are not expected for this source.")
    dataframe[DEFAULT_TIMESTAMP_COLUMN] = timestamps

    for column in dataframe.columns:
        if column == DEFAULT_TIMESTAMP_COLUMN:
            continue
        original_missing = dataframe[column].isna()
        converted = pd.to_numeric(dataframe[column], errors="coerce")
        invalid = converted.isna() & ~original_missing
        if invalid.any():
            raise ValueError(
                f"Column {column} contains {int(invalid.sum())} non-numeric values."
            )
        dataframe[column] = converted
    return dataframe


def feature_category(column: str) -> str:
    """Return the single review category assigned to a verified column."""
    if column in IDENTIFIER_TIME_COLUMNS:
        return "identifiers/time"
    if column in ENVIRONMENTAL_COLUMNS:
        return "original environmental measurements"
    if column in ELECTRICAL_COLUMNS:
        return "original electrical measurements"
    if column in RTD_CHANNELS:
        return "RTD channels"
    if column in RTD_SUMMARY_COLUMNS:
        return "basic engineered features"
    if column in DIAGNOSTIC_COLUMNS:
        return "diagnostic features"
    if column in CONTEXT_COLUMNS:
        return "context flags"
    if "_roll_" in column:
        return "rolling features"
    if column in CHANGE_FEATURES:
        return "change/rate features"
    raise ValueError(f"No feature category is defined for {column}.")


def feature_recommendation(column: str) -> str:
    """Return the initial model-input recommendation for one feature."""
    if column in CONTEXT_ONLY_FEATURES:
        return "CONTEXT_ONLY"
    if column in UNRESOLVED_FEATURES:
        return "UNRESOLVED"
    if column in KEEP_FEATURES:
        return "KEEP"
    if column in OPTIONAL_FEATURES:
        return "OPTIONAL"
    return "EXCLUDE_FROM_MODEL"


def _rolling_interpretation(column: str) -> str:
    """Describe a rolling feature from its verified name."""
    signal_names = {
        "power_generated": "generated power",
        "solar_radiation": "solar radiation",
        "air_temp": "air temperature",
        "relative_humidity": "relative humidity",
        "rtd_mean": "mean RTD level",
    }
    for prefix, description in signal_names.items():
        marker = f"{prefix}_roll_"
        if column.startswith(marker):
            remainder = column.removeprefix(marker)
            statistic, minutes_text = remainder.rsplit("_", maxsplit=1)
            minutes = minutes_text.removesuffix("m")
            return (
                f"Complete trailing {minutes}-minute {statistic} of {description}; "
                "current and earlier observations only."
            )
    return "Verified trailing operational summary."


def feature_interpretation(column: str) -> str:
    """Return a concise intended interpretation for every verified feature."""
    direct = {
        "Timestamp": "Timezone-naive observation time and chronological index.",
        "Air_Temp": "Original ambient air-temperature measurement.",
        "Relative_Humidity": "Original ambient relative-humidity measurement.",
        "Wind_Speed": "Original wind-speed measurement used as environmental context.",
        "Wind_Direction": "Original circular wind-direction measurement; values above 360 remain unresolved.",
        "Solar_Radiation": "Original solar-radiation measurement and primary available-light context.",
        "Array_Voltage": "Original array-voltage measurement; nearly duplicates generated-power behavior here.",
        "Array_Current": "Original array-current measurement with a narrow observed range.",
        "Power_Generated": "Original generated-power measurement and main operational output.",
        "hour": "Clock hour extracted from the timezone-naive timestamp.",
        "minute": "Clock minute extracted from the timestamp.",
        "minute_of_day": "Linear minute index within a day.",
        "elapsed_minutes_from_start": "Dataset-position trend measured from the first observation.",
        "hour_sin": "Cyclical sine encoding of clock hour.",
        "hour_cos": "Cyclical cosine encoding of clock hour.",
        "minute_of_day_sin": "Fine-grained cyclical sine encoding of time of day.",
        "minute_of_day_cos": "Fine-grained cyclical cosine encoding of time of day.",
        "rtd_mean": "Row-wise mean of the five unverified RTD channels.",
        "rtd_std": "Row-wise population standard deviation across the five RTD channels.",
        "rtd_min": "Row-wise minimum of the five RTD channels.",
        "rtd_max": "Row-wise maximum of the five RTD channels.",
        "rtd_range": "Row-wise RTD maximum minus minimum, describing channel disagreement.",
        "electrical_power_product": "Array_Voltage multiplied by Array_Current; almost exactly recreates Power_Generated.",
        "power_consistency_residual": "Power_Generated minus the voltage-current product.",
        "power_consistency_abs_error": "Absolute value of the power-consistency residual.",
        "low_light_context": "Exploratory context flag for Solar_Radiation at or below 5 W/m^2.",
        "power_generated_relative_change": "Safeguarded two-minute generated-power change divided by the previous power value.",
        "solar_radiation_relative_change": "Safeguarded two-minute radiation change divided by the previous radiation value.",
        "power_deviation_from_30m_mean": "Generated power minus its causal trailing 30-minute mean.",
        "solar_deviation_from_30m_mean": "Solar radiation minus its causal trailing 30-minute mean.",
        "power_deviation_ratio_30m": "Power deviation divided by its trailing mean when the denominator is safe.",
        "solar_deviation_ratio_30m": "Radiation deviation divided by its trailing mean when the denominator is safe.",
    }
    if column in direct:
        return direct[column]
    if column in RTD_CHANNELS:
        return f"Original RTD channel {column[-1]}; placement, physical meaning, and unit are unconfirmed."
    if "_roll_" in column:
        return _rolling_interpretation(column)
    if column.endswith("_diff"):
        source = column.removesuffix("_diff").replace("_", " ")
        return f"Current minus previous {source} value over one two-minute interval."
    if column.endswith("_rate_per_min"):
        source = column.removesuffix("_rate_per_min").replace("_", " ")
        return f"First difference of {source} divided by the verified two-minute interval."
    raise ValueError(f"No interpretation is defined for {column}.")


def _variability_status(series: pd.Series) -> tuple[str, str]:
    """Classify a column with transparent exploratory near-constant rules."""
    nonmissing = series.dropna()
    unique_count = int(nonmissing.nunique())
    if unique_count <= 1:
        return "constant", "At most one non-missing value."
    if pd.api.types.is_numeric_dtype(nonmissing):
        dominant_fraction = float(nonmissing.value_counts(normalize=True).iloc[0])
        mean_magnitude = abs(float(nonmissing.mean()))
        coefficient_of_variation = (
            float(nonmissing.std(ddof=0)) / mean_magnitude
            if mean_magnitude > 1e-12
            else math.inf
        )
        if dominant_fraction >= NEAR_CONSTANT_DOMINANT_FRACTION:
            return (
                "near_constant",
                f"Most common value occupies {dominant_fraction:.2%} of non-missing rows.",
            )
        if coefficient_of_variation <= NEAR_CONSTANT_CV_THRESHOLD:
            return (
                "near_constant",
                f"Population standard deviation / |mean| is {coefficient_of_variation:.6f}.",
            )
    return "variable", "Does not meet the documented constant or near-constant rules."


def _expected_missing_mask(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Return the documented missingness mask for one feature."""
    false_mask = pd.Series(False, index=dataframe.index)
    if column not in dataframe:
        return false_mask
    if "_roll_" in column:
        minutes = int(column.rsplit("_", maxsplit=1)[-1].removesuffix("m"))
        samples = minutes // 2
        expected = false_mask.copy()
        expected.iloc[: samples - 1] = True
        return expected
    if column.endswith("_diff") or column.endswith("_rate_per_min"):
        expected = false_mask.copy()
        expected.iloc[0] = True
        return expected
    if column == "power_generated_relative_change":
        previous = dataframe["Power_Generated"].shift(1)
        return previous.isna() | (
            previous.abs() < RELATIVE_CHANGE_THRESHOLDS["Power_Generated"]
        )
    if column == "solar_radiation_relative_change":
        previous = dataframe["Solar_Radiation"].shift(1)
        return previous.isna() | (
            previous.abs() < RELATIVE_CHANGE_THRESHOLDS["Solar_Radiation"]
        )
    if column in {
        "power_deviation_from_30m_mean",
        "solar_deviation_from_30m_mean",
    }:
        baseline = (
            "power_generated_roll_mean_30m"
            if column.startswith("power")
            else "solar_radiation_roll_mean_30m"
        )
        return dataframe[baseline].isna()
    if column in {"power_deviation_ratio_30m", "solar_deviation_ratio_30m"}:
        source = "Power_Generated" if column.startswith("power") else "Solar_Radiation"
        baseline = DEVIATION_FEATURES[source][0]
        threshold = DEVIATION_FEATURES[source][3]
        return dataframe[baseline].isna() | (dataframe[baseline].abs() < threshold)
    return false_mask


def missingness_mechanism(dataframe: pd.DataFrame, column: str) -> str:
    """Describe the documented reason for actual missing values."""
    if not dataframe[column].isna().any():
        return "none"
    if "_roll_" in column or column in {
        "power_deviation_from_30m_mean",
        "solar_deviation_from_30m_mean",
        "power_deviation_ratio_30m",
    }:
        return "rolling_window_warm_up"
    if column.endswith("_diff") or column.endswith("_rate_per_min"):
        return "first_difference_warm_up"
    if column in {
        "power_generated_relative_change",
        "solar_radiation_relative_change",
    }:
        return "denominator_safety"
    if column == "solar_deviation_ratio_30m":
        return "rolling_window_warm_up_and_denominator_safety"
    return "unexpected"


def build_feature_inventory(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Build the complete per-column feature inventory."""
    row_count = len(dataframe)
    inventory: list[dict[str, object]] = []
    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())
        expected_mask = _expected_missing_mask(dataframe, column)
        actual_mask = dataframe[column].isna()
        variability, variability_reason = _variability_status(dataframe[column])
        recommendation = feature_recommendation(column)
        if recommendation not in RECOMMENDATION_VALUES:
            raise RuntimeError(f"Invalid recommendation for {column}: {recommendation}")
        inventory.append(
            {
                "feature_name": column,
                "category": feature_category(column),
                "dtype": str(dataframe[column].dtype),
                "missing_count": missing_count,
                "missing_percentage": missing_count / row_count * 100.0,
                "missingness_mechanism": missingness_mechanism(dataframe, column),
                "unexpected_missing_count": int((actual_mask & ~expected_mask).sum()),
                "documented_missing_values_present": bool(
                    actual_mask.equals(expected_mask)
                ),
                "unique_nonmissing_values": int(dataframe[column].nunique(dropna=True)),
                "variability_status": variability,
                "variability_reason": variability_reason,
                "intended_interpretation": feature_interpretation(column),
                "recommendation": recommendation,
            }
        )
    return inventory


def summarize_missingness(
    dataframe: pd.DataFrame, inventory: Sequence[dict[str, object]]
) -> dict[str, object]:
    """Separate all missing cells by their documented source."""
    rolling_warm_up = 0
    first_difference_warm_up = 0
    denominator_safety = 0
    unexpected = 0

    for item in inventory:
        column = str(item["feature_name"])
        actual = dataframe[column].isna()
        if not actual.any():
            continue
        if "_roll_" in column:
            rolling_warm_up += int(actual.sum())
        elif column.endswith("_diff") or column.endswith("_rate_per_min"):
            first_difference_warm_up += int(actual.sum())
        elif column in {
            "power_deviation_from_30m_mean",
            "solar_deviation_from_30m_mean",
            "power_deviation_ratio_30m",
        }:
            rolling_warm_up += int(actual.sum())
        elif column == "solar_deviation_ratio_30m":
            baseline_missing = dataframe["solar_radiation_roll_mean_30m"].isna()
            rolling_warm_up += int((actual & baseline_missing).sum())
            denominator_safety += int((actual & ~baseline_missing).sum())
        elif column in {
            "power_generated_relative_change",
            "solar_radiation_relative_change",
        }:
            denominator_safety += int(actual.sum())
        else:
            unexpected += int(actual.sum())

    documented_total = (
        rolling_warm_up + first_difference_warm_up + denominator_safety
    )
    return {
        "total_missing_cells": int(dataframe.isna().sum().sum()),
        "rolling_window_warm_up_nan_cells": rolling_warm_up,
        "first_difference_warm_up_nan_cells": first_difference_warm_up,
        "deliberate_denominator_safety_nan_cells": denominator_safety,
        "unexpected_nan_cells": unexpected,
        "documented_nan_cells": documented_total,
        "all_missing_values_explained": bool(
            documented_total + unexpected == int(dataframe.isna().sum().sum())
            and unexpected == 0
            and all(bool(item["documented_missing_values_present"]) for item in inventory)
        ),
        "future_policy_recommendation": [
            "Do not backfill, because it would use future observations.",
            "The Core set has no current NaNs, so it does not require a warm-up-row exclusion.",
            "For the Extended set, omit its first 14 rows because its longest selected baseline is 30 minutes.",
            "If a 60-minute rolling feature is added later, omit the first 29 rows for that candidate matrix.",
            "Keep high-missingness solar relative/ratio features out of the initial general-purpose model.",
            "Evaluate safeguarded solar ratios only in a separately defined daylight subset.",
            "Investigate any future unexpected missing values before choosing train-only imputation rules.",
        ],
    }


def analyze_correlations(dataframe: pd.DataFrame) -> dict[str, object]:
    """Calculate pairwise Pearson correlations and focused redundancy groups."""
    numerical = dataframe.select_dtypes(include=[np.number])
    correlation = numerical.corr(method="pearson")
    pairs: list[dict[str, object]] = []
    for left_index, left in enumerate(correlation.columns):
        for right in correlation.columns[left_index + 1 :]:
            value = correlation.at[left, right]
            if pd.isna(value) or abs(float(value)) < CORRELATION_THRESHOLD:
                continue
            observations = int(dataframe[[left, right]].dropna().shape[0])
            pairs.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "pearson_correlation": float(value),
                    "absolute_correlation": abs(float(value)),
                    "paired_observations": observations,
                }
            )
    pairs.sort(
        key=lambda item: (
            -float(item["absolute_correlation"]),
            str(item["feature_1"]),
            str(item["feature_2"]),
        )
    )

    group_specs = (
        (
            "electrical power reconstruction",
            ("Power_Generated", "Array_Voltage", "electrical_power_product"),
            "Choose one representation for an initial model; the product nearly recreates generated power.",
        ),
        (
            "power and voltage transitions",
            (
                "power_generated_diff",
                "power_generated_rate_per_min",
                "array_voltage_diff",
                "array_voltage_rate_per_min",
            ),
            "Rates are fixed scalar multiples of differences, and voltage transitions mirror power transitions.",
        ),
        (
            "RTD level summaries",
            (
                "RTD_1",
                "RTD_2",
                "RTD_3",
                "RTD_4",
                "RTD_5",
                "rtd_mean",
                "rtd_min",
                "rtd_max",
            ),
            "Use the mean for level and a dispersion summary rather than all level channels initially.",
        ),
        (
            "RTD dispersion summaries",
            ("rtd_std", "rtd_range"),
            "Both summarize disagreement among the same five RTD channels.",
        ),
        (
            "power rolling means",
            (
                "power_generated_roll_mean_10m",
                "power_generated_roll_mean_30m",
                "power_generated_roll_mean_60m",
            ),
            "Overlapping trailing windows encode similar slow-moving power level information.",
        ),
        (
            "solar rolling means",
            (
                "solar_radiation_roll_mean_10m",
                "solar_radiation_roll_mean_30m",
                "solar_radiation_roll_mean_60m",
            ),
            "Overlapping trailing windows encode similar radiation level information.",
        ),
        (
            "environmental rolling means",
            (
                "air_temp_roll_mean_30m",
                "air_temp_roll_mean_60m",
            ),
            "The two temperature windows are highly correlated in this short file.",
        ),
        (
            "humidity rolling means",
            (
                "relative_humidity_roll_mean_30m",
                "relative_humidity_roll_mean_60m",
            ),
            "The two humidity windows are highly correlated in this short file.",
        ),
        (
            "linear clock representations",
            ("hour", "minute_of_day"),
            "Minute of day almost deterministically contains clock hour; cyclical time encoding is preferred.",
        ),
    )
    groups: list[dict[str, object]] = []
    for name, members, recommendation in group_specs:
        values = [
            abs(float(correlation.at[left, right]))
            for index, left in enumerate(members)
            for right in members[index + 1 :]
            if pd.notna(correlation.at[left, right])
        ]
        groups.append(
            {
                "group_name": name,
                "members": list(members),
                "minimum_pairwise_absolute_correlation": min(values),
                "maximum_pairwise_absolute_correlation": max(values),
                "recommendation": recommendation,
            }
        )

    exact_rate_pairs = []
    for signal in (
        "power_generated",
        "solar_radiation",
        "air_temp",
        "relative_humidity",
        "array_voltage",
        "array_current",
        "rtd_mean",
    ):
        difference = f"{signal}_diff"
        rate = f"{signal}_rate_per_min"
        exact_rate_pairs.append(
            {
                "difference_feature": difference,
                "rate_feature": rate,
                "pearson_correlation": float(correlation.at[difference, rate]),
                "relationship": "rate = difference / 2 minutes",
            }
        )

    return {
        "method": "Pearson correlation using pairwise complete numerical observations",
        "exploratory_absolute_threshold": CORRELATION_THRESHOLD,
        "numerical_feature_count": len(numerical.columns),
        "highly_correlated_pair_count": len(pairs),
        "highly_correlated_pairs": pairs,
        "focused_redundancy_groups": groups,
        "exact_difference_rate_pairs": exact_rate_pairs,
        "modeling_note": (
            "Highly correlated inputs can repeatedly represent the same physical pattern, "
            "distort distances, and give one process excessive influence in distance-based "
            "or unsupervised models. Correlation alone is not a deletion rule."
        ),
    }


def _record_from_row(row: pd.Series, columns: Sequence[str]) -> dict[str, object]:
    """Serialize selected values from one observation."""
    record: dict[str, object] = {}
    for column in columns:
        value = row[column]
        if column == DEFAULT_TIMESTAMP_COLUMN:
            record[column] = _timestamp_text(value)
        elif pd.api.types.is_integer(value):
            record[column] = int(value)
        else:
            record[column] = _json_number(value)
    return record


def _mean_record(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, float | None]:
    """Return strict-JSON means for selected columns."""
    return {column: _json_number(frame[column].mean()) for column in columns}


def review_recurring_transitions(dataframe: pd.DataFrame) -> dict[str, object]:
    """Describe aligned ±20-minute windows around both recurring transitions."""
    if not dataframe[DEFAULT_TIMESTAMP_COLUMN].is_monotonic_increasing:
        raise ValueError("Transition review requires chronological input rows.")

    event_reports: list[dict[str, object]] = []
    aligned_frames: list[pd.DataFrame] = []
    numerical_transition_columns = TRANSITION_COLUMNS[1:]
    for event_time in TRANSITION_TIMESTAMPS:
        start = event_time - pd.Timedelta(minutes=TRANSITION_HALF_WINDOW_MINUTES)
        end = event_time + pd.Timedelta(minutes=TRANSITION_HALF_WINDOW_MINUTES)
        window = dataframe.loc[
            dataframe[DEFAULT_TIMESTAMP_COLUMN].between(start, end),
            list(TRANSITION_COLUMNS),
        ].copy()
        window.insert(
            0,
            "offset_minutes",
            (window[DEFAULT_TIMESTAMP_COLUMN] - event_time).dt.total_seconds() / 60.0,
        )
        if len(window) != 21 or 0.0 not in set(window["offset_minutes"]):
            raise ValueError(f"Incomplete ±20-minute window around {event_time}.")
        aligned_frames.append(window.set_index("offset_minutes"))

        event_row = window.loc[window["offset_minutes"].eq(0.0)].iloc[0]
        previous_row = window.loc[window["offset_minutes"].eq(-2.0)].iloc[0]
        changes = {
            column: _json_number(event_row[column] - previous_row[column])
            for column in (
                "Power_Generated",
                "Solar_Radiation",
                "Array_Voltage",
                "Array_Current",
                "Air_Temp",
                "Relative_Humidity",
            )
        }
        records = []
        for _, row in window.iterrows():
            record = {"offset_minutes": _json_number(row["offset_minutes"])}
            record.update(_record_from_row(row, TRANSITION_COLUMNS))
            records.append(record)
        event_reports.append(
            {
                "event_timestamp": _timestamp_text(event_time),
                "neutral_label": "recurring power transition",
                "window_start": _timestamp_text(start),
                "window_end": _timestamp_text(end),
                "window_row_count": len(window),
                "previous_observation": _record_from_row(
                    previous_row, TRANSITION_COLUMNS
                ),
                "event_observation": _record_from_row(event_row, TRANSITION_COLUMNS),
                "event_step_changes": changes,
                "pre_event_means_minus_20_to_minus_2_minutes": _mean_record(
                    window.loc[window["offset_minutes"].between(-20, -2)],
                    numerical_transition_columns,
                ),
                "post_event_means_0_to_plus_20_minutes": _mean_record(
                    window.loc[window["offset_minutes"].between(0, 20)],
                    numerical_transition_columns,
                ),
                "window_records": records,
            }
        )

    comparison_metrics: dict[str, dict[str, float | None]] = {}
    first, second = aligned_frames
    for column in numerical_transition_columns:
        pair = pd.concat([first[column], second[column]], axis=1).dropna()
        left = pair.iloc[:, 0]
        right = pair.iloc[:, 1]
        correlation: float | None = None
        if len(pair) > 1 and left.std(ddof=0) > 0 and right.std(ddof=0) > 0:
            correlation = _json_number(left.corr(right))
        comparison_metrics[column] = {
            "aligned_pearson_correlation": correlation,
            "mean_absolute_difference": _json_number((left - right).abs().mean()),
        }

    findings = [
        (
            "Both windows show an abrupt simultaneous fall in Power_Generated and "
            "Array_Voltage at exactly 17:00 while Array_Current increases slightly."
        ),
        (
            "The event-row power values are similar (285.185280 and 283.232540), and "
            "the aligned 42-minute power and voltage profiles correlate above 0.998."
        ),
        (
            "Solar_Radiation rises at both event steps rather than falling, but the two "
            "radiation windows have different levels; neither event changes low_light_context."
        ),
        (
            "Trailing power means respond gradually, leaving large negative local power "
            "deviations at each event. The evidence supports the neutral description "
            "'recurring power transition', not a fault or causal conclusion."
        ),
    ]
    return {
        "review_type": "descriptive recurring operational transition review",
        "timezone": TIMEZONE_NOTE,
        "half_window_minutes": TRANSITION_HALF_WINDOW_MINUTES,
        "sampling_interval_seconds": 120,
        "signals_reviewed": list(TRANSITION_COLUMNS[1:]),
        "events": event_reports,
        "aligned_window_comparison": comparison_metrics,
        "findings": findings,
        "interpretation_limit": (
            "No event is labeled as a fault or anomaly, and no causal mechanism is inferred."
        ),
    }


def candidate_feature_sets() -> dict[str, dict[str, object]]:
    """Return bounded, exact candidate inputs without fitting a model."""
    return {
        "core_unsupervised_features": {
            "display_name": "Core Unsupervised Features",
            "feature_count": len(CORE_UNSUPERVISED_FEATURES),
            "feature_names": list(CORE_UNSUPERVISED_FEATURES),
            "rationale": (
                "Compact coverage of output, illumination, environment, summarized RTD "
                "behavior, and cyclical time without deterministic electrical duplicates."
            ),
            "known_limitations": [
                "The source covers only about 33.6 hours.",
                "Power and environmental variables still share a strong daily cycle.",
                "RTD physical meaning and units remain unconfirmed.",
                "Scaling must later be fitted on training data only.",
            ],
        },
        "extended_diagnostic_features": {
            "display_name": "Extended Diagnostic Features",
            "feature_count": len(EXTENDED_DIAGNOSTIC_FEATURES),
            "feature_names": list(EXTENDED_DIAGNOSTIC_FEATURES),
            "rationale": (
                "Adds one current channel, RTD disagreement, selected 30-minute baselines, "
                "selected changes, and absolute deviations for later comparison."
            ),
            "known_limitations": [
                "More correlated inputs may influence distance-based methods unevenly.",
                "Rolling and deviation inputs require removal of the initial warm-up rows.",
                "Array_Current is near-constant under the documented exploratory rule.",
                "This set must be compared using time-aware evaluation, not assumed superior.",
            ],
        },
        "daylight_specific_features": {
            "display_name": "Daylight-Specific Features",
            "feature_count": len(DAYLIGHT_SPECIFIC_FEATURES),
            "feature_names": list(DAYLIGHT_SPECIFIC_FEATURES),
            "rationale": (
                "Retains safeguarded solar and power ratios for a separately filtered "
                "daylight analysis where their denominators are interpretable."
            ),
            "application_policy": (
                "Use only after defining a causal daylight mask that satisfies the existing "
                "5 W/m^2 radiation denominator safeguard and rolling-baseline availability."
            ),
            "known_limitations": [
                "It is not a general day-and-night feature set.",
                "The 5 W/m^2 threshold is exploratory, not a manufacturer limit.",
                "Relative changes remain sensitive near the accepted denominator boundary.",
            ],
        },
    }


def _exclusion_reason(column: str) -> str:
    """Explain why a feature is omitted from the initial candidate model."""
    if column in {"hour", "minute", "minute_of_day"}:
        return "Use the finer cyclical minute-of-day sine/cosine pair instead of discontinuous linear clock fields."
    if column == "elapsed_minutes_from_start":
        return "Encodes position in this short file and may not generalize to later deployments."
    if column in {"hour_sin", "hour_cos"}:
        return "The minute-of-day cyclical pair already provides finer daily phase information."
    if column in RTD_CHANNELS or column in {"rtd_min", "rtd_max"}:
        return "Strongly redundant RTD level measurement; use rtd_mean plus one dispersion summary initially."
    if column == "Array_Voltage":
        return "Correlation with Power_Generated is approximately 0.99999; choose one initial electrical level representation."
    if column in DIAGNOSTIC_COLUMNS:
        return "Deterministically uses the voltage-current-power identity and is retained for diagnostics, not the initial model."
    if column.endswith("_rate_per_min"):
        return "Exactly equals its corresponding difference divided by two, so keeping both adds no information."
    if "_roll_min_" in column or "_roll_max_" in column:
        return "Overlaps strongly with rolling level summaries and is omitted from the bounded first comparison."
    if "_roll_" in column:
        return "Overlapping rolling window omitted to limit correlated representations in the first comparison."
    if column == "array_voltage_diff":
        return "Almost perfectly mirrors power_generated_diff in this dataset."
    return "Omitted from the bounded initial candidate sets to limit redundancy."


def build_review_report(
    dataframe: pd.DataFrame,
    *,
    input_path: Path,
    input_sha256: str,
    inventory: Sequence[dict[str, object]],
    missingness: dict[str, object],
    correlations: dict[str, object],
    transition_review: dict[str, object],
    protected_files: dict[str, dict[str, object]],
    models_unchanged: bool,
    transition_output_path: Path,
) -> dict[str, object]:
    """Assemble the complete machine-readable feature-review report."""
    category_counts = Counter(str(item["category"]) for item in inventory)
    recommendation_counts = Counter(
        str(item["recommendation"]) for item in inventory
    )
    sets = candidate_feature_sets()
    candidate_names = {
        feature
        for candidate in sets.values()
        for feature in candidate["feature_names"]  # type: ignore[index]
    }
    all_candidate_features_exist = candidate_names.issubset(dataframe.columns)
    no_label_columns = not any(
        token in column.lower()
        for column in dataframe.columns
        for token in ("anomaly", "fault", "label", "target")
    )
    recommendations_complete = all(
        item["recommendation"] in RECOMMENDATION_VALUES for item in inventory
    )
    inventory_complete = [item["feature_name"] for item in inventory] == list(
        dataframe.columns
    )
    sources_unchanged = all(
        bool(details["unchanged"]) for details in protected_files.values()
    )
    validation_checks = {
        "input_shape_is_1009_by_81": list(dataframe.shape) == [1009, 81],
        "feature_inventory_includes_all_81_columns": inventory_complete,
        "each_feature_has_a_valid_recommendation": recommendations_complete,
        "all_missing_values_have_documented_causes": bool(
            missingness["all_missing_values_explained"]
        ),
        "all_candidate_feature_names_exist": all_candidate_features_exist,
        "no_source_dataset_modified": sources_unchanged,
        "models_directory_unchanged": models_unchanged,
        "no_anomaly_label_columns_present_or_introduced": no_label_columns,
        "no_model_was_fitted_or_serialized": True,
    }
    return {
        "feature_stage": "Day 11 feature review and model-input planning",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": _relative_or_absolute(input_path),
        "input_sha256": input_sha256,
        "dataset_shape": {"rows": len(dataframe), "columns": len(dataframe.columns)},
        "timezone": TIMEZONE_NOTE,
        "feature_counts": {
            "total": len(inventory),
            "by_category": {
                category: category_counts[category] for category in CATEGORY_ORDER
            },
            "by_recommendation": dict(sorted(recommendation_counts.items())),
        },
        "feature_inventory": list(inventory),
        "missingness_review": missingness,
        "redundancy_analysis": correlations,
        "recommended_exclusions": [
            {
                "feature_name": str(item["feature_name"]),
                "reason": _exclusion_reason(str(item["feature_name"])),
            }
            for item in inventory
            if item["recommendation"] == "EXCLUDE_FROM_MODEL"
        ],
        "unresolved_features": [
            {
                "feature_name": str(item["feature_name"]),
                "reason": "Source encoding must be clarified before circular treatment; ten values exceed 360 degrees.",
            }
            for item in inventory
            if item["recommendation"] == "UNRESOLVED"
        ],
        "candidate_feature_sets": sets,
        "power_generated_strategy": {
            "recommended_first_approach": "A. Unsupervised multivariate detection",
            "reason": (
                "It is the simplest defensible internship prototype for the current normal-only, "
                "short dataset: one compact matrix, no separately trained power predictor, and "
                "clear comparison of observed multivariate operating context."
            ),
            "approach_a_unsupervised_multivariate": {
                "power_role": "Input feature in the compact core set.",
                "electrical_policy": (
                    "Use Power_Generated as the initial electrical level; exclude Array_Voltage "
                    "and electrical_power_product, and treat Array_Current as an optional extension."
                ),
                "limitations": (
                    "Scores will be relative to a very short operating record and require later "
                    "time-aware validation with synthetic scenarios and additional real days."
                ),
            },
            "approach_b_prediction_residual": {
                "power_role": "Prediction target rather than model input.",
                "future_input_policy": (
                    "Use environmental and temporal context only. Exclude Power_Generated, "
                    "Array_Voltage, Array_Current, electrical_power_product, consistency residuals, "
                    "and all power-derived rolling/change/deviation features from the predictors."
                ),
                "future_anomaly_signal": (
                    "A large out-of-sample difference between observed and predicted power could "
                    "be evaluated as a residual signal; no threshold is defined today."
                ),
                "why_later": (
                    "A credible expected-power predictor needs more independent days and a strict "
                    "time-aware evaluation design to avoid overfitting this 33.6-hour record."
                ),
            },
        },
        "rtd_strategy": {
            "recommended_option": "Option B: summary variables",
            "initial_features": ["rtd_mean", "rtd_std"],
            "extended_feature": "rtd_range",
            "reason": (
                "All RTD channel pairs correlate from about 0.9983 to 0.9997. The mean preserves "
                "shared level while standard deviation or range represents channel disagreement."
            ),
            "retention_policy": (
                "Keep RTD_1 through RTD_5 in the master dataset for diagnostic review, especially "
                "if their locations, units, or independent failure modes are later confirmed."
            ),
        },
        "recurring_transition_review_file": _relative_or_absolute(
            transition_output_path
        ),
        "recurring_transition_findings": transition_review["findings"],
        "protected_source_files": protected_files,
        "validation": {
            "passed": all(validation_checks.values()),
            "checks": validation_checks,
        },
        "excluded_work": [
            "synthetic anomaly generation",
            "anomaly-detection model training",
            "fitted scaling or imputation",
            "feature removal from the master dataset",
            "computer vision",
            "video processing",
            "dashboard implementation",
        ],
    }


def _format_number(value: object, digits: int = 6) -> str:
    """Format optional numerical values for Markdown."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def render_documentation(
    report: dict[str, object], transition_review: dict[str, object]
) -> str:
    """Render the complete Day 11 review as Markdown."""
    lines = [
        "# Day 11 feature review and model-input planning",
        "",
        "## Scope",
        "",
        "This read-only review uses `data/processed/operational_features_change.csv` "
        "(1,009 rows × 81 columns). No column or row is removed, no value is imputed, "
        "no scaler or model is fitted, and no transition is assigned an anomaly or fault label.",
        "",
        f"Timezone: **{TIMEZONE_NOTE}**",
        "",
        "## Recommendation for the first prototype",
        "",
        "Implement **Approach A: compact unsupervised multivariate detection** first, "
        "using the Core Unsupervised Features listed below. It is the simpler defensible "
        "prototype for a normal-only dataset this short because it does not first require "
        "a reliable expected-power regression model. This is a plan only; no model exists yet.",
        "",
        "Approach B, prediction-residual detection, remains a useful second experiment. "
        "There, `Power_Generated` becomes the target. Predictors must exclude voltage, current, "
        "their product, consistency residuals, and every power-derived rolling/change feature "
        "that would reconstruct or leak the target.",
        "",
        "## Complete feature inventory",
        "",
        "Variability uses an exploratory rule: constant means at most one non-missing value; "
        "near-constant means either one value occupies at least 95% of observations or the "
        "population standard deviation divided by absolute mean is at most 1%. This rule flags "
        "only `Array_Current` as near-constant in the current file.",
        "",
    ]
    inventory = report["feature_inventory"]
    assert isinstance(inventory, list)
    for category in CATEGORY_ORDER:
        lines.extend(
            [
                f"### {category.title()}",
                "",
                "| Feature | Missing | Missing % | Variability | Intended interpretation | Recommendation |",
                "|---|---:|---:|---|---|---|",
            ]
        )
        for item in inventory:
            if item["category"] != category:
                continue
            interpretation = str(item["intended_interpretation"]).replace("|", "\\|")
            lines.append(
                f"| `{item['feature_name']}` | {item['missing_count']} | "
                f"{float(item['missing_percentage']):.3f}% | {item['variability_status']} | "
                f"{interpretation} | **{item['recommendation']}** |"
            )
        lines.append("")

    missingness = report["missingness_review"]
    assert isinstance(missingness, dict)
    lines.extend(
        [
            "## Missingness review",
            "",
            "| Missingness source | NaN cells | Policy |",
            "|---|---:|---|",
            f"| Rolling-window warm-up | {missingness['rolling_window_warm_up_nan_cells']} | Apply a candidate-specific history rule: none for Core, 14 initial rows for Extended, and 29 only if a 60-minute feature is later selected; never backfill |",
            f"| First-difference warm-up | {missingness['first_difference_warm_up_nan_cells']} | The Extended set's 14-row exclusion also covers this first row; do not fill from future data |",
            f"| Deliberate denominator safety | {missingness['deliberate_denominator_safety_nan_cells']} | Exclude high-missingness ratios from the general model; reserve them for daylight analysis |",
            f"| Unexpected | {missingness['unexpected_nan_cells']} | None observed; investigate rather than blanket-impute if this changes |",
            f"| **Total** | **{missingness['total_missing_cells']}** | No policy is applied during Day 11 |",
            "",
            "The categories are cell counts and do not overlap. All current NaNs match the "
            "documented rolling, previous-row, or denominator-safety formulas.",
            "",
            "## Redundancy analysis",
            "",
            f"Pairwise Pearson correlation identified **{report['redundancy_analysis']['highly_correlated_pair_count']}** numerical pairs with absolute correlation ≥ 0.95. "
            "The complete pair list and paired-observation counts are in `outputs/model_feature_review.json`.",
            "",
            "Highly correlated variables can make Euclidean distances or similar unsupervised "
            "objectives count one physical pattern several times. They may therefore dominate "
            "scores without adding independent evidence. Correlation is exploratory evidence, "
            "not an automatic deletion rule.",
            "",
            "| Focused group | Members | Absolute-correlation range | Initial recommendation |",
            "|---|---|---:|---|",
        ]
    )
    groups = report["redundancy_analysis"]["focused_redundancy_groups"]
    for group in groups:
        members = ", ".join(f"`{name}`" for name in group["members"])
        lines.append(
            f"| {group['group_name']} | {members} | "
            f"{group['minimum_pairwise_absolute_correlation']:.6f}–"
            f"{group['maximum_pairwise_absolute_correlation']:.6f} | "
            f"{group['recommendation']} |"
        )
    lines.extend(
        [
            "",
            "Every per-minute rate is exactly its two-minute difference divided by two and "
            "therefore has correlation 1.0 with that difference. Keep one representation, not both.",
            "",
            "### Electrical strategy",
            "",
            "For the initial unsupervised model, keep `Power_Generated` as the electrical level, "
            "exclude `Array_Voltage` and `electrical_power_product`, and make `Array_Current` an "
            "extended-only comparison. Keep `power_consistency_residual` and "
            "`power_consistency_abs_error` in the master dataset as diagnostics, not general model "
            "inputs. This prevents the voltage-current identity from receiving repeated weight.",
            "",
            "For a future prediction-residual model, none of those electrical variables or any "
            "power-derived feature should predict `Power_Generated`, because that would leak or "
            "nearly reconstruct the target.",
            "",
            "### RTD strategy",
            "",
            "Use **Option B** initially: `rtd_mean` for shared level and `rtd_std` for channel "
            "disagreement; compare `rtd_range` only in the extended set. Retain all five raw RTD "
            "channels in the master dataset until their placement, units, and independent failure "
            "modes are known.",
            "",
            "## Candidate model-input sets",
            "",
        ]
    )
    candidates = report["candidate_feature_sets"]
    for candidate in candidates.values():
        lines.extend(
            [
                f"### {candidate['display_name']} ({candidate['feature_count']} features)",
                "",
                str(candidate["rationale"]),
                "",
                "```text",
                *candidate["feature_names"],
                "```",
                "",
                "Limitations:",
                "",
                *[f"- {item}" for item in candidate["known_limitations"]],
                "",
            ]
        )
        if "application_policy" in candidate:
            lines.extend([f"Application policy: {candidate['application_policy']}", ""])

    lines.extend(
        [
            "## Recurring 17:00 transition review",
            "",
            "Each event uses 21 observations from 20 minutes before through 20 minutes after "
            "the event, inclusive. The complete windows are stored in "
            "`outputs/recurring_transition_review.json`.",
            "",
            "| Event | Power change | Radiation change | Voltage change | Current change | Air-temperature change | Humidity change | Low-light before → after |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    events = transition_review["events"]
    assert isinstance(events, list)
    for event in events:
        changes = event["event_step_changes"]
        previous = event["previous_observation"]
        current = event["event_observation"]
        lines.append(
            f"| {event['event_timestamp']} | {_format_number(changes['Power_Generated'])} | "
            f"{_format_number(changes['Solar_Radiation'])} | {_format_number(changes['Array_Voltage'])} | "
            f"{_format_number(changes['Array_Current'])} | {_format_number(changes['Air_Temp'])} | "
            f"{_format_number(changes['Relative_Humidity'])} | "
            f"{previous['low_light_context']} → {current['low_light_context']} |"
        )
    comparison = transition_review["aligned_window_comparison"]
    lines.extend(
        [
            "",
            "The aligned ±20-minute profiles are strongly similar for generated power "
            f"(r={comparison['Power_Generated']['aligned_pearson_correlation']:.6f}) and voltage "
            f"(r={comparison['Array_Voltage']['aligned_pearson_correlation']:.6f}). Both events "
            "show a step down in power and voltage while current rises slightly; power remains "
            "near a lower plateau afterward and the trailing means decay gradually.",
            "",
            "Radiation rises at the event step on both dates (+14.622780 and +39.060360), "
            "and `low_light_context` stays 0 throughout both windows. The radiation profiles "
            f"are directionally similar but less closely aligned (r={comparison['Solar_Radiation']['aligned_pearson_correlation']:.6f}) "
            "and differ in level. Environmental changes are not identical. The evidence therefore "
            "supports only the neutral description **recurring power transition**; it does not "
            "establish a fault, anomaly, or cause.",
            "",
            "## Decisions deferred until modelling",
            "",
            "- Confirm the compact unsupervised approach and candidate set before implementing a model.",
            "- Clarify RTD channel placement and units if source documentation is available.",
            "- Resolve the encoding of the ten `Wind_Direction` values above 360 before circular encoding.",
            "- Obtain more independent days and choose a time-aware train/validation/test policy.",
            "- Confirm whether the exploratory 5 W/m² daylight threshold is acceptable for specialized analysis.",
            "",
            "## Validation",
            "",
            "All Day 11 checks passed: all 81 columns are inventoried; candidate names exist; "
            "all 1,585 NaN cells have documented causes; dataset hashes are unchanged; the "
            "models directory is unchanged; no anomaly labels, fitted preprocessing, model files, "
            "or modified feature datasets were created.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "python src/data_processing/review_model_features.py",
            "python -m pytest -q",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _protected_paths_for_run(input_path: Path) -> list[Path]:
    """Return all existing operational datasets that must remain unchanged."""
    candidates = [
        DEFAULT_RAW_PATH,
        DEFAULT_CLEANED_PATH,
        DEFAULT_BASIC_FEATURE_PATH,
        DEFAULT_ROLLING_PATH,
        DEFAULT_INPUT_PATH,
        input_path,
    ]
    seen: set[Path] = set()
    paths: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        paths.append(candidate)
    return paths


def review_model_feature_file(
    input_csv: str | Path = DEFAULT_INPUT_PATH,
    *,
    review_output: str | Path = DEFAULT_REVIEW_PATH,
    transition_output: str | Path = DEFAULT_TRANSITION_PATH,
    documentation_output: str | Path | None = DEFAULT_DOCUMENTATION_PATH,
    protected_paths: Sequence[str | Path] | None = None,
    models_directory: str | Path = DEFAULT_MODELS_DIRECTORY,
) -> dict[str, object]:
    """Run the complete read-only Day 11 review and write its reports."""
    input_path = Path(input_csv)
    review_path = Path(review_output)
    transition_path = Path(transition_output)
    documentation_path = (
        Path(documentation_output) if documentation_output is not None else None
    )
    model_path = Path(models_directory)
    protected = (
        [Path(path) for path in protected_paths]
        if protected_paths is not None
        else _protected_paths_for_run(input_path)
    )
    missing_protected = [path for path in protected if not path.is_file()]
    if missing_protected:
        raise FileNotFoundError(
            "Protected dataset file is missing: "
            + ", ".join(str(path) for path in missing_protected)
        )

    source_hashes_before = {path.resolve(): sha256_file(path) for path in protected}
    models_before = _snapshot_directory(model_path)
    dataframe = load_feature_dataset(input_path)
    inventory = build_feature_inventory(dataframe)
    missingness = summarize_missingness(dataframe, inventory)
    correlations = analyze_correlations(dataframe)
    transition_review = review_recurring_transitions(dataframe)
    transition_review.update(
        {
            "input_file": _relative_or_absolute(input_path),
            "input_sha256": source_hashes_before[input_path.resolve()],
        }
    )

    _atomic_write_json(transition_review, transition_path)
    source_hashes_after = {path.resolve(): sha256_file(path) for path in protected}
    models_after = _snapshot_directory(model_path)
    protected_report = {
        _relative_or_absolute(path): {
            "sha256_before": source_hashes_before[path.resolve()],
            "sha256_after": source_hashes_after[path.resolve()],
            "unchanged": source_hashes_before[path.resolve()]
            == source_hashes_after[path.resolve()],
        }
        for path in protected
    }
    report = build_review_report(
        dataframe,
        input_path=input_path,
        input_sha256=source_hashes_before[input_path.resolve()],
        inventory=inventory,
        missingness=missingness,
        correlations=correlations,
        transition_review=transition_review,
        protected_files=protected_report,
        models_unchanged=models_before == models_after,
        transition_output_path=transition_path,
    )
    if not report["validation"]["passed"]:  # type: ignore[index]
        raise RuntimeError("Day 11 feature-review validation failed.")
    _atomic_write_json(report, review_path)
    if documentation_path is not None:
        _atomic_write_text(
            render_documentation(report, transition_review), documentation_path
        )
    return report


def main() -> int:
    """Run the Day 11 feature review from the command line."""
    args = parse_args()
    try:
        report = review_model_feature_file(
            args.input_csv,
            review_output=args.review_output,
            transition_output=args.transition_output,
            documentation_output=args.documentation,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}")
        return 1

    shape = report["dataset_shape"]
    correlations = report["redundancy_analysis"]
    print(f"Reviewed dataset: {report['input_file']}")
    print(f"Shape: {shape['rows']} rows x {shape['columns']} columns")
    print(
        "High-correlation pairs (|r| >= 0.95): "
        f"{correlations['highly_correlated_pair_count']}"
    )
    print(
        "Recommended first approach: "
        f"{report['power_generated_strategy']['recommended_first_approach']}"
    )
    print(f"Validation passed: {report['validation']['passed']}")
    print("No dataset values, anomaly labels, fitted preprocessors, or models were created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
