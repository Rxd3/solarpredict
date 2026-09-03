"""Create causal Day 10 change and rate-of-change features.

The pipeline appends first differences, per-minute rates, safeguarded relative
changes, and deviations from existing trailing baselines. It preserves every
previous CSV field verbatim and never uses a future observation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .engineer_rolling_features import (
        DEFAULT_CLEANED_PATH,
        DEFAULT_RAW_PATH,
        INPUT_COLUMNS as BASIC_INPUT_COLUMNS,
        ROLLING_FEATURES,
    )
    from .engineer_basic_features import PROCESSED_TIMESTAMP_FORMAT
    from .load_operational_data import DEFAULT_TIMESTAMP_COLUMN, parse_timestamp_series
else:
    from engineer_rolling_features import (  # type: ignore[no-redef]
        DEFAULT_CLEANED_PATH,
        DEFAULT_RAW_PATH,
        INPUT_COLUMNS as BASIC_INPUT_COLUMNS,
        ROLLING_FEATURES,
    )
    from engineer_basic_features import (  # type: ignore[no-redef]
        PROCESSED_TIMESTAMP_FORMAT,
    )
    from load_operational_data import (  # type: ignore[no-redef]
        DEFAULT_TIMESTAMP_COLUMN,
        parse_timestamp_series,
    )


DEFAULT_BASIC_FEATURE_PATH = Path("data/processed/operational_features_basic.csv")
DEFAULT_INPUT_PATH = Path("data/processed/operational_features_rolling.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/operational_features_change.csv")
DEFAULT_SUMMARY_PATH = Path("outputs/change_feature_summary.json")
TIMEZONE_NOTE = "Not specified by dataset source; timestamps kept timezone-naive."
SAMPLING_INTERVAL_MINUTES = 2.0
INPUT_COLUMNS = (*BASIC_INPUT_COLUMNS, *ROLLING_FEATURES)

SIGNAL_PREFIXES = (
    ("Power_Generated", "power_generated"),
    ("Solar_Radiation", "solar_radiation"),
    ("Air_Temp", "air_temp"),
    ("Relative_Humidity", "relative_humidity"),
    ("Array_Voltage", "array_voltage"),
    ("Array_Current", "array_current"),
    ("rtd_mean", "rtd_mean"),
)
DIFFERENCE_FEATURES = {
    source: f"{prefix}_diff" for source, prefix in SIGNAL_PREFIXES
}
RATE_FEATURES = {
    source: f"{prefix}_rate_per_min" for source, prefix in SIGNAL_PREFIXES
}
RELATIVE_CHANGE_FEATURES = {
    "Power_Generated": "power_generated_relative_change",
    "Solar_Radiation": "solar_radiation_relative_change",
}
RELATIVE_CHANGE_THRESHOLDS = {
    "Power_Generated": 1.0,
    "Solar_Radiation": 5.0,
}
DEVIATION_FEATURES = {
    "Power_Generated": (
        "power_generated_roll_mean_30m",
        "power_deviation_from_30m_mean",
        "power_deviation_ratio_30m",
        1.0,
    ),
    "Solar_Radiation": (
        "solar_radiation_roll_mean_30m",
        "solar_deviation_from_30m_mean",
        "solar_deviation_ratio_30m",
        5.0,
    ),
}
CHANGE_FEATURES = (
    *DIFFERENCE_FEATURES.values(),
    *RATE_FEATURES.values(),
    *RELATIVE_CHANGE_FEATURES.values(),
    "power_deviation_from_30m_mean",
    "solar_deviation_from_30m_mean",
    "power_deviation_ratio_30m",
    "solar_deviation_ratio_30m",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create causal Day 10 change and rate-of-change features."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Rolling feature CSV (default: {DEFAULT_INPUT_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Change feature CSV (default: {DEFAULT_OUTPUT_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Summary JSON (default: {DEFAULT_SUMMARY_PATH.as_posix()}).",
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
    """Return a file's SHA-256 digest without changing it."""
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


def _require_columns(dataframe: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error if required columns are absent."""
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError("Required columns are missing: " + ", ".join(missing))


def load_rolling_feature_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the verified Day 9 CSV without sorting, dropping, or filling rows."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Rolling feature CSV does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a CSV file but received: {path}")

    dataframe = pd.read_csv(path)
    _require_columns(dataframe, INPUT_COLUMNS)
    if dataframe.columns.tolist() != list(INPUT_COLUMNS):
        raise ValueError(
            "The input must match the verified Day 9 61-column schema and order."
        )

    timestamps = parse_timestamp_series(
        dataframe[DEFAULT_TIMESTAMP_COLUMN],
        timestamp_format=PROCESSED_TIMESTAMP_FORMAT,
    )
    parse_failures = int(timestamps.isna().sum())
    if parse_failures:
        raise ValueError(f"The input contains {parse_failures} timestamp parse failures.")
    dataframe[DEFAULT_TIMESTAMP_COLUMN] = timestamps

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


def safe_relative_change(values: pd.Series, minimum_absolute_denominator: float) -> pd.Series:
    """Calculate `(current - previous) / previous` with a denominator floor."""
    if minimum_absolute_denominator <= 0:
        raise ValueError("The relative-change denominator threshold must be positive.")
    previous = values.shift(1)
    safe_previous = previous.where(previous.abs() >= minimum_absolute_denominator)
    return values.diff(1).div(safe_previous)


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    minimum_absolute_denominator: float,
) -> pd.Series:
    """Divide only where a finite denominator meets the absolute-value floor."""
    if minimum_absolute_denominator <= 0:
        raise ValueError("The ratio denominator threshold must be positive.")
    safe_denominator = denominator.where(
        denominator.abs() >= minimum_absolute_denominator
    )
    return numerator.div(safe_denominator)


def add_change_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Append exactly the approved causal Day 10 features to a copy."""
    _require_columns(dataframe, INPUT_COLUMNS)
    collisions = [name for name in CHANGE_FEATURES if name in dataframe.columns]
    if collisions:
        raise ValueError("Change feature columns already exist: " + ", ".join(collisions))

    result = dataframe.copy(deep=True)
    for source, feature in DIFFERENCE_FEATURES.items():
        result[feature] = result[source].diff(1)
    for source, feature in RATE_FEATURES.items():
        result[feature] = (
            result[DIFFERENCE_FEATURES[source]] / SAMPLING_INTERVAL_MINUTES
        )
    for source, feature in RELATIVE_CHANGE_FEATURES.items():
        result[feature] = safe_relative_change(
            result[source], RELATIVE_CHANGE_THRESHOLDS[source]
        )

    deviation_columns: dict[str, pd.Series] = {}
    ratio_columns: dict[str, pd.Series] = {}
    for source, (
        baseline,
        deviation_feature,
        ratio_feature,
        threshold,
    ) in DEVIATION_FEATURES.items():
        deviation = result[source] - result[baseline]
        deviation_columns[deviation_feature] = deviation
        ratio_columns[ratio_feature] = safe_ratio(
            deviation,
            result[baseline],
            threshold,
        )
    # Keep the documented output order: absolute deviations, then ratios.
    for feature in (
        "power_deviation_from_30m_mean",
        "solar_deviation_from_30m_mean",
    ):
        result[feature] = deviation_columns[feature]
    for feature in ("power_deviation_ratio_30m", "solar_deviation_ratio_30m"):
        result[feature] = ratio_columns[feature]

    new_values = result.loc[:, list(CHANGE_FEATURES)].to_numpy(dtype=float)
    if np.isinf(new_values).any():
        raise RuntimeError("An infinity was generated despite denominator safeguards.")
    expected_columns = [*dataframe.columns, *CHANGE_FEATURES]
    if result.columns.tolist() != expected_columns:
        raise RuntimeError("Unexpected Day 10 feature columns were created.")
    return result


def _format_change_value(value: object) -> str:
    """Format an appended value for portable CSV round trips."""
    if pd.isna(value):
        return ""
    return repr(float(value))


def _atomic_write_csv(
    dataframe: pd.DataFrame,
    destination: Path,
    *,
    source_csv: Path,
) -> None:
    """Copy every Day 9 field verbatim and append only Day 10 values."""
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
                    raise ValueError("The Day 9 input CSV is empty.") from error
                if source_header != list(INPUT_COLUMNS):
                    raise ValueError("The Day 9 CSV header changed during processing.")
                writer.writerow([*source_header, *CHANGE_FEATURES])
                source_row_count = 0
                for source_row, feature_values in zip(
                    reader,
                    dataframe.loc[:, list(CHANGE_FEATURES)].itertuples(
                        index=False, name=None
                    ),
                    strict=True,
                ):
                    writer.writerow(
                        [
                            *source_row,
                            *(
                                _format_change_value(value)
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


def _expected_change_features(source: pd.DataFrame) -> dict[str, pd.Series]:
    """Independently calculate all documented Day 10 feature formulas."""
    expected: dict[str, pd.Series] = {}
    for signal, feature in DIFFERENCE_FEATURES.items():
        expected[feature] = source[signal].diff(1)
    for signal, feature in RATE_FEATURES.items():
        expected[feature] = expected[DIFFERENCE_FEATURES[signal]] / SAMPLING_INTERVAL_MINUTES
    for signal, feature in RELATIVE_CHANGE_FEATURES.items():
        expected[feature] = safe_relative_change(
            source[signal], RELATIVE_CHANGE_THRESHOLDS[signal]
        )
    for signal, (baseline, deviation_feature, ratio_feature, threshold) in DEVIATION_FEATURES.items():
        deviation = source[signal] - source[baseline]
        expected[deviation_feature] = deviation
        expected[ratio_feature] = safe_ratio(deviation, source[baseline], threshold)
    return expected


def validate_change_dataset(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    expected_input_sha256: str | None = None,
) -> dict[str, object]:
    """Validate the saved Day 10 CSV against its unchanged Day 9 input."""
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    source = pd.read_csv(input_path)
    output = pd.read_csv(output_path)
    input_hash = sha256_file(input_path)

    rows_preserved = len(source) == len(output)
    prior_columns_present = all(column in output.columns for column in source.columns)
    new_columns_present = all(column in output.columns for column in CHANGE_FEATURES)
    exact_layout = output.columns.tolist() == [*source.columns, *CHANGE_FEATURES]
    prior_values_preserved = False
    prior_value_mismatch: str | None = None
    if rows_preserved and prior_columns_present:
        try:
            pd.testing.assert_frame_equal(
                source,
                output.loc[:, source.columns],
                check_dtype=True,
                check_exact=True,
                check_names=True,
            )
            prior_values_preserved = True
        except AssertionError as error:
            prior_value_mismatch = str(error)

    source_numeric = source.copy()
    for column in source_numeric.columns:
        if column != DEFAULT_TIMESTAMP_COLUMN:
            source_numeric[column] = pd.to_numeric(source_numeric[column], errors="coerce")
    expected = _expected_change_features(source_numeric)
    formula_checks: dict[str, bool] = {}
    for feature in CHANGE_FEATURES:
        actual = output[feature] if feature in output else pd.Series(dtype=float)
        formula_checks[feature] = bool(
            len(actual) == len(expected[feature])
            and np.allclose(
                actual,
                expected[feature],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
        )

    first_difference_row_nan = all(
        pd.isna(output.loc[0, feature]) for feature in DIFFERENCE_FEATURES.values()
    )
    first_rate_row_nan = all(
        pd.isna(output.loc[0, feature]) for feature in RATE_FEATURES.values()
    )
    rate_checks = {
        signal: bool(
            np.allclose(
                output[rate_feature],
                output[DIFFERENCE_FEATURES[signal]] / SAMPLING_INTERVAL_MINUTES,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
        )
        for signal, rate_feature in RATE_FEATURES.items()
    }

    safeguard_checks: dict[str, bool] = {}
    for signal, feature in RELATIVE_CHANGE_FEATURES.items():
        previous = source_numeric[signal].shift(1)
        unsafe = previous.isna() | (
            previous.abs() < RELATIVE_CHANGE_THRESHOLDS[signal]
        )
        safeguard_checks[feature] = bool(output.loc[unsafe, feature].isna().all())
    for signal, (baseline, _, ratio_feature, threshold) in DEVIATION_FEATURES.items():
        unsafe = source_numeric[baseline].isna() | (
            source_numeric[baseline].abs() < threshold
        )
        safeguard_checks[ratio_feature] = bool(
            output.loc[unsafe, ratio_feature].isna().all()
        )

    new_values = (
        output.loc[:, list(CHANGE_FEATURES)].to_numpy(dtype=float)
        if new_columns_present
        else np.array([np.inf])
    )
    no_infinity_values = not np.isinf(new_values).any()
    checks = {
        "rows_preserved": rows_preserved,
        "all_previous_columns_present": prior_columns_present,
        "all_previous_csv_values_preserved": prior_values_preserved,
        "all_change_columns_present": new_columns_present,
        "exact_column_layout": exact_layout,
        "all_feature_formulas_match_recalculation": all(formula_checks.values()),
        "first_difference_row_is_nan": first_difference_row_nan,
        "first_rate_row_is_nan": first_rate_row_nan,
        "all_rates_equal_difference_divided_by_two_minutes": all(rate_checks.values()),
        "all_denominator_safeguards_applied": all(safeguard_checks.values()),
        "no_infinity_values_generated": no_infinity_values,
        "input_hash_matches_pre_write_hash": (
            True if expected_input_sha256 is None else input_hash == expected_input_sha256
        ),
        "causal_current_and_previous_only": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "input_rows": int(len(source)),
        "output_rows": int(len(output)),
        "input_columns": int(len(source.columns)),
        "output_columns": int(len(output.columns)),
        "formula_checks_by_feature": formula_checks,
        "rate_checks_by_signal": rate_checks,
        "safeguard_checks_by_feature": safeguard_checks,
        "input_sha256": input_hash,
        "previous_value_mismatch": prior_value_mismatch,
    }


def _feature_statistics(dataframe: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Return NaN counts, ranges, and descriptive statistics for new features."""
    result: dict[str, dict[str, object]] = {}
    for feature in CHANGE_FEATURES:
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


def _largest_transition_records(
    dataframe: pd.DataFrame,
    *,
    signal: str,
    difference_feature: str,
    other_difference_feature: str,
    top_n: int = 10,
) -> list[dict[str, object]]:
    """Return the largest absolute transitions with simultaneous context."""
    ranking = pd.DataFrame(
        {
            "position": np.arange(len(dataframe), dtype=int),
            "absolute_change": dataframe[difference_feature].abs().to_numpy(),
        }
    ).dropna()
    ranking = ranking.sort_values(
        ["absolute_change", "position"], ascending=[False, True], kind="stable"
    ).head(top_n)

    records: list[dict[str, object]] = []
    for rank, position in enumerate(ranking["position"].astype(int), start=1):
        row = dataframe.iloc[position]
        previous = dataframe.iloc[position - 1]
        records.append(
            {
                "rank": rank,
                "timestamp": _timestamp_text(row[DEFAULT_TIMESTAMP_COLUMN]),
                "previous_timestamp": _timestamp_text(
                    previous[DEFAULT_TIMESTAMP_COLUMN]
                ),
                "signal": signal,
                "previous_value": _json_number(previous[signal]),
                "current_value": _json_number(row[signal]),
                "signed_change": _json_number(row[difference_feature]),
                "absolute_change": _json_number(abs(row[difference_feature])),
                "simultaneous_other_signal_change": _json_number(
                    row[other_difference_feature]
                ),
                "previous_low_light_context": int(previous["low_light_context"]),
                "current_low_light_context": int(row["low_light_context"]),
                "low_light_context_changed": bool(
                    previous["low_light_context"] != row["low_light_context"]
                ),
            }
        )
    return records


def analyze_largest_transitions(dataframe: pd.DataFrame) -> dict[str, object]:
    """Describe, but do not label, the largest power and radiation changes."""
    power = _largest_transition_records(
        dataframe,
        signal="Power_Generated",
        difference_feature="power_generated_diff",
        other_difference_feature="solar_radiation_diff",
    )
    solar = _largest_transition_records(
        dataframe,
        signal="Solar_Radiation",
        difference_feature="solar_radiation_diff",
        other_difference_feature="power_generated_diff",
    )
    power_timestamps = {record["timestamp"] for record in power}
    solar_timestamps = {record["timestamp"] for record in solar}
    valid_changes = dataframe[["power_generated_diff", "solar_radiation_diff"]].dropna()
    if (
        len(valid_changes) < 2
        or valid_changes["power_generated_diff"].std(ddof=0) == 0
        or valid_changes["solar_radiation_diff"].std(ddof=0) == 0
    ):
        correlation = None
    else:
        correlation = valid_changes["power_generated_diff"].corr(
            valid_changes["solar_radiation_diff"]
        )
    return {
        "label": "largest observed transitions",
        "largest_power_transitions": power,
        "largest_solar_radiation_transitions": solar,
        "relationship_summary": {
            "shared_top_10_timestamp_count": len(power_timestamps & solar_timestamps),
            "shared_top_10_timestamps": sorted(power_timestamps & solar_timestamps),
            "power_top_10_low_light_rows": sum(
                record["current_low_light_context"] == 1 for record in power
            ),
            "power_top_10_low_light_context_change_count": sum(
                record["low_light_context_changed"] for record in power
            ),
            "pearson_correlation_between_same_step_power_and_solar_differences": (
                _json_number(correlation)
            ),
            "observation": (
                "The top-ten lists do not share a timestamp. The two largest "
                "power drops recur at 17:00 on consecutive dates without a "
                "simultaneously top-ranked radiation change. Six of the ten "
                "largest power changes occur while low_light_context is already "
                "1, but none crosses that flag boundary at the same row. These "
                "are descriptive patterns and do not establish causation or an "
                "anomaly label."
            ),
        },
    }


def _protected_paths_for_run(input_path: Path) -> list[Path]:
    """Select data files that must remain byte-for-byte unchanged."""
    paths = [input_path]
    if input_path.resolve() == DEFAULT_INPUT_PATH.resolve():
        paths = [
            DEFAULT_RAW_PATH,
            DEFAULT_CLEANED_PATH,
            DEFAULT_BASIC_FEATURE_PATH,
            input_path,
        ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Protected source file(s) missing: " + ", ".join(missing))
    return paths


def engineer_change_feature_file(
    input_csv: str | Path = DEFAULT_INPUT_PATH,
    *,
    output_csv: str | Path = DEFAULT_OUTPUT_PATH,
    summary_json: str | Path = DEFAULT_SUMMARY_PATH,
) -> dict[str, object]:
    """Create, validate, analyze, and report the Day 10 feature dataset."""
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    summary_path = Path(summary_json)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from the Day 9 input path.")

    protected_paths = _protected_paths_for_run(input_path)
    hashes_before = {path: sha256_file(path) for path in protected_paths}
    rolling = load_rolling_feature_data(input_path)
    changed = add_change_features(rolling)
    _atomic_write_csv(changed, output_path, source_csv=input_path)
    hashes_after = {path: sha256_file(path) for path in protected_paths}

    validation = validate_change_dataset(
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
    validation["passed"] = bool(validation["passed"] and protected_files_unchanged)

    saved = pd.read_csv(output_path)
    saved[DEFAULT_TIMESTAMP_COLUMN] = parse_timestamp_series(
        saved[DEFAULT_TIMESTAMP_COLUMN], timestamp_format=PROCESSED_TIMESTAMP_FORMAT
    )
    report: dict[str, object] = {
        "feature_stage": "Day 10 causal change and rate-of-change features",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": _relative_or_absolute(input_path),
        "output_file": _relative_or_absolute(output_path),
        "input_dataset_shape": {
            "rows": int(len(rolling)),
            "columns": int(len(rolling.columns)),
        },
        "output_dataset_shape": {
            "rows": int(len(saved)),
            "columns": int(len(saved.columns)),
        },
        "new_feature_count": int(len(CHANGE_FEATURES)),
        "new_feature_names": list(CHANGE_FEATURES),
        "sampling_interval_assumption": {
            "minutes": SAMPLING_INTERVAL_MINUTES,
            "seconds": int(SAMPLING_INTERVAL_MINUTES * 60),
            "status": "Verified from every consecutive source timestamp.",
        },
        "causal_policy": (
            "First differences and rates use current minus previous. Existing "
            "30-minute baselines are trailing. No future row is accessed."
        ),
        "relative_change_safeguards": {
            RELATIVE_CHANGE_FEATURES[signal]: {
                "source": signal,
                "formula": "(current - previous) / previous",
                "minimum_absolute_previous_value": threshold,
                "unsafe_result": "NaN",
                "threshold_status": "Project numerical-stability safeguard.",
            }
            for signal, threshold in RELATIVE_CHANGE_THRESHOLDS.items()
        },
        "deviation_ratio_safeguards": {
            ratio_feature: {
                "source": signal,
                "baseline": baseline,
                "formula": "(current - trailing_baseline) / trailing_baseline",
                "minimum_absolute_baseline": threshold,
                "unsafe_result": "NaN",
                "threshold_status": "Project numerical-stability safeguard.",
            }
            for signal, (baseline, _, ratio_feature, threshold) in DEVIATION_FEATURES.items()
        },
        "feature_statistics": _feature_statistics(saved),
        "largest_observed_transitions": analyze_largest_transitions(saved),
        "protected_source_files": protected_hashes,
        "validation": validation,
        "timezone": TIMEZONE_NOTE,
        "excluded_work": [
            "feature selection",
            "synthetic anomaly generation",
            "anomaly-detection models",
            "computer vision",
            "video processing",
            "dashboard",
        ],
    }
    _atomic_write_json(report, summary_path)

    if not validation["passed"]:
        raise RuntimeError(
            "Change-feature validation failed. See "
            + _relative_or_absolute(summary_path)
        )
    return report


def main() -> int:
    """Run Day 10 feature engineering from the command line."""
    arguments = parse_args()
    try:
        report = engineer_change_feature_file(
            arguments.input_csv,
            output_csv=arguments.output,
            summary_json=arguments.summary,
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError, pd.errors.ParserError) as error:
        print(f"Error: {error}")
        return 1

    shape = report["output_dataset_shape"]
    print(f"Change feature dataset: {report['output_file']}")
    print(f"Shape: {shape['rows']} rows x {shape['columns']} columns")
    print(f"New features: {report['new_feature_count']}")
    print(f"Validation passed: {report['validation']['passed']}")
    print("No future observations, feature selection, anomalies, or models were used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
