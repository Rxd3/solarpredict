"""Conservatively preprocess the verified operational solar dataset.

The pipeline removes only exact duplicate rows, never removes outliers, never
imputes missing measurements, and verifies that the source file is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__:
    from .load_operational_data import (
        DEFAULT_TIMESTAMP_COLUMN,
        DEFAULT_TIMESTAMP_FORMAT,
        SELECTED_NUMERIC_COLUMNS,
        build_timestamp_profile,
        load_operational_csv,
        parse_timestamp_series,
    )
else:
    from load_operational_data import (  # type: ignore[no-redef]
        DEFAULT_TIMESTAMP_COLUMN,
        DEFAULT_TIMESTAMP_FORMAT,
        SELECTED_NUMERIC_COLUMNS,
        build_timestamp_profile,
        load_operational_csv,
        parse_timestamp_series,
    )


DEFAULT_RAW_PATH = Path(
    "data/operational/solar-power-dataset/solar_data.csv"
)
DEFAULT_PROCESSED_PATH = Path("data/processed/operational_cleaned.csv")
DEFAULT_SUMMARY_PATH = Path("outputs/preprocessing_summary.json")
OUTPUT_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMEZONE_NOTE = "Not specified by dataset source; timestamps kept timezone-naive."
REQUIRED_COLUMNS = (DEFAULT_TIMESTAMP_COLUMN, *SELECTED_NUMERIC_COLUMNS)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Conservatively preprocess the verified operational CSV."
    )
    parser.add_argument(
        "raw_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help=f"Raw CSV path (default: {DEFAULT_RAW_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROCESSED_PATH,
        help=f"Processed CSV path (default: {DEFAULT_PROCESSED_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Summary JSON path (default: {DEFAULT_SUMMARY_PATH.as_posix()}).",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute(path: Path) -> str:
    """Use a portable project-relative path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _numeric_range(values: pd.Series) -> dict[str, float | None]:
    """Return JSON-safe minimum and maximum values after numeric coercion."""
    numeric = pd.to_numeric(values, errors="coerce")

    def finite_or_none(value: object) -> float | None:
        if pd.isna(value):
            return None
        converted = float(value)
        return converted if math.isfinite(converted) else None

    return {
        "minimum": finite_or_none(numeric.min()),
        "maximum": finite_or_none(numeric.max()),
    }


def _same_number(left: float | None, right: float | None) -> bool:
    """Compare optional floats using a tight round-trip tolerance."""
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _atomic_write_csv(dataframe: pd.DataFrame, destination: Path) -> None:
    """Write a CSV atomically so a failed write does not leave a partial file."""
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
                date_format=OUTPUT_TIMESTAMP_FORMAT,
            )
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(report: dict[str, object], destination: Path) -> None:
    """Write the preprocessing report atomically."""
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


def validate_preprocessed_data(
    raw_csv: str | Path,
    processed_csv: str | Path,
    *,
    expected_raw_sha256: str | None = None,
) -> dict[str, object]:
    """Compare the raw and processed datasets and return validation evidence."""
    raw_path = Path(raw_csv)
    processed_path = Path(processed_csv)
    raw = pd.read_csv(raw_path)
    processed_raw = pd.read_csv(processed_path)

    missing_required_columns = [
        column for column in REQUIRED_COLUMNS if column not in processed_raw.columns
    ]
    unexpected_columns = [
        column for column in processed_raw.columns if column not in raw.columns
    ]
    lost_columns = [column for column in raw.columns if column not in processed_raw.columns]

    raw_duplicate_rows = int(raw.duplicated().sum())
    expected_rows = int(len(raw) - raw_duplicate_rows)

    raw_timestamp_profile = build_timestamp_profile(
        raw[DEFAULT_TIMESTAMP_COLUMN], timestamp_format=DEFAULT_TIMESTAMP_FORMAT
    )
    timestamp_column_present = DEFAULT_TIMESTAMP_COLUMN in processed_raw.columns
    if timestamp_column_present:
        processed_timestamp = parse_timestamp_series(
            processed_raw[DEFAULT_TIMESTAMP_COLUMN],
            timestamp_format=OUTPUT_TIMESTAMP_FORMAT,
        )
        processed_timestamp_profile = build_timestamp_profile(
            processed_raw[DEFAULT_TIMESTAMP_COLUMN],
            timestamp_format=OUTPUT_TIMESTAMP_FORMAT,
        )
        missing_timestamp_mask = processed_timestamp.isna().tolist()
        unparseable_timestamps_sorted_last = (
            not any(missing_timestamp_mask)
            or missing_timestamp_mask == sorted(missing_timestamp_mask)
        )
    else:
        processed_timestamp = pd.Series(dtype="datetime64[us]")
        processed_timestamp_profile = None
        unparseable_timestamps_sorted_last = False

    numerical_ranges: dict[str, dict[str, object]] = {}
    ranges_preserved = True
    for column in SELECTED_NUMERIC_COLUMNS:
        if column not in raw.columns or column not in processed_raw.columns:
            ranges_preserved = False
            continue
        raw_range = _numeric_range(raw[column])
        processed_range = _numeric_range(processed_raw[column])
        range_matches = _same_number(
            raw_range["minimum"], processed_range["minimum"]
        ) and _same_number(raw_range["maximum"], processed_range["maximum"])
        ranges_preserved = ranges_preserved and range_matches
        numerical_ranges[column] = {
            "raw": raw_range,
            "processed": processed_range,
            "preserved": range_matches,
        }

    if not missing_required_columns:
        expected = load_operational_csv(
            raw_path,
            timestamp_column=DEFAULT_TIMESTAMP_COLUMN,
            timestamp_format=DEFAULT_TIMESTAMP_FORMAT,
            numeric_columns=SELECTED_NUMERIC_COLUMNS,
            sort_chronologically=True,
            drop_duplicate_rows=True,
        )
        actual = load_operational_csv(
            processed_path,
            timestamp_column=DEFAULT_TIMESTAMP_COLUMN,
            timestamp_format=OUTPUT_TIMESTAMP_FORMAT,
            numeric_columns=SELECTED_NUMERIC_COLUMNS,
            sort_chronologically=False,
            drop_duplicate_rows=False,
        )
        try:
            pd.testing.assert_frame_equal(
                expected,
                actual,
                check_dtype=True,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
                check_names=True,
            )
            content_matches_expected = True
            content_mismatch = None
        except AssertionError as error:
            content_matches_expected = False
            content_mismatch = str(error)
    else:
        content_matches_expected = False
        content_mismatch = "Processed file is missing required columns."

    current_raw_hash = sha256_file(raw_path)
    raw_hash_matches = (
        expected_raw_sha256 is None or current_raw_hash == expected_raw_sha256
    )
    checks = {
        "raw_file_hash_unchanged": raw_hash_matches,
        "row_count_matches_exact_duplicate_policy": len(processed_raw) == expected_rows,
        "all_raw_columns_preserved": not lost_columns,
        "all_required_columns_present": not missing_required_columns,
        "no_unexpected_columns_added": not unexpected_columns,
        "column_order_preserved": processed_raw.columns.tolist() == raw.columns.tolist(),
        "timestamp_parse_failure_count_preserved": (
            processed_timestamp_profile is not None
            and processed_timestamp_profile["parse_failures"]
            == raw_timestamp_profile["parse_failures"]
        ),
        "valid_timestamps_monotonic_increasing": bool(
            processed_timestamp.dropna().is_monotonic_increasing
        ),
        "unparseable_timestamps_sorted_last": unparseable_timestamps_sorted_last,
        "numerical_ranges_preserved": ranges_preserved,
        "processed_content_matches_expected_pipeline": content_matches_expected,
    }
    issues = [name for name, passed in checks.items() if not passed]

    return {
        "passed": not issues,
        "checks": checks,
        "issues": issues,
        "raw_rows": int(len(raw)),
        "processed_rows": int(len(processed_raw)),
        "expected_processed_rows": expected_rows,
        "rows_removed_fraction": (
            float((len(raw) - len(processed_raw)) / len(raw)) if len(raw) else 0.0
        ),
        "lost_columns": lost_columns,
        "missing_required_columns": missing_required_columns,
        "unexpected_columns": unexpected_columns,
        "raw_timestamp_profile": raw_timestamp_profile,
        "processed_timestamp_profile": processed_timestamp_profile,
        "numerical_ranges": numerical_ranges,
        "content_mismatch": content_mismatch,
        "raw_sha256_after_processing": current_raw_hash,
    }


def preprocess_operational_data(
    raw_csv: str | Path = DEFAULT_RAW_PATH,
    *,
    output_csv: str | Path = DEFAULT_PROCESSED_PATH,
    summary_json: str | Path = DEFAULT_SUMMARY_PATH,
) -> dict[str, object]:
    """Run conservative preprocessing, validation, and summary generation."""
    raw_path = Path(raw_csv)
    output_path = Path(output_csv)
    summary_path = Path(summary_json)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw operational CSV does not exist: {raw_path}")
    if not raw_path.is_file():
        raise ValueError(f"Expected a raw CSV file but received: {raw_path}")
    if raw_path.resolve() == output_path.resolve():
        raise ValueError("The processed output must not overwrite the raw CSV.")

    raw_hash_before = sha256_file(raw_path)
    raw = pd.read_csv(raw_path)
    missing_required = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing_required:
        raise ValueError(
            "Raw file does not match the verified schema; missing columns: "
            + ", ".join(missing_required)
        )

    missing_before = {
        column: int(count) for column, count in raw.isna().sum().items()
    }
    duplicate_rows_found = int(raw.duplicated().sum())
    timestamp_before = parse_timestamp_series(
        raw[DEFAULT_TIMESTAMP_COLUMN], timestamp_format=DEFAULT_TIMESTAMP_FORMAT
    )
    timestamp_profile_before = build_timestamp_profile(
        raw[DEFAULT_TIMESTAMP_COLUMN], timestamp_format=DEFAULT_TIMESTAMP_FORMAT
    )

    deduplicated = raw.drop_duplicates(keep="first").copy()
    parsed_for_order = parse_timestamp_series(
        deduplicated[DEFAULT_TIMESTAMP_COLUMN],
        timestamp_format=DEFAULT_TIMESTAMP_FORMAT,
    )
    sorted_source_indices = (
        deduplicated.assign(_parsed_timestamp=parsed_for_order)
        .sort_values("_parsed_timestamp", kind="stable", na_position="last")
        .index.tolist()
    )
    source_order_changed = sorted_source_indices != deduplicated.index.tolist()

    processed = load_operational_csv(
        raw_path,
        timestamp_column=DEFAULT_TIMESTAMP_COLUMN,
        timestamp_format=DEFAULT_TIMESTAMP_FORMAT,
        numeric_columns=SELECTED_NUMERIC_COLUMNS,
        sort_chronologically=True,
        drop_duplicate_rows=True,
        duplicate_keep="first",
    )
    missing_after = {
        column: int(count) for column, count in processed.isna().sum().items()
    }
    invalid_numeric_created_missing = {
        column: max(missing_after[column] - missing_before[column], 0)
        for column in SELECTED_NUMERIC_COLUMNS
    }
    timestamp_profile_after = build_timestamp_profile(
        processed[DEFAULT_TIMESTAMP_COLUMN], timestamp_format=None
    )

    warnings: list[str] = [
        "Timezone is not specified by the source; timestamps remain timezone-naive."
    ]
    if sum(missing_after.values()):
        warnings.append(
            "Missing values were preserved as missing; no rows were dropped and no values were imputed."
        )
    if sum(invalid_numeric_created_missing.values()):
        warnings.append(
            "Invalid numeric text was coerced to missing values and retained for review."
        )
    if timestamp_profile_before["parse_failures"]:
        warnings.append(
            "Unparseable timestamps were retained as missing timestamps and sorted last."
        )
    if timestamp_profile_before["irregular_gap_count"]:
        warnings.append(
            "Irregular timestamp gaps were documented and not filled or interpolated."
        )

    # Duplicate timestamps that remain after exact-row deduplication may contain
    # different measurements and must not be discarded automatically.
    remaining_duplicate_timestamps = int(
        processed[DEFAULT_TIMESTAMP_COLUMN].dropna().duplicated().sum()
    )
    if remaining_duplicate_timestamps:
        warnings.append(
            "Duplicate timestamps with non-identical rows were preserved for manual review."
        )

    negative_radiation = int((processed["Solar_Radiation"] < 0).sum())
    invalid_wind_direction = int(
        ((processed["Wind_Direction"] < 0) | (processed["Wind_Direction"] > 360)).sum()
    )
    if negative_radiation:
        warnings.append(
            f"Preserved {negative_radiation} Solar_Radiation values below 0 W/m^2."
        )
    if invalid_wind_direction:
        warnings.append(
            f"Preserved {invalid_wind_direction} Wind_Direction values outside 0 to 360 degrees."
        )

    _atomic_write_csv(processed, output_path)
    validation = validate_preprocessed_data(
        raw_path,
        output_path,
        expected_raw_sha256=raw_hash_before,
    )
    if not validation["passed"]:
        raise RuntimeError(
            "Processed dataset validation failed: "
            + ", ".join(validation["issues"])
        )

    raw_hash_after = sha256_file(raw_path)
    report: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Solar Power Dataset",
        "raw_file": _relative_or_absolute(raw_path),
        "processed_file": _relative_or_absolute(output_path),
        "raw_sha256_before_processing": raw_hash_before,
        "raw_sha256_after_processing": raw_hash_after,
        "processed_sha256": sha256_file(output_path),
        "raw_file_unchanged": raw_hash_before == raw_hash_after,
        "timezone": TIMEZONE_NOTE,
        "timestamp_input_format": DEFAULT_TIMESTAMP_FORMAT,
        "timestamp_output_format": OUTPUT_TIMESTAMP_FORMAT,
        "rows_before_preprocessing": int(len(raw)),
        "rows_after_preprocessing": int(len(processed)),
        "duplicate_rows_found": duplicate_rows_found,
        "duplicate_rows_removed": duplicate_rows_found,
        "duplicate_timestamps_before": int(timestamp_before.dropna().duplicated().sum()),
        "duplicate_timestamps_after": remaining_duplicate_timestamps,
        "source_order_was_changed_by_sorting": source_order_changed,
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "total_missing_values_before": int(sum(missing_before.values())),
        "total_missing_values_after": int(sum(missing_after.values())),
        "invalid_numeric_values_coerced_to_missing": invalid_numeric_created_missing,
        "columns_converted_to_numeric": list(SELECTED_NUMERIC_COLUMNS),
        "timestamp_parsing_failures_before": int(
            timestamp_profile_before["parse_failures"]
        ),
        "timestamp_parsing_failures_after": int(
            timestamp_profile_after["parse_failures"]
        ),
        "start_timestamp": timestamp_profile_after["minimum"],
        "end_timestamp": timestamp_profile_after["maximum"],
        "observed_median_interval_seconds": timestamp_profile_after[
            "median_interval_seconds"
        ],
        "irregular_timestamp_gaps": timestamp_profile_after[
            "irregular_gap_count"
        ],
        "preprocessing_steps": [
            "Validated the raw file against the verified 14-column schema.",
            "Parsed Timestamp using %d-%m-%Y %H:%M without timezone localization.",
            "Removed exact duplicate rows only, keeping the first occurrence.",
            "Stable-sorted rows chronologically; equal timestamps retain source order.",
            "Coerced the 13 verified measurement columns to numeric dtypes.",
            "Preserved missing values without dropping rows or imputing values.",
            "Preserved range-flagged and unusual measurements without clipping or outlier removal.",
            "Serialized timestamps as %Y-%m-%d %H:%M:%S for an unambiguous processed CSV.",
        ],
        "processing_warnings": warnings,
        "validation": validation,
    }
    _atomic_write_json(report, summary_path)
    return report


def main() -> int:
    """Run preprocessing and return a process exit code."""
    args = parse_args()
    try:
        report = preprocess_operational_data(
            args.raw_csv,
            output_csv=args.output,
            summary_json=args.summary,
        )
    except (
        FileNotFoundError,
        PermissionError,
        RuntimeError,
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Error: unable to preprocess operational data: {error}", file=sys.stderr)
        return 1

    print(f"Raw rows: {report['rows_before_preprocessing']}")
    print(f"Processed rows: {report['rows_after_preprocessing']}")
    print(f"Duplicate rows removed: {report['duplicate_rows_removed']}")
    print(f"Missing values after: {report['total_missing_values_after']}")
    print(f"Processed CSV: {report['processed_file']}")
    print(f"Summary JSON: {args.summary}")
    print(f"Validation passed: {report['validation']['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
