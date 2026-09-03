"""Inspect one CSV or every CSV below a directory without changing source data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__:
    from .load_operational_data import (
        DEFAULT_TIMESTAMP_FORMAT,
        build_timestamp_profile,
        discover_csv_files,
        infer_timestamp_column,
    )
else:
    from load_operational_data import (  # type: ignore[no-redef]
        DEFAULT_TIMESTAMP_FORMAT,
        build_timestamp_profile,
        discover_csv_files,
        infer_timestamp_column,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect one CSV file or recursively inspect a directory of CSVs."
    )
    parser.add_argument(
        "dataset_path", type=Path, help="CSV file or directory containing CSV files."
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        metavar="N",
        help="Number of rows to show in each preview (default: 5).",
    )
    parser.add_argument(
        "--timestamp-column",
        help="Timestamp column name; inferred when omitted.",
    )
    parser.add_argument(
        "--timestamp-format",
        default=DEFAULT_TIMESTAMP_FORMAT,
        help=(
            "pandas timestamp format (default for the inspected dataset: "
            "%%d-%%m-%%Y %%H:%%M)."
        ),
    )
    parser.add_argument(
        "--unique-threshold",
        type=int,
        default=50,
        metavar="N",
        help="Print values for columns with at most N unique values (default: 50).",
    )
    return parser.parse_args()


def build_inspection_report(
    csv_path: str | Path,
    *,
    timestamp_column: str | None = None,
    timestamp_format: str | None = DEFAULT_TIMESTAMP_FORMAT,
    unique_threshold: int = 50,
) -> dict[str, object]:
    """Build a reusable inspection summary for one CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a CSV file but received: {path}")
    if unique_threshold < 0:
        raise ValueError("--unique-threshold must be zero or a positive integer.")

    dataframe = pd.read_csv(path)
    resolved_timestamp_column = timestamp_column
    if resolved_timestamp_column is None:
        try:
            resolved_timestamp_column = infer_timestamp_column(dataframe.columns)
        except ValueError:
            resolved_timestamp_column = None

    timestamp_profile = None
    if resolved_timestamp_column is not None:
        if resolved_timestamp_column not in dataframe.columns:
            raise ValueError(f"Timestamp column not found: {resolved_timestamp_column}")
        timestamp_profile = build_timestamp_profile(
            dataframe[resolved_timestamp_column], timestamp_format=timestamp_format
        )

    unique_summary: dict[str, dict[str, object]] = {}
    for column in dataframe.columns:
        unique_count = int(dataframe[column].nunique(dropna=False))
        entry: dict[str, object] = {"count_including_missing": unique_count}
        if unique_count <= unique_threshold:
            entry["values"] = [
                "<missing>" if pd.isna(value) else str(value)
                for value in dataframe[column].drop_duplicates().tolist()
            ]
        unique_summary[column] = entry

    numeric_statistics = dataframe.select_dtypes(include="number").describe().transpose()

    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": dataframe.columns.tolist(),
        "inferred_dtypes": {
            column: str(dtype) for column, dtype in dataframe.dtypes.items()
        },
        "first_rows": dataframe.head(5),
        "missing_values": {
            column: int(count) for column, count in dataframe.isna().sum().items()
        },
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "unique_values": unique_summary,
        "numeric_statistics": numeric_statistics,
        "timestamp_column": resolved_timestamp_column,
        "timestamp_profile": timestamp_profile,
    }


def print_inspection_report(report: dict[str, object], head_rows: int = 5) -> None:
    """Print a human-readable inspection report."""
    if head_rows < 0:
        raise ValueError("--head must be zero or a positive integer.")

    print("=" * 80)
    print(f"Filename: {report['filename']}")
    print(f"File: {report['path']}")
    print(f"Shape: {report['rows']} rows x {report['columns']} columns")

    print("\nColumn names:")
    for column in report["column_names"]:  # type: ignore[union-attr]
        print(f"  - {column}")

    print(f"\nFirst {head_rows} rows:")
    first_rows = report["first_rows"]
    print(first_rows.head(head_rows).to_string(index=False))  # type: ignore[union-attr]

    print("\nData types:")
    for column, dtype in report["inferred_dtypes"].items():  # type: ignore[union-attr]
        print(f"  {column}: {dtype}")

    print("\nMissing values:")
    for column, count in report["missing_values"].items():  # type: ignore[union-attr]
        print(f"  {column}: {count}")

    print(f"\nDuplicate rows: {report['duplicate_rows']}")

    print("\nUnique values:")
    for column, details in report["unique_values"].items():  # type: ignore[union-attr]
        print(f"  {column}: {details['count_including_missing']} unique")
        if "values" in details:
            print(f"    Values: {details['values']}")

    print("\nDescriptive statistics:")
    print("  Numerical columns:")
    statistics = report["numeric_statistics"]
    if statistics.empty:  # type: ignore[union-attr]
        print("  (no numerical columns)")
    else:
        print(statistics.to_string())  # type: ignore[union-attr]

    profile = report["timestamp_profile"]
    print("\nTimestamp analysis:")
    if profile is None:
        print("  No timestamp column was identified.")
    else:
        print(f"  Column: {report['timestamp_column']}")
        print(f"  Parse failures: {profile['parse_failures']}")
        print(f"  Minimum timestamp: {profile['minimum']}")
        print(f"  Maximum timestamp: {profile['maximum']}")
        print(
            "  Approximate sampling interval: "
            f"{profile['median_interval_seconds']} seconds (median)"
        )
        print(f"  Duplicate timestamps: {profile['duplicate_timestamps']}")
        print(
            "  Source order monotonic increasing: "
            f"{profile['source_order_monotonic_increasing']}"
        )
        print(f"  Irregular timestamp gaps: {profile['irregular_gap_count']}")
        print(
            "  Estimated missing intervals from long gaps: "
            f"{profile['estimated_missing_intervals']}"
        )
        for example in profile["irregular_gap_examples"]:
            print(
                "    "
                f"{example['previous_timestamp']} -> {example['next_timestamp']} "
                f"({example['gap_seconds']} seconds)"
            )


def inspect_path(
    dataset_path: str | Path,
    *,
    head_rows: int = 5,
    timestamp_column: str | None = None,
    timestamp_format: str | None = DEFAULT_TIMESTAMP_FORMAT,
    unique_threshold: int = 50,
) -> list[dict[str, object]]:
    """Inspect every CSV represented by *dataset_path* and print each report."""
    csv_files = discover_csv_files(dataset_path)
    if not csv_files:
        raise ValueError(f"No CSV files found under: {dataset_path}")

    reports = []
    for csv_file in csv_files:
        report = build_inspection_report(
            csv_file,
            timestamp_column=timestamp_column,
            timestamp_format=timestamp_format,
            unique_threshold=unique_threshold,
        )
        print_inspection_report(report, head_rows=head_rows)
        reports.append(report)
    return reports


def main() -> int:
    """Run the command-line utility and return a process exit code."""
    args = parse_args()
    try:
        inspect_path(
            args.dataset_path,
            head_rows=args.head,
            timestamp_column=args.timestamp_column,
            timestamp_format=args.timestamp_format,
            unique_threshold=args.unique_threshold,
        )
    except (FileNotFoundError, PermissionError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        print(f"Error: unable to parse CSV file: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Error: unable to read CSV file: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
