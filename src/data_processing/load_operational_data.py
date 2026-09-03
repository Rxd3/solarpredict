"""Reusable, conservative loading helpers for the operational solar dataset.

The loader never writes to or edits a source CSV. Its default timestamp format
and known numerical fields reflect the inspected Kaggle dataset version 1.
Callers can override them when loading a different file.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd


DEFAULT_TIMESTAMP_COLUMN = "Timestamp"
DEFAULT_TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M"

# These fields were verified in solar_data.csv. Units are intentionally not
# encoded here because several units are not stated by the source metadata.
SELECTED_NUMERIC_COLUMNS = (
    "Air_Temp",
    "Relative_Humidity",
    "Wind_Speed",
    "Wind_Direction",
    "Solar_Radiation",
    "RTD_1",
    "RTD_2",
    "RTD_3",
    "RTD_4",
    "RTD_5",
    "Array_Voltage",
    "Array_Current",
    "Power_Generated",
)


def discover_csv_files(dataset_path: str | Path) -> list[Path]:
    """Return all CSV files represented by a file or directory path."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file but received: {path}")
        return [path]

    if not path.is_dir():
        raise ValueError(f"Expected a CSV file or directory but received: {path}")

    return sorted(candidate for candidate in path.rglob("*.csv") if candidate.is_file())


def infer_timestamp_column(columns: Iterable[object]) -> str:
    """Find an unambiguous timestamp-like column name."""
    names = [str(column) for column in columns]
    normalized = {
        name: "".join(character.lower() for character in name if character.isalnum())
        for name in names
    }
    preferred = ("timestamp", "datetime", "dateandtime", "date", "time")

    for candidate in preferred:
        matches = [name for name, value in normalized.items() if value == candidate]
        if len(matches) == 1:
            return matches[0]

    partial_matches = [
        name
        for name, value in normalized.items()
        if "timestamp" in value or "datetime" in value
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]

    raise ValueError(
        "Could not identify one timestamp column. Pass timestamp_column explicitly."
    )


def parse_timestamp_series(
    values: pd.Series,
    *,
    timestamp_format: str | None = DEFAULT_TIMESTAMP_FORMAT,
) -> pd.Series:
    """Parse timestamp values without assigning or converting a timezone."""
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = values.copy()
    else:
        parsed = pd.to_datetime(values, format=timestamp_format, errors="coerce")

    # The source timezone is unknown. Refuse to silently retain timezone-aware
    # values if a caller supplies them from elsewhere.
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise ValueError(
            "Timezone-aware timestamps are not expected for this dataset. "
            "The source timezone is not specified."
        )
    return parsed


def infer_numeric_columns(
    dataframe: pd.DataFrame,
    *,
    excluded_columns: Sequence[str] = (),
    minimum_parse_ratio: float = 0.95,
) -> list[str]:
    """Identify columns that are already numeric or overwhelmingly numeric."""
    excluded = set(excluded_columns)
    numeric_columns: list[str] = []

    for column in dataframe.columns:
        if column in excluded:
            continue
        series = dataframe[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue

        non_missing = series.dropna()
        if non_missing.empty:
            continue
        converted = pd.to_numeric(non_missing, errors="coerce")
        if float(converted.notna().mean()) >= minimum_parse_ratio:
            numeric_columns.append(column)

    return numeric_columns


def build_timestamp_profile(
    values: pd.Series,
    *,
    timestamp_format: str | None = DEFAULT_TIMESTAMP_FORMAT,
    tolerance_seconds: float = 1.0,
) -> dict[str, object]:
    """Summarize parsing, ordering, cadence, and gaps for timestamp values."""
    parsed = parse_timestamp_series(values, timestamp_format=timestamp_format)
    valid_source_order = parsed.dropna()
    sorted_unique = valid_source_order.drop_duplicates().sort_values().reset_index(drop=True)
    gaps = sorted_unique.diff().dropna()
    positive_gaps = gaps[gaps > pd.Timedelta(0)]
    median_gap = positive_gaps.median() if not positive_gaps.empty else pd.NaT
    tolerance = pd.Timedelta(seconds=tolerance_seconds)

    if pd.isna(median_gap):
        irregular_mask = pd.Series(False, index=positive_gaps.index)
    else:
        irregular_mask = (positive_gaps - median_gap).abs() > tolerance
    irregular_gaps = positive_gaps[irregular_mask]

    gap_examples: list[dict[str, object]] = []
    for index in irregular_gaps.index[:10]:
        current = sorted_unique.iloc[index]
        previous = sorted_unique.iloc[index - 1]
        duration = current - previous
        gap_examples.append(
            {
                "previous_timestamp": previous.isoformat(sep=" "),
                "next_timestamp": current.isoformat(sep=" "),
                "gap_seconds": float(duration.total_seconds()),
            }
        )

    estimated_missing_intervals = 0
    if not pd.isna(median_gap) and median_gap > pd.Timedelta(0):
        for gap in positive_gaps[positive_gaps > median_gap + tolerance]:
            estimated_missing_intervals += max(int(round(gap / median_gap)) - 1, 0)

    return {
        "parse_failures": int(parsed.isna().sum()),
        "valid_timestamp_count": int(parsed.notna().sum()),
        "minimum": (
            valid_source_order.min().isoformat(sep=" ")
            if not valid_source_order.empty
            else None
        ),
        "maximum": (
            valid_source_order.max().isoformat(sep=" ")
            if not valid_source_order.empty
            else None
        ),
        "source_order_monotonic_increasing": bool(
            valid_source_order.is_monotonic_increasing
        ),
        "backward_timestamp_steps": int((valid_source_order.diff() < pd.Timedelta(0)).sum()),
        "duplicate_timestamps": int(valid_source_order.duplicated().sum()),
        "median_interval_seconds": (
            float(median_gap.total_seconds()) if not pd.isna(median_gap) else None
        ),
        "minimum_interval_seconds": (
            float(positive_gaps.min().total_seconds()) if not positive_gaps.empty else None
        ),
        "maximum_interval_seconds": (
            float(positive_gaps.max().total_seconds()) if not positive_gaps.empty else None
        ),
        "irregular_gap_count": int(len(irregular_gaps)),
        "estimated_missing_intervals": int(estimated_missing_intervals),
        "irregular_gap_examples": gap_examples,
    }


def load_operational_csv(
    csv_path: str | Path,
    *,
    timestamp_column: str | None = None,
    timestamp_format: str | None = DEFAULT_TIMESTAMP_FORMAT,
    numeric_columns: Sequence[str] | None = None,
    sort_chronologically: bool = True,
    drop_duplicate_rows: bool = True,
    duplicate_keep: str | bool = "first",
) -> pd.DataFrame:
    """Load and minimally normalize one operational CSV.

    Exact duplicate rows may be removed, but no missing, invalid, out-of-range,
    or anomalous measurements are filtered. Numeric coercion uses ``NaN`` to
    expose invalid text rather than hiding it.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a CSV file but received: {path}")

    dataframe = pd.read_csv(path)
    resolved_timestamp_column = timestamp_column or infer_timestamp_column(dataframe.columns)
    if resolved_timestamp_column not in dataframe.columns:
        raise ValueError(f"Timestamp column not found: {resolved_timestamp_column}")

    duplicate_rows = int(dataframe.duplicated().sum())
    if drop_duplicate_rows:
        dataframe = dataframe.drop_duplicates(keep=duplicate_keep).copy()
    else:
        dataframe = dataframe.copy()

    dataframe[resolved_timestamp_column] = parse_timestamp_series(
        dataframe[resolved_timestamp_column], timestamp_format=timestamp_format
    )

    resolved_numeric_columns = (
        list(numeric_columns)
        if numeric_columns is not None
        else infer_numeric_columns(
            dataframe, excluded_columns=(resolved_timestamp_column,)
        )
    )
    missing_numeric_columns = [
        column for column in resolved_numeric_columns if column not in dataframe.columns
    ]
    if missing_numeric_columns:
        raise ValueError(
            "Numeric columns not found: " + ", ".join(missing_numeric_columns)
        )

    for column in resolved_numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    if sort_chronologically:
        dataframe = dataframe.sort_values(
            resolved_timestamp_column, kind="stable", na_position="last"
        ).reset_index(drop=True)

    dataframe.attrs.update(
        {
            "source_path": str(path.resolve()),
            "timestamp_column": resolved_timestamp_column,
            "timestamp_format": timestamp_format,
            "timezone": "not specified by source; timestamps kept timezone-naive",
            "numeric_columns": resolved_numeric_columns,
            "duplicate_rows_found": duplicate_rows,
            "duplicate_rows_removed": duplicate_rows if drop_duplicate_rows else 0,
            "timestamp_parse_failures": int(
                dataframe[resolved_timestamp_column].isna().sum()
            ),
        }
    )
    return dataframe
