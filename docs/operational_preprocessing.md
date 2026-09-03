# Operational data preprocessing

## Scope

This document records the Day 6 preprocessing applied to the verified Kaggle
`solar_data.csv` file. The goal is a reliable, minimally transformed input for
later work—not anomaly detection, feature engineering, resampling, or outlier
removal.

### Files

- **Raw input:** `data/operational/solar-power-dataset/solar_data.csv`
- **Processed output:** `data/processed/operational_cleaned.csv`
- **Machine-readable report:** `outputs/preprocessing_summary.json`
- **Pipeline:** `src/data_processing/preprocess_operational_data.py`
- **Raw SHA-256 before and after:**
  `0615c4d92c7869074f99c4a6552cc201ab6db8152fc77650758664ed2c4ba393`
- **Processed SHA-256:**
  `3e440f11bf0a00227a4b2ae3ce29e38de0a03bb647b577daa2ff1f5d098624fc`

The matching raw hashes prove that the source file was not modified by this
pipeline run.

## Actual before/after summary

| Check | Raw Data | Processed Data |
|---|---:|---:|
| Rows | 1,009 | 1,009 |
| Columns | 14 | 14 |
| Missing values | 0 | 0 |
| Exact duplicate rows | 0 | 0 |
| Duplicate timestamps | 0 | 0 |
| Timestamp parse failures | 0 | 0 |
| Chronologically ordered | Yes | Yes |
| Irregular timestamp gaps | 0 | 0 |
| Median sampling interval | 120 seconds | 120 seconds |

No rows disappeared because the raw file contained no exact duplicates.

## Preprocessing rules and rationale

1. **Validate the verified schema before processing.**
   The input must contain `Timestamp` and all 13 measurement columns documented
   in `docs/operational_dataset.md`. Failing early prevents a new dataset
   version from being silently processed with an outdated schema.
2. **Parse timestamps with the observed source format.**
   `Timestamp` is parsed using `%d-%m-%Y %H:%M`. The processed CSV writes it as
   `%Y-%m-%d %H:%M:%S`, which is unambiguous while preserving the same instant.
3. **Keep timestamps timezone-naive.**
   The dataset source does not specify a timezone. The pipeline does not assume
   UTC or any geographical timezone.
4. **Remove exact duplicate rows only.**
   If exact duplicates occur, the first copy is retained and the removal count
   is recorded. Rows that share a timestamp but contain different measurements
   are preserved and reported for manual review. The actual file contained no
   duplicates of either kind.
5. **Stable-sort chronologically.**
   Sorting uses a stable algorithm, so equal timestamps retain their original
   relative order. The actual input was already ordered, so its sampling order
   did not change.
6. **Coerce the verified measurement columns to numeric dtypes.**
   The following columns are converted with invalid text becoming a visible
   missing value rather than causing silent string arithmetic:
   `Air_Temp`, `Relative_Humidity`, `Wind_Speed`, `Wind_Direction`,
   `Solar_Radiation`, `RTD_1`, `RTD_2`, `RTD_3`, `RTD_4`, `RTD_5`,
   `Array_Voltage`, `Array_Current`, and `Power_Generated`.
   The actual file contained zero invalid numeric strings.
7. **Preserve missing values conservatively.**
   Rows are not dropped and values are not imputed. Unparseable numeric values
   would remain as missing and be reported. The actual input and output both
   contain zero missing values.
8. **Preserve unusual or range-flagged measurements.**
   No clipping, filtering, smoothing, or outlier removal is performed. This is
   essential because unusual observations may be relevant to later anomaly
   detection.
9. **Preserve the complete 14-column schema and column order.**
   No operational measurement is discarded and no modeling feature is added at
   this stage.
10. **Write outputs atomically and validate afterward.**
    Temporary files are replaced only after complete writes. The processed data
    is then compared with the exact expected result of parsing, exact-row
    deduplication, numeric coercion, and stable chronological sorting.

## Intentionally not changed

- The 314 `Solar_Radiation` values below 0 W/m² remain unchanged. Their cause is
  not established, and they are not automatically treated as anomalies.
- The 10 `Wind_Direction` values outside 0–360 degrees remain unchanged. They
  are not wrapped, clipped, or removed.
- Missing values are not interpolated or filled.
- Timestamp gaps are not resampled or filled; none were present in this file.
- Measurements are not scaled, normalized, smoothed, or aggregated.
- No rows are removed based on statistical rarity or numerical range.
- No feature engineering, synthetic anomalies, anomaly labels, or models are
  created.

## Validation performed

The pipeline verified all of the following for the actual output:

- raw SHA-256 unchanged;
- processed row count equals raw rows minus exact duplicates;
- every raw and required column is present in the original order;
- no unexpected columns were added;
- every processed timestamp parses and is monotonically increasing;
- minimum and maximum of every numerical measurement are preserved;
- the processed content exactly matches the documented transformations, within
  a tight floating-point CSV round-trip tolerance;
- no unexpectedly large row loss occurred (actual loss: 0%).

All validation checks passed.

## Reproduction

From the project root with the virtual environment active:

```powershell
python src/data_processing/preprocess_operational_data.py
python -m pytest -q
```

The processed CSV is excluded from Git because it is reproducible from the raw
file and pipeline. The summary JSON and this document retain the audit trail.
