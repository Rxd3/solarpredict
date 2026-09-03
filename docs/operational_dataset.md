# Operational dataset: verified schema and EDA record

## Provenance

- **Dataset:** Solar Power Dataset
- **Creator:** Matthews Ankon Baroi (`s1nister` on Kaggle)
- **Source:** <https://www.kaggle.com/datasets/s1nister/solar-power-generation-dataset>
- **Downloaded version:** 1 (initial release)
- **License:** CC0: Public Domain
- **Downloaded file:** `data/operational/solar-power-dataset/solar_data.csv`
- **File size:** 138,228 bytes
- **SHA-256:** `0615c4d92c7869074f99c4a6552cc201ab6db8152fc77650758664ed2c4ba393`

The raw CSV is intentionally excluded from Git. This document describes the
file downloaded and inspected on 2026-09-03; a later source version must be
re-inspected rather than assumed to have the same schema.

## Verified file-level schema

- **Shape:** 1,009 rows x 14 columns
- **Raw timestamp format:** `%d-%m-%Y %H:%M`
- **Date range:** 2022-04-27 15:32 through 2022-04-29 01:08
- **Observed sampling interval:** exactly 120 seconds between every consecutive
  timestamp in this file (median, minimum, and maximum are all 120 seconds)
- **Timestamp parse failures:** 0
- **Irregular timestamp gaps:** 0
- **Duplicate timestamps:** 0
- **Duplicate complete rows:** 0
- **Missing values:** 0 in every column
- **Invalid non-numeric values in expected numeric columns:** 0

**Timezone: Not specified by dataset source; timestamps kept timezone-naive.**
No UTC or geographical timezone is inferred.

## Columns

Units appear only where the Kaggle source description explicitly states them.
The raw pandas dtypes below were observed with pandas 3.0.5. The loader parses
`Timestamp` to a timezone-naive pandas datetime dtype (`datetime64[us]` in the
current environment) and conservatively coerces the remaining measurement
fields to numeric values. Datetime resolution can vary across compatible pandas
versions without changing the timestamp values.

| Original column | Interpreted meaning | Raw inferred dtype | Confirmed unit | Likely useful for later anomaly detection? |
|---|---|---:|---|---|
| `Timestamp` | Observation date and time | `str` | Not applicable; timezone unavailable | Yes - temporal index and cadence checks |
| `Air_Temp` | Air temperature | `float64` | degrees Celsius | Yes - environmental context |
| `Relative_Humidity` | Relative humidity | `float64` | `%RH` | Yes - environmental context |
| `Wind_Speed` | Wind speed | `float64` | `m/s` | Potentially - environmental context |
| `Wind_Direction` | Wind direction | `float64` | degrees | Potentially - environmental context; circular handling belongs to a later feature-engineering stage |
| `Solar_Radiation` | Solar radiation incident measurement | `float64` | `W/m^2` | Yes - key input for expected generation |
| `RTD_1` | RTD sensor channel 1; exact placement and measured quantity are not defined by the source | `float64` | Not specified | Potentially - retain pending sensor clarification |
| `RTD_2` | RTD sensor channel 2; exact placement and measured quantity are not defined by the source | `float64` | Not specified | Potentially - retain pending sensor clarification |
| `RTD_3` | RTD sensor channel 3; exact placement and measured quantity are not defined by the source | `float64` | Not specified | Potentially - retain pending sensor clarification |
| `RTD_4` | RTD sensor channel 4; exact placement and measured quantity are not defined by the source | `float64` | Not specified | Potentially - retain pending sensor clarification |
| `RTD_5` | RTD sensor channel 5; exact placement and measured quantity are not defined by the source | `float64` | Not specified | Potentially - retain pending sensor clarification |
| `Array_Voltage` | Solar-array voltage reading | `float64` | Not specified | Yes - direct electrical behavior |
| `Array_Current` | Solar-array current reading | `float64` | Not specified | Yes - direct electrical behavior |
| `Power_Generated` | Generated electrical power | `float64` | `W` | Yes - key operational output; deterministic relationships with voltage/current must be considered later |

The source states that generated power is available from the voltage and current
readings, but it does not explicitly state the units for `Array_Voltage` or
`Array_Current`. Their units are therefore left undocumented rather than assumed.

## Numerical ranges observed

| Column | Minimum | Median | Maximum |
|---|---:|---:|---:|
| `Air_Temp` | 12.075478 | 19.883582 | 49.631508 |
| `Relative_Humidity` | 10.648168 | 34.883796 | 51.489112 |
| `Wind_Speed` | 0.000000 | 0.000000 | 2.933334 |
| `Wind_Direction` | 1.577505 | 212.511440 | 385.630880 |
| `Solar_Radiation` | -0.838326 | 1.765152 | 1132.128400 |
| `RTD_1` | 19.737124 | 27.453532 | 116.766760 |
| `RTD_2` | 21.188662 | 28.789080 | 111.630940 |
| `RTD_3` | 20.738694 | 28.103806 | 109.622220 |
| `RTD_4` | 20.615538 | 27.971720 | 111.680800 |
| `RTD_5` | 20.744964 | 28.191964 | 111.628120 |
| `Array_Voltage` | 1.574978 | 51.296516 | 81.778936 |
| `Array_Current` | 5.351294 | 5.391468 | 5.407162 |
| `Power_Generated` | 8.485232 | 276.001580 | 438.556840 |

## Data-quality findings

The automated report is stored in
`outputs/operational_data_quality.json`. It flags observations but does not
alter or delete them.

- `Solar_Radiation` has 314 values below 0 `W/m^2`; the minimum is -0.838326.
  These mostly small negative readings may reflect sensor offset or nighttime
  behavior, but their cause is not established. They remain in the raw data.
- `Wind_Direction` has 10 values above 360 degrees; the maximum is 385.630880.
  These are range violations and remain available for later investigation.
- No values violated the conservative checks for relative humidity, negative
  wind speed, temperature below absolute zero, or negative generated power.
- No missing values, exact duplicate rows, invalid numeric strings, timestamp
  parsing failures, ordering reversals, duplicate timestamps, or sampling gaps
  were found.

These range flags are data-quality observations, not anomaly labels.

## Exploratory observations

- The file spans only 33 hours 36 minutes: one complete calendar day
  (2022-04-28) plus partial boundary days. Daily behavior shown here should not
  be treated as a seasonal or long-term profile.
- Generated power ranges from 8.485232 to 438.556840 W and follows a clear
  day/night pattern in this short window.
- Pearson correlation between `Power_Generated` and `Solar_Radiation` is about
  0.730. Correlation is descriptive for this short interval and is not evidence
  of causality.
- `Power_Generated` and `Array_Voltage` have Pearson correlation about 0.99999.
  Also, `Array_Voltage * Array_Current` reproduces `Power_Generated` with a
  maximum absolute difference below 0.000088 in the downloaded file. This near
  deterministic relationship is important for avoiding target leakage in a
  later modeling stage.
- `Array_Current` varies over a narrow observed range (5.351294 to 5.407162)
  compared with the large change in `Array_Voltage`. No fault conclusion is
  drawn from this pattern.

## Reproduction commands

```powershell
python src/data_processing/inspect_dataset.py data/operational/solar-power-dataset
python src/data_processing/analyze_operational_data.py data/operational/solar-power-dataset
python -m pytest -q
```

The source CSV must remain untouched. Loader sorting, duplicate handling, and
numeric coercion occur only in returned in-memory DataFrames.
