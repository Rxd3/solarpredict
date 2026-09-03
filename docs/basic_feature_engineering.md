# Day 8 basic feature engineering

## Scope

The Day 8 pipeline reads `data/processed/operational_cleaned.csv` and writes
`data/processed/operational_features_basic.csv`. It preserves all 1,009 rows and
all 14 original columns, then appends 17 safe, interpretable features.

No rolling-window, first-difference, rate-of-change, weekday, month, season, or
wind-direction cyclical features are included. No anomaly labels, synthetic
anomalies, models, or model predictions are created.

**Timezone:** Not specified by the dataset source; timestamps and derived clock
features remain timezone-naive.

## Engineered features

### Time features

| Feature | Source | Formula | Intended purpose | Limitations or assumptions |
|---|---|---|---|---|
| `hour` | `Timestamp` | Timestamp hour in `[0, 23]` | Interpretable hour-of-day context | Source timezone is unknown; raw hour has a midnight discontinuity |
| `minute` | `Timestamp` | Timestamp minute in `[0, 59]` | Preserve within-hour position | The current two-minute cadence produces even minute values only |
| `minute_of_day` | `Timestamp` | `60 * hour + minute` | Single within-day position from 0 to 1439 | Source timezone is unknown; raw value is discontinuous at midnight |
| `elapsed_minutes_from_start` | `Timestamp` | `(Timestamp - minimum Timestamp) / 60 seconds` | Relative chronological position and reproducible elapsed time | Dataset-specific origin; should not be interpreted as site age or long-term trend |
| `hour_sin` | `hour` | `sin(2*pi*hour/24)` | Circular daily encoding without a midnight jump | Uses whole-hour resolution and duplicates some information in minute-of-day encoding |
| `hour_cos` | `hour` | `cos(2*pi*hour/24)` | Companion coordinate for the circular hour encoding | Must be used with `hour_sin` to represent phase correctly |
| `minute_of_day_sin` | `minute_of_day` | `sin(2*pi*minute_of_day/1440)` | Circular time-of-day encoding at minute resolution | Timezone is unknown and the dataset contains only one complete day |
| `minute_of_day_cos` | `minute_of_day` | `cos(2*pi*minute_of_day/1440)` | Companion coordinate for minute-level daily phase | Must be used with `minute_of_day_sin` |

No weekday, month, or season features were created because approximately 33.6
hours cannot support meaningful weekly or seasonal interpretation.

### RTD sensor summaries

Let `x_1` through `x_5` denote `RTD_1` through `RTD_5` for one timestamp. Their
physical quantity, unit, and placement are **unconfirmed from available source
metadata**. The features therefore describe only cross-channel behavior.

| Feature | Source | Formula | Intended purpose | Limitations or assumptions |
|---|---|---|---|---|
| `rtd_mean` | `RTD_1`-`RTD_5` | `(x_1 + ... + x_5) / 5` | Common RTD-channel level | Can hide one disagreeing channel; no physical label is assigned |
| `rtd_std` | `RTD_1`-`RTD_5` | Population standard deviation `sqrt(sum((x_i-mean)^2)/5)` | Cross-channel dispersion | Uses `ddof=0`; sensor locations and expected tolerances are unknown |
| `rtd_min` | `RTD_1`-`RTD_5` | `min(x_1, ..., x_5)` | Lowest simultaneous channel value | Does not identify which channel supplied the minimum |
| `rtd_max` | `RTD_1`-`RTD_5` | `max(x_1, ..., x_5)` | Highest simultaneous channel value | Does not identify which channel supplied the maximum |
| `rtd_range` | `rtd_min`, `rtd_max` | `rtd_max - rtd_min` | Simple cross-channel disagreement magnitude | Expected normal range is not yet known |

All five original RTD columns remain available; the summary columns do not
replace them.

### Electrical consistency diagnostics

| Feature | Source | Formula | Intended purpose | Limitations or assumptions |
|---|---|---|---|---|
| `electrical_power_product` | `Array_Voltage`, `Array_Current` | `Array_Voltage * Array_Current` | Reproduce the recorded electrical relationship for consistency review | Units of voltage/current are unconfirmed; nearly duplicates `Power_Generated` |
| `power_consistency_residual` | `Power_Generated`, `electrical_power_product` | `Power_Generated - electrical_power_product` | Signed diagnostic showing direction of disagreement | Primarily a consistency signal, not automatically a normal model input |
| `power_consistency_abs_error` | `power_consistency_residual` | `abs(power_consistency_residual)` | Magnitude of electrical disagreement | Very small values may reflect storage or numerical rounding |

`electrical_power_product` should not automatically be supplied alongside
`Power_Generated` as an ordinary model feature. In this dataset it recreates
the recorded power almost exactly, so using both may introduce target leakage
or give one relationship excessive weight.

### Low-light context

| Feature | Source | Formula | Intended purpose | Limitations or assumptions |
|---|---|---|---|---|
| `low_light_context` | `Solar_Radiation` | `1` when `Solar_Radiation <= 5`, otherwise `0` | Distinguish descriptive low-light operating context | `5 W/m^2` is an exploratory project threshold, not a manufacturer specification or anomaly boundary |

The original `Solar_Radiation` values are unchanged. In particular, all 314
negative readings remain negative; none was clipped, replaced, or set to zero.
The flag selects 518 observations because it also includes 204 nonnegative
readings at or below 5 W/m^2.

## `Power_Generated` policy

The final anomaly-detection strategy has not been selected:

1. `Power_Generated` may be an input in a future unsupervised anomaly detector.
2. It may instead become the prediction target in a prediction-based detector.
3. Feature selection must wait until that choice is made.
4. `electrical_power_product` requires special care because it is almost
   perfectly redundant with `Power_Generated`.
5. `power_consistency_residual` is primarily intended as a diagnostic
   consistency signal.

## Actual output summary

| Item | Result |
|---|---:|
| Original columns | 14 |
| Original numerical measurements, excluding timestamp | 13 |
| Engineered features | 17 |
| Final rows | 1,009 |
| Final columns | 31 |
| New missing values | 0 |
| `low_light_context` rows | 518 (51.337958%) |

Observed engineered-feature ranges are:

| Feature | Minimum | Maximum |
|---|---:|---:|
| `hour` | 0 | 23 |
| `minute` | 0 | 58 |
| `minute_of_day` | 0 | 1,438 |
| `elapsed_minutes_from_start` | 0 | 2,016 |
| `hour_sin` | -1 | 1 |
| `hour_cos` | -1 | 1 |
| `minute_of_day_sin` | -1 | 1 |
| `minute_of_day_cos` | -1 | 1 |
| `rtd_mean` | 20.613020 | 112.205420 |
| `rtd_std` | 0.178294 | 8.330871 |
| `rtd_min` | 19.737124 | 109.622220 |
| `rtd_max` | 21.188662 | 116.766760 |
| `rtd_range` | 0.551380 | 19.111136 |
| `electrical_power_product` | 8.485233 | 438.556881 |
| `power_consistency_residual` | -0.000087646 | 0.000047115 |
| `power_consistency_abs_error` | 0.000000023 | 0.000087646 |
| `low_light_context` | 0 | 1 |

Electrical residual statistics calculated from all 1,009 rows are:

| Statistic | Value |
|---|---:|
| Minimum signed residual | -0.000087646 |
| Maximum signed residual | 0.000047115 |
| Mean signed residual | -0.000008759 |
| Median signed residual | -0.000002036 |
| Signed residual standard deviation | 0.000019202 |
| Mean absolute error | 0.000013006 |
| Median absolute error | 0.000003834 |
| Maximum absolute error | 0.000087646 |

The very small residuals are consistent with the Day 7 observation that
`Power_Generated` is almost exactly the voltage-current product. This is an
important modeling safeguard, not a newly claimed physical unit or experimental
model result.

## Validation

The generated CSV passed every implemented check:

- 1,009 input rows and all 14 original columns are preserved;
- every original value is byte-round-trip equivalent after CSV loading;
- exactly the 17 approved features are appended in the documented order;
- zero unexpected missing engineered values are present;
- all cyclical values fall within `[-1, 1]`;
- `rtd_min <= rtd_mean <= rtd_max` for every row;
- every `rtd_range` is nonnegative;
- electrical product, signed residual, and absolute residual formulas match;
- every low-light flag matches `Solar_Radiation <= 5`;
- wind-direction values are unchanged and no wind-direction cyclical feature is
  present;
- no rolling or rate-of-change feature is present;
- the cleaned input SHA-256 remained
  `3e440f11bf0a00227a4b2ae3ce29e38de0a03bb647b577daa2ff1f5d098624fc`.

No feature produced a validation failure. The two results requiring continued
caution are the 518-row breadth of the exploratory low-light flag and the
near-zero electrical residual caused by the known deterministic relationship.

## Reproduction

From the repository root with the project virtual environment active:

```powershell
python src/data_processing/engineer_basic_features.py
python -m pytest -q
```

The feature CSV is ignored by Git and can be regenerated. The versioned audit
summary is `outputs/basic_feature_summary.json`.
