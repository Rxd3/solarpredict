# Day 9 trailing rolling-window feature engineering

## Scope

The Day 9 pipeline reads `data/processed/operational_features_basic.csv` and
writes `data/processed/operational_features_rolling.csv`. All 1,009 rows and all
31 previous columns are preserved verbatim. Thirty rolling features are
appended, producing a 1,009-row by 61-column dataset.

Only rolling-window statistics are implemented here. There are no difference,
rate-of-change, percentage-change, synthetic-anomaly, anomaly-label, model,
computer-vision, video, or dashboard outputs.

## Why rolling statistics are useful

A single measurement shows the current state but not its recent context.
Trailing means summarize the recent level, standard deviations summarize local
variability, and minima/maxima describe the recent envelope. These quantities
can later help distinguish a sustained operating change from a short
fluctuation. They are descriptive inputs only and are not anomaly decisions.

The rolling standard deviations use `ddof=0`, so each value is the population
standard deviation of the observations in its complete trailing window.

## Window definitions

The verified sampling interval is exactly two minutes, so the requested time
windows map to fixed sample counts:

| Window | Samples | Rows required at the current timestamp | Initial NaNs per feature |
|---|---:|---|---:|
| 10 minutes | 5 | Current row plus 4 previous rows | 4 |
| 30 minutes | 15 | Current row plus 14 previous rows | 14 |
| 60 minutes | 30 | Current row plus 29 previous rows | 29 |

The three windows provide short, medium, and longer local context while staying
small relative to the approximately 33.6-hour file. They are project choices,
not manufacturer specifications.

Every operation uses pandas rolling configuration equivalent to:

```python
series.rolling(window=samples, min_periods=samples, center=False)
```

`center=False` is essential: the feature at row `t` uses only row `t` and
earlier rows. It never accesses later observations. Centered windows would leak
future measurements into the features and make later real-time evaluation
invalid.

## Exact rolling features

For `Power_Generated` and `Solar_Radiation`, mean, population standard
deviation, minimum, and maximum are calculated at all three windows:

```text
power_generated_roll_mean_10m
power_generated_roll_std_10m
power_generated_roll_min_10m
power_generated_roll_max_10m
power_generated_roll_mean_30m
power_generated_roll_std_30m
power_generated_roll_min_30m
power_generated_roll_max_30m
power_generated_roll_mean_60m
power_generated_roll_std_60m
power_generated_roll_min_60m
power_generated_roll_max_60m
solar_radiation_roll_mean_10m
solar_radiation_roll_std_10m
solar_radiation_roll_min_10m
solar_radiation_roll_max_10m
solar_radiation_roll_mean_30m
solar_radiation_roll_std_30m
solar_radiation_roll_min_30m
solar_radiation_roll_max_30m
solar_radiation_roll_mean_60m
solar_radiation_roll_std_60m
solar_radiation_roll_min_60m
solar_radiation_roll_max_60m
```

The selected contextual features are:

```text
air_temp_roll_mean_30m
air_temp_roll_mean_60m
relative_humidity_roll_mean_30m
relative_humidity_roll_mean_60m
rtd_mean_roll_mean_30m
rtd_mean_roll_std_30m
```

This gives 8 ten-minute features, 12 thirty-minute features, and 10 sixty-minute
features.

## Why the feature set is limited

- `Power_Generated` is the main operational output and `Solar_Radiation` is the
  main available-light measurement, so both receive full rolling summaries.
- `Air_Temp` and `Relative_Humidity` receive only medium and longer means for
  environmental context.
- The five uncertain RTD channels were already condensed into `rtd_mean`; only
  its 30-minute mean and variability are added.
- Rolling features are not added for every original or Day 8 column. This avoids
  unnecessary feature growth and highly repetitive inputs.
- `Array_Voltage` and `Array_Current` are not expanded because their product is
  already nearly identical to `Power_Generated`.
- `Wind_Direction` remains untouched because values above 360 are unresolved
  and circular encoding has not been justified.

All negative `Solar_Radiation` values and every previous feature remain
unchanged.

## Initial missing values

Full windows are mandatory. Before enough historical observations exist, the
rolling value is therefore undefined:

| Window | New features | NaNs per feature | Total new NaN cells |
|---|---:|---:|---:|
| 10 minutes | 8 | 4 | 32 |
| 30 minutes | 12 | 14 | 168 |
| 60 minutes | 10 | 29 | 290 |
| **Total** | **30** | — | **490** |

These NaNs are confined to the expected initial prefixes. They were not
backfilled, interpolated, filled from future values, or removed, and the 1,009
rows remain intact. A later modeling stage must define a transparent handling
policy, such as starting evaluation after the longest warm-up period.

## Observed rolling behavior

- The 30-minute power mean visibly lags rapid dawn/dusk-like changes. This is
  expected for a trailing smoother and confirms that it is not centered.
- Radiation rolling minima and some rolling means remain negative in low-light
  periods because the original negative measurements are intentionally
  preserved. For example, the minimum observed 60-minute radiation rolling mean
  is -0.329725 W/m^2.
- Wider windows can show larger peak standard deviations when they span sharp
  transitions. Maximum power rolling standard deviation increases from
  70.543383 W at 10 minutes to 122.804164 W at 60 minutes.
- No mathematically unexpected rolling behavior or validation failure remains.

The requested visual comparison is
[`day9_power_rolling_mean.png`](../outputs/figures/operational/day9_power_rolling_mean.png).
It compares `Power_Generated` with its 30-minute trailing mean and displays the
expected smoothing and delay at rapid transitions.

## Validation results

All automated validation checks passed:

- input shape 1,009 x 31 and output shape 1,009 x 61;
- all 31 prior columns and their CSV field values are unchanged;
- all 30 approved rolling columns are present in the documented order;
- every rolling calculation matches an independent trailing recomputation;
- all 4-, 14-, and 29-row initial NaN prefixes match their full windows;
- every primary rolling minimum is at or below its mean, which is at or below
  its maximum;
- no protected source file changed.

Protected SHA-256 values before and after the run were identical:

| Protected file | SHA-256 |
|---|---|
| Raw operational CSV | `0615c4d92c7869074f99c4a6552cc201ab6db8152fc77650758664ed2c4ba393` |
| Cleaned operational CSV | `3e440f11bf0a00227a4b2ae3ce29e38de0a03bb647b577daa2ff1f5d098624fc` |
| Day 8 basic feature CSV | `99f9f3a61f4602d518b6b2fa024f415f95acbd988cd9a4e5588f62e1f19dbcd7` |

The complete ranges, descriptive statistics, per-feature NaN counts, formulas,
and validation evidence are stored in
`outputs/rolling_feature_summary.json`.

## Short-coverage limitation

The dataset contains only about 33.6 hours and one complete calendar day.
Rolling rows overlap heavily, so 1,009 rolling observations do not represent
1,009 independent operating periods. The file is suitable for verifying feature
logic, but not for establishing robust seasonal thresholds or evaluating
generalization across many days.

## Reproduction

From the project root with the virtual environment active:

```powershell
python src/data_processing/engineer_rolling_features.py
python -m pytest -q
```

The rolling CSV and validation figure are ignored by Git and reproducible from
the versioned script. The summary JSON and this document provide the audit
trail.
