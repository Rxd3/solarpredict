# Day 10 change and rate-of-change feature engineering

## Scope

The Day 10 pipeline reads `data/processed/operational_features_rolling.csv` and
writes `data/processed/operational_features_change.csv`. It preserves all 1,009
rows and all 61 Day 9 columns verbatim, then appends 20 causal change features.
The final dataset has 81 columns.

No feature selection, synthetic anomaly generation, anomaly detection,
computer vision, video processing, or dashboard work is performed here.

## Causal calculation policy

Every first difference uses `current value - previous value`. Each rate is that
difference divided by the verified two-minute interval. Relative changes use
only the same current and previous pair. The 30-minute baselines were created in
Day 9 from the current row and 14 earlier rows with `center=False`.

No operation reads a future row. The first row therefore has no previous value
and retains `NaN` for all first-difference and rate features.

## First differences and per-minute rates

Differences can expose sudden transitions that are less obvious in absolute
levels. Dividing by elapsed time expresses the same transition per minute,
which can later help compare how quickly operational signals move. In this file,
the rate calculation assumes the verified 120-second cadence; a future dataset
with gaps or variable intervals must validate or calculate elapsed time directly.

| Feature | Source | Formula | Intended purpose | Expected behavior and limitations |
|---|---|---|---|---|
| `power_generated_diff` | `Power_Generated` | `P_t - P_(t-1)` | Signed two-minute generated-power transition | First row is NaN; large legitimate operating transitions are possible |
| `solar_radiation_diff` | `Solar_Radiation` | `S_t - S_(t-1)` | Signed two-minute radiation transition | Sensitive to fast illumination changes and retained negative readings |
| `air_temp_diff` | `Air_Temp` | `T_t - T_(t-1)` | Short environmental-temperature change | May emphasize sensor noise; temperature unit is confirmed but timezone is not |
| `relative_humidity_diff` | `Relative_Humidity` | `H_t - H_(t-1)` | Short humidity change | Environmental movement is contextual, not automatically abnormal |
| `array_voltage_diff` | `Array_Voltage` | `V_t - V_(t-1)` | Electrical voltage transition | Voltage unit is unconfirmed; strongly related to generated power |
| `array_current_diff` | `Array_Current` | `I_t - I_(t-1)` | Electrical current transition | Current unit is unconfirmed and its observed range is narrow |
| `rtd_mean_diff` | `rtd_mean` | `R_t - R_(t-1)` | Change in the five-channel RTD summary | RTD meaning, placement, and units remain unconfirmed |
| `power_generated_rate_per_min` | `power_generated_diff` | `power_generated_diff / 2` | Generated-power change per minute | Assumes every interval is exactly two minutes |
| `solar_radiation_rate_per_min` | `solar_radiation_diff` | `solar_radiation_diff / 2` | Radiation change per minute | Same fixed-cadence assumption; not an anomaly score |
| `air_temp_rate_per_min` | `air_temp_diff` | `air_temp_diff / 2` | Temperature change per minute | Two-minute changes may be noisy |
| `relative_humidity_rate_per_min` | `relative_humidity_diff` | `relative_humidity_diff / 2` | Humidity change per minute | Does not establish a physical cause |
| `array_voltage_rate_per_min` | `array_voltage_diff` | `array_voltage_diff / 2` | Voltage change per minute | Unit is source units per minute until confirmed |
| `array_current_rate_per_min` | `array_current_diff` | `array_current_diff / 2` | Current change per minute | Narrow input range makes small numerical changes prominent |
| `rtd_mean_rate_per_min` | `rtd_mean_diff` | `rtd_mean_diff / 2` | RTD-summary change per minute | Unit and physical interpretation remain unconfirmed |

## Safeguarded relative changes

Ordinary relative change divides by the previous value. At zero or near zero,
that calculation is undefined or can produce a misleadingly huge magnitude.
Unsafe denominators therefore produce `NaN`; infinity is never replaced by an
arbitrary finite number.

| Feature | Source | Formula when safe | Denominator safeguard | Purpose and limitation |
|---|---|---|---|---|
| `power_generated_relative_change` | `Power_Generated` | `(P_t - P_(t-1)) / P_(t-1)` | Require `abs(P_(t-1)) >= 1 W` | Dimensionless relative power transition; threshold is a project numerical floor |
| `solar_radiation_relative_change` | `Solar_Radiation` | `(S_t - S_(t-1)) / S_(t-1)` | Require `abs(S_(t-1)) >= 5 W/m^2` | Dimensionless radiation transition; the 5 W/m^2 floor is exploratory, not a manufacturer specification |

The solar safeguard deliberately masks many night/low-light rows. Near zero,
small measurement noise or a sign change would dominate the ratio and make it
difficult to interpret. The original radiation values, including all negative
measurements, remain unchanged.

## Deviations from the 30-minute trailing baseline

A local deviation describes how far the current observation is from its recent
operating context. A large value may later be useful as evidence of local
unusual behavior, but it is not an anomaly label. The existing trailing mean
includes the current observation, so the feature is causal and somewhat less
extreme than a baseline ending at the previous row.

| Feature | Source | Formula | Intended purpose | Expected behavior and limitations |
|---|---|---|---|---|
| `power_deviation_from_30m_mean` | `Power_Generated`, `power_generated_roll_mean_30m` | `P_t - power_mean_30m_t` | Signed local power deviation | First 14 rows are NaN because the full baseline is unavailable |
| `solar_deviation_from_30m_mean` | `Solar_Radiation`, `solar_radiation_roll_mean_30m` | `S_t - solar_mean_30m_t` | Signed local radiation deviation | First 14 rows are NaN; legitimate fast irradiance movement can be large |
| `power_deviation_ratio_30m` | Power deviation, power 30-minute mean | `(P_t - mean_t) / mean_t` | Scale-relative local power deviation | Require `abs(power mean) >= 1 W`; otherwise NaN |
| `solar_deviation_ratio_30m` | Solar deviation, solar 30-minute mean | `(S_t - mean_t) / mean_t` | Scale-relative local radiation deviation | Require `abs(solar mean) >= 5 W/m^2`; otherwise NaN, especially in low light |

## Actual missing-value results

| Feature group | Features | NaNs per feature | Reason |
|---|---:|---:|---|
| First differences | 7 | 1 | No previous observation for the first row |
| Per-minute rates | 7 | 1 | Derived from the first differences |
| Power relative change | 1 | 1 | First-row previous value is unavailable; later power denominators exceed 1 W |
| Solar relative change | 1 | 518 | First row plus previous radiation values below the 5 W/m^2 absolute floor |
| Absolute 30-minute deviations | 2 | 14 | Day 9 rolling baseline warm-up |
| Power 30-minute deviation ratio | 1 | 14 | Same rolling warm-up; later baseline exceeds 1 W |
| Solar 30-minute deviation ratio | 1 | 520 | Rolling warm-up plus near-zero solar baselines below 5 W/m^2 |

Across the 20 new columns there are 1,095 NaN cells. They are expected results
of unavailable history or explicit safeguards. No row is dropped, no NaN is
backfilled, and no infinity is generated.

## Largest observed transitions

These are descriptive **largest observed transitions**, not anomaly labels.
Changes are measured over one two-minute interval.

### `Power_Generated`

| Rank | Timestamp | Signed power change | Absolute change | Simultaneous radiation change | Low-light context |
|---:|---|---:|---:|---:|---:|
| 1 | 2022-04-27 17:00 | -125.304960 | 125.304960 | 14.622780 | 0 |
| 2 | 2022-04-28 17:00 | -103.485460 | 103.485460 | 39.060360 | 0 |
| 3 | 2022-04-27 17:52 | -70.362020 | 70.362020 | -0.700040 | 0 |
| 4 | 2022-04-27 17:50 | 70.268020 | 70.268020 | -3.617920 | 0 |
| 5 | 2022-04-28 05:28 | 58.708580 | 58.708580 | 0.000000 | 1 |
| 6 | 2022-04-28 19:14 | -52.167420 | 52.167420 | -0.405335 | 1 |
| 7 | 2022-04-27 19:14 | -51.060680 | 51.060680 | -0.555786 | 1 |
| 8 | 2022-04-28 05:26 | 51.049912 | 51.049912 | 0.000000 | 1 |
| 9 | 2022-04-27 19:16 | -50.387180 | 50.387180 | 0.687874 | 1 |
| 10 | 2022-04-28 05:30 | 49.282640 | 49.282640 | 0.152222 | 1 |

### `Solar_Radiation`

| Rank | Timestamp | Signed radiation change | Absolute change | Simultaneous power change | Low-light context |
|---:|---|---:|---:|---:|---:|
| 1 | 2022-04-28 09:58 | 199.305080 | 199.305080 | 0.591380 | 0 |
| 2 | 2022-04-28 10:00 | 180.418080 | 180.418080 | -3.995700 | 0 |
| 3 | 2022-04-28 09:52 | 104.300180 | 104.300180 | 1.235440 | 0 |
| 4 | 2022-04-27 17:22 | -76.284300 | 76.284300 | -0.927720 | 0 |
| 5 | 2022-04-28 13:14 | 71.248600 | 71.248600 | 2.353480 | 0 |
| 6 | 2022-04-28 13:32 | 60.825000 | 60.825000 | 0.610920 | 0 |
| 7 | 2022-04-28 12:50 | -54.514800 | 54.514800 | -1.901320 | 0 |
| 8 | 2022-04-28 15:50 | -54.413920 | 54.413920 | -0.691080 | 0 |
| 9 | 2022-04-28 14:50 | -51.466880 | 51.466880 | -2.936360 | 0 |
| 10 | 2022-04-28 10:06 | 47.061280 | 47.061280 | 46.828400 | 0 |

The two top-ten lists share no timestamp. Same-step power and radiation
differences have Pearson correlation -0.017771 in this short file. The two
largest power drops recur at 17:00 on consecutive dates without a top-ranked
simultaneous radiation change. The 17:50 increase followed by the 17:52 decrease
on April 27 is a short power reversal with little radiation movement. Six of the
ten largest power changes occur during an already-active low-light context, but
none coincides with a change in the low-light flag. The largest radiation
transitions occur in daylight context; most have small same-step power changes,
although 10:06 has a simultaneous 46.828400 W power increase.

These observations suggest several step-like or transition-period operating
patterns, but the dataset does not establish their mechanism or cause.

## Validation

All validation checks passed:

- input shape 1,009 x 61 and output shape 1,009 x 81;
- every previous CSV field is preserved verbatim;
- all 20 formulas independently reproduce the saved values;
- all rates equal their differences divided by two minutes;
- every unsafe relative or normalized denominator produces NaN;
- no positive or negative infinity exists in any new feature;
- the 30-minute deviations match their existing trailing baselines;
- tests verify that changing a future row cannot change an earlier feature;
- raw, cleaned, Day 8, and Day 9 datasets retain their previous SHA-256 hashes.

The complete statistics, transitions, safeguards, hashes, and validation
details are stored in `outputs/change_feature_summary.json`.

## Limitations

- The file spans only 33.6 hours, so large observed changes cannot be assigned a
  reliable long-term frequency.
- Differencing can amplify sensor noise as well as meaningful transitions.
- Fixed per-minute rates rely on the verified regular cadence in this file.
- Relative solar behavior during low light is deliberately unavailable rather
  than numerically exaggerated.
- Generated power is nearly determined by voltage and current, so later feature
  review must address redundancy before modeling.
- None of these features constitutes an anomaly decision.

## Reproduction

From the project root with the virtual environment active:

```powershell
python src/data_processing/engineer_change_features.py
python -m pytest -q
```

The generated CSV is ignored by Git and reproducible from the Day 9 dataset.
