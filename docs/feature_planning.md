# Day 7 operational data quality review and feature plan

## Scope and evidence

This review uses `data/processed/operational_cleaned.csv`. The file contains
1,009 observations, one timestamp, and 13 numerical measurements. It was
reviewed without changing values, creating features, assigning anomaly labels,
or training a model. The complete machine-readable evidence, including all 314
negative-radiation timestamps, every correlation, and the full neighboring rows
around unusual wind-direction readings, is in
`outputs/operational_quality_review.json`.

The dataset source does not specify a timezone. Timestamps remain timezone-naive
and are not treated as UTC or assigned a geographical timezone.

## Actual schema and observed statistics

Statistics below were calculated from the processed file. Units are included
only where the available source metadata confirms them.

| Exact column | Processed dtype | Observed min | Observed max | Mean | Median | Likely role | Meaning understood? | Unit verified? | Potential anomaly-detection use |
|---|---|---:|---:|---:|---:|---|---|---|---|
| `Timestamp` | `datetime64[us]` | 2022-04-27 15:32 | 2022-04-29 01:08 | N/A | N/A | Temporal index, ordering, and window definition | Yes, but timezone unavailable | N/A; timezone not specified | Essential temporal context |
| `Air_Temp` | `float64` | 12.075478 | 49.631508 | 25.353897 | 19.883582 | Environmental temperature context | Yes | Yes: degrees Celsius | Yes; generation and sensor behavior depend on environment |
| `Relative_Humidity` | `float64` | 10.648168 | 51.489112 | 31.806119 | 34.883796 | Environmental moisture context | Yes | Yes: `%RH` | Yes; contextual and change-based behavior |
| `Wind_Speed` | `float64` | 0.000000 | 2.933334 | 0.179716 | 0.000000 | Environmental and cooling context | Yes | Yes: `m/s` | Potentially useful as context |
| `Wind_Direction` | `float64` | 1.577505 | 385.630880 | 210.653877 | 212.511440 | Circular wind context | Mostly; values above 360 need source clarification | Yes: degrees; encoding above 360 is unconfirmed | Potentially useful after a documented encoding decision |
| `Solar_Radiation` | `float64` | -0.838326 | 1132.128400 | 224.051042 | 1.765152 | Primary available-light input | Yes; negative-reading cause is unconfirmed | Yes: `W/m^2` | Essential for expected-power context |
| `RTD_1` | `float64` | 19.737124 | 116.766760 | 47.160849 | 27.453532 | Unknown RTD sensor channel | **Unconfirmed from available source metadata.** | **Unconfirmed from available source metadata.** | Potentially useful; retain pending sensor clarification |
| `RTD_2` | `float64` | 21.188662 | 111.630940 | 46.948065 | 28.789080 | Unknown RTD sensor channel | **Unconfirmed from available source metadata.** | **Unconfirmed from available source metadata.** | Potentially useful; retain pending sensor clarification |
| `RTD_3` | `float64` | 20.738694 | 109.622220 | 45.755955 | 28.103806 | Unknown RTD sensor channel | **Unconfirmed from available source metadata.** | **Unconfirmed from available source metadata.** | Potentially useful; retain pending sensor clarification |
| `RTD_4` | `float64` | 20.615538 | 111.680800 | 46.669776 | 27.971720 | Unknown RTD sensor channel | **Unconfirmed from available source metadata.** | **Unconfirmed from available source metadata.** | Potentially useful; retain pending sensor clarification |
| `RTD_5` | `float64` | 20.744964 | 111.628120 | 46.975862 | 28.191964 | Unknown RTD sensor channel | **Unconfirmed from available source metadata.** | **Unconfirmed from available source metadata.** | Potentially useful; retain pending sensor clarification |
| `Array_Voltage` | `float64` | 1.574978 | 81.778936 | 34.105456 | 51.296516 | Direct electrical measurement | Yes at a general level | **Unconfirmed from available source metadata.** | Yes; direct operating behavior and consistency checks |
| `Array_Current` | `float64` | 5.351294 | 5.407162 | 5.383970 | 5.391468 | Direct electrical measurement | Yes at a general level | **Unconfirmed from available source metadata.** | Yes; direct behavior, but observed range is narrow |
| `Power_Generated` | `float64` | 8.485232 | 438.556840 | 183.151609 | 276.001580 | Main generated-power measurement | Yes | Yes: `W` | Essential operational output |

The exact NumPy datetime resolution shown by pandas can vary by supported pandas
version; timestamp values and their timezone-naive status do not change.

## Negative `Solar_Radiation` review

The condition `Solar_Radiation < 0` selects 314 of 1,009 observations, or
31.119921%.

| Statistic | Result |
|---|---:|
| Minimum negative value | -0.838326 W/m^2 |
| Median negative value | -0.295704 W/m^2 |
| Maximum negative value (closest to zero) | -0.005310 W/m^2 |
| First negative timestamp | 2022-04-27 19:14 |
| Last negative timestamp | 2022-04-29 01:08 |
| Contiguous negative runs | 54 |
| Longest contiguous negative run | 21 observations |

For descriptive review only, `Solar_Radiation <= 5 W/m^2` was treated as
night-like/low-light and values above 5 W/m^2 as daylight-like. This is a
transparent exploratory threshold, not a source-provided label, preprocessing
rule, or anomaly definition. All 314 negative readings lie in the resulting two
low-light windows and occur only during clock hours 19:00-05:59 in the available
timezone-naive timestamps.

| Low-light window | First/last negative reading | Negative observations |
|---|---|---:|
| 2022-04-27 18:52 to 2022-04-28 05:50 | 2022-04-27 19:14 to 2022-04-28 05:28 | 200 |
| 2022-04-28 18:54 to dataset end at 2022-04-29 01:08 | 2022-04-28 19:14 to 2022-04-29 01:08 | 114 |

Counts by timestamp date are 86 on April 27, 201 on April 28, and 27 on April
29. The 54 shorter negative runs within the two broad windows show that readings
fluctuate across zero rather than staying continuously negative.

Electrical behavior during the 314 negative readings was:

| Measurement | Minimum | Median | Mean | Maximum |
|---|---:|---:|---:|---:|
| `Power_Generated` | 8.485232 | 14.197639 | 16.391188 | 209.426260 |
| `Array_Voltage` | 1.574978 | 2.631795 | 3.036707 | 38.737212 |
| `Array_Current` | 5.384044 | 5.397938 | 5.397700 | 5.406605 |

Observed evidence: most negative readings coincide with low power and voltage,
and the median generated power is 14.197639 W versus 276.001580 W for the full
file. Higher power values occur at transition edges; for example, the maximum
within this subset is 209.426260 W. The readings are therefore temporally
clustered around two evening-to-morning windows, but are not proof of a fault.

Possible interpretations include a small sensor zero offset, nighttime noise,
or another source-specific encoding behavior. These are hypotheses only. The
available metadata does not establish the cause, so the readings remain
unchanged and are not classified as errors or anomalies.

## `Wind_Direction > 360` review

Ten observations meet this condition. The maximum is 385.630880. Immediate
direction readings show the local context:

| Timestamp | Previous direction | Current direction | Next direction |
|---|---:|---:|---:|
| 2022-04-27 15:46 | 262.860220 | 365.686800 | 307.629800 |
| 2022-04-27 16:02 | 259.091860 | 371.098160 | 358.292680 |
| 2022-04-27 18:02 | 333.668020 | 364.005840 | 261.840860 |
| 2022-04-28 10:34 | 337.702280 | 371.158640 | 366.872600 |
| 2022-04-28 10:36 | 371.158640 | 366.872600 | 205.434820 |
| 2022-04-28 10:40 | 205.434820 | 385.630880 | 289.294740 |
| 2022-04-28 10:46 | 306.752040 | 385.596040 | 280.218680 |
| 2022-04-28 12:14 | 308.606100 | 363.535320 | 243.638320 |
| 2022-04-28 13:52 | 196.238960 | 377.425480 | 329.255800 |
| 2022-04-28 14:24 | 327.544700 | 376.914440 | 337.868120 |

Selected before/current/after measurements are shown below. The JSON report
stores all 14 fields for every neighboring row.

| Timestamp | Wind speed (previous/current/next) | Solar radiation (previous/current/next) | Generated power (previous/current/next) |
|---|---|---|---|
| 2022-04-27 15:46 | 0.933333 / 1.200000 / 2.333334 | 657.941480 / 660.139840 / 649.333880 | 432.272080 / 438.556840 / 437.424760 |
| 2022-04-27 16:02 | 0.933333 / 1.600000 / 0.733333 | 597.210320 / 599.454720 / 576.911760 | 433.291640 / 433.858760 / 432.244000 |
| 2022-04-27 18:02 | 0.600000 / 0.000000 / 0.000000 | 78.142600 / 76.429224 / 73.259120 | 279.614500 / 278.156920 / 278.355840 |
| 2022-04-28 10:34 | 0.400000 / 0.000000 / 0.000000 | 937.164080 / 940.283280 / 941.324480 | 400.058840 / 400.418960 / 400.597880 |
| 2022-04-28 10:36 | 0.000000 / 0.000000 / 0.866667 | 940.283280 / 941.324480 / 962.929360 | 400.418960 / 400.597880 / 403.424760 |
| 2022-04-28 10:40 | 0.866667 / 0.066667 / 0.000000 | 962.929360 / 957.571440 / 974.260960 | 403.424760 / 402.800800 / 403.199800 |
| 2022-04-28 10:46 | 0.000000 / 1.066667 / 0.000000 | 965.125760 / 984.449040 / 995.518720 | 402.460440 / 403.086840 / 403.880600 |
| 2022-04-28 12:14 | 0.666667 / 1.133334 / 0.733333 | 1110.091800 / 1108.415400 / 1088.722200 | 374.562400 / 373.990680 / 373.542720 |
| 2022-04-28 13:52 | 0.600000 / 0.066667 / 0.933333 | 994.269280 / 982.601280 / 980.774640 | 372.996680 / 372.896280 / 373.864080 |
| 2022-04-28 14:24 | 0.600000 / 0.066667 / 0.400000 | 888.645120 / 869.203280 / 864.893280 | 374.958200 / 374.621800 / 374.540800 |

Eight readings are isolated. The readings at 2022-04-28 10:34 and 10:36 form
the only consecutive pair, so the 10 observations form nine sequences in total.
They occur under varied wind speeds and are not accompanied by one consistent
power or radiation change in their immediate neighbors. Their meaning cannot be
conclusively determined unless the dataset source confirms the measurement
encoding. They remain unchanged and have not been wrapped, clipped, removed, or
labeled.

## Temporal coverage

- Coverage is 33.6 hours, from 2022-04-27 15:32 through 2022-04-29 01:08.
- Three calendar dates are represented: April 27 (254 observations), April 28
  (720), and April 29 (35). Only April 28 is complete at the 120-second cadence.
- The descriptive 5 W/m^2 threshold reveals two daylight-like periods and two
  night-like periods. The first daylight period and final night-like period are
  truncated by the dataset boundaries.
- There is only one complete daily cycle, with partial context on either side.
  This is not enough repeated daily behavior for robust model validation and
  cannot support seasonal or long-term conclusions.

The limited temporal coverage is a central limitation of the current
operational dataset. Any future model evaluation will need more independent
days or another suitable time series, and time-aware splits must be used.

## Variable relationships

Pearson correlations are descriptive calculations over this short,
autocorrelated period. They do not establish causality or independently justify
removing a variable.

Important relationships with `Power_Generated` include:

| Variable | Pearson correlation |
|---|---:|
| `Array_Voltage` | 0.999990 |
| `Air_Temp` | 0.818539 |
| `RTD_5` | 0.808869 |
| `RTD_4` | 0.806862 |
| `RTD_1` | 0.803235 |
| `RTD_2` | 0.800739 |
| `RTD_3` | 0.800618 |
| `Solar_Radiation` | 0.730455 |
| `Relative_Humidity` | -0.746166 |
| `Array_Current` | -0.809220 |

Important relationships with `Solar_Radiation` include `Air_Temp` (0.926546),
the five `RTD_*` channels (0.901310 to 0.915295), `Array_Voltage` (0.732484),
`Power_Generated` (0.730455), `Relative_Humidity` (-0.685155), and
`Array_Current` (-0.820339).

The most important redundancy and leakage considerations are:

- `Array_Voltage` and `Power_Generated` correlate at 0.999990.
- `Array_Voltage * Array_Current` reproduces `Power_Generated` with a maximum
  absolute difference of 0.000087646 and median absolute difference of
  0.000003834. This can support an electrical-consistency check, but including
  both a calculated product and generated power may leak a target or give a
  deterministic relationship excess weight, depending on the future objective.
- Every pair among `RTD_1` through `RTD_5` correlates between 0.998309 and
  0.999661. They may be redundant, or they may be spatially distinct sensors
  whose agreement/disagreement is useful. Their placement is unknown, so none
  is removed.
- Several environmental relationships are strong because measurements share a
  daily pattern. The complete matrix and all 29 pairs with absolute correlation
  at least 0.90 are retained in the JSON report.

## Proposed Day 8 feature-engineering plan

These are proposals only. None of these columns has been created in the dataset.
Trailing windows are preferred over centered windows so future observations do
not leak into a real-time feature. With a 120-second cadence, 5, 15, and 30
observations correspond to 10, 30, and 60 minutes.

| Proposed feature | Source columns | Reason | Limitation or safeguard | Scaling later? |
|---|---|---|---|---|
| `hour_sin`, `hour_cos` | `Timestamp` | Represent daily phase without a discontinuity at midnight | Timezone is unknown and coverage includes one full day only | Usually no |
| `minute_of_day_sin`, `minute_of_day_cos` | `Timestamp` | Preserve within-day timing at two-minute resolution | Same timezone and limited-cycle caveats | Usually no |
| `elapsed_minutes` | `Timestamp` | Support time deltas, ordering, and rate calculations | Absolute trend may not generalize to later files | Yes for scale-sensitive methods |
| `low_light_indicator` | `Solar_Radiation` | Separate low-light operating context from productive periods | Threshold must be validated; current 5 W/m^2 value is exploratory only | No |
| `power_mean_10m`, `power_std_10m` | `Power_Generated` | Local expected level and variability | Trailing-only; first rows need explicit minimum-period handling | Yes |
| `power_mean_30m`, `power_std_30m` | `Power_Generated` | Slower local baseline and variation | May smooth short events; trailing-only | Yes |
| `radiation_mean_10m`, `radiation_std_10m` | `Solar_Radiation` | Context for transient irradiance changes | Negative/near-zero values remain valid inputs | Yes |
| `voltage_mean_10m`, `voltage_std_10m` | `Array_Voltage` | Identify local electrical level and instability | Strongly redundant with generated power | Yes |
| `current_mean_10m`, `current_std_10m` | `Array_Current` | Detect changes in a channel with a narrow observed range | Small scale makes precision and scaling important | Yes |
| `power_min_30m`, `power_max_30m` | `Power_Generated` | Capture recent operating envelope | Trailing-only; boundary rows need explicit policy | Yes |
| `power_diff_1` | `Power_Generated` | Two-minute first difference for abrupt changes | First observation is undefined; legitimate dawn/dusk ramps exist | Yes |
| `radiation_diff_1` | `Solar_Radiation` | Capture rapid illumination change | Weather-related changes are not automatically faults | Yes |
| `voltage_diff_1`, `current_diff_1` | `Array_Voltage`, `Array_Current` | Capture electrical transitions | Voltage/power redundancy must be managed | Yes |
| `power_rate_per_minute` | `Power_Generated`, `Timestamp` | Make change comparable if cadence later varies | Requires positive, valid time deltas | Yes |
| `radiation_rate_per_minute` | `Solar_Radiation`, `Timestamp` | Quantify irradiance ramp rate | Sensitive near transitions and sensor noise | Yes |
| `power_pct_change_safe` | `Power_Generated` | Relative rather than absolute change | Compute only with a documented denominator floor | Often yes |
| `power_per_radiation_safe` | `Power_Generated`, `Solar_Radiation` | Review generation relative to available radiation | Undefined/unstable near zero; restrict to validated daylight threshold | Yes |
| `electrical_power_product` | `Array_Voltage`, `Array_Current` | Physical consistency calculation supported by actual schema | Nearly identical to `Power_Generated`; leakage/redundancy risk | Yes |
| `power_consistency_residual` | `Power_Generated`, `Array_Voltage`, `Array_Current` | Detect disagreement in the recorded electrical identity | Normal tolerance must be established on independent data | Yes |
| `rtd_mean`, `rtd_std`, `rtd_range` | `RTD_1` through `RTD_5` | Summarize common behavior and cross-channel disagreement | Sensor meanings and placements are unconfirmed | Yes |
| `air_rtd_delta_*` | `Air_Temp`, each `RTD_*` | Compare ambient and RTD behavior if RTDs measure relevant temperatures | Defer until RTD meaning is confirmed | Yes |
| `wind_direction_sin`, `wind_direction_cos` | `Wind_Direction` | Represent circular direction without a 0/360 discontinuity | Do not implement until values above 360 have a documented policy | Usually no |
| `humidity_diff_1`, `temperature_diff_1` | `Relative_Humidity`, `Air_Temp` | Environmental change context | Daily co-movement can dominate this short sample | Yes |

Before Day 8 implementation, define the anomaly-detection objective and whether
`Power_Generated` is a modeled target, an input, or both. This decision controls
whether voltage, current, their product, and power-derived features create
target leakage. Scaling parameters must later be fitted on the training period
only, never on the full time series.

## Unresolved interpretation questions

- What physical quantity and panel/location does each `RTD_1`-`RTD_5` channel
  measure, and in what unit?
- What are the confirmed units and instrumentation details for
  `Array_Voltage` and `Array_Current`?
- Are wind directions above 360 a documented encoding, calibration behavior,
  or invalid range?
- Are small negative radiation values expected sensor-zero behavior?
- What is the timestamp timezone and measurement site location?

Until reliable source or engineering evidence answers these questions, the
values and names should be preserved exactly as supplied.

## Reproduction

From the project root with the virtual environment active:

```powershell
python src/data_processing/review_operational_quality.py
python -m pytest -q
```
