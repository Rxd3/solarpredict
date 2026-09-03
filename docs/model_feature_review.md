# Day 11 feature review and model-input planning

## Scope

This read-only review uses `data/processed/operational_features_change.csv` (1,009 rows × 81 columns). No column or row is removed, no value is imputed, no scaler or model is fitted, and no transition is assigned an anomaly or fault label.

Timezone: **Not specified by dataset source; timestamps kept timezone-naive.**

## Recommendation for the first prototype

Implement **Approach A: compact unsupervised multivariate detection** first, using the Core Unsupervised Features listed below. It is the simpler defensible prototype for a normal-only dataset this short because it does not first require a reliable expected-power regression model. This is a plan only; no model exists yet.

Approach B, prediction-residual detection, remains a useful second experiment. There, `Power_Generated` becomes the target. Predictors must exclude voltage, current, their product, consistency residuals, and every power-derived rolling/change feature that would reconstruct or leak the target.

## Complete feature inventory

Variability uses an exploratory rule: constant means at most one non-missing value; near-constant means either one value occupies at least 95% of observations or the population standard deviation divided by absolute mean is at most 1%. This rule flags only `Array_Current` as near-constant in the current file.

### Identifiers/Time

| Feature | Missing | Missing % | Variability | Intended interpretation | Recommendation |
|---|---:|---:|---|---|---|
| `Timestamp` | 0 | 0.000% | variable | Timezone-naive observation time and chronological index. | **CONTEXT_ONLY** |
| `hour` | 0 | 0.000% | variable | Clock hour extracted from the timezone-naive timestamp. | **EXCLUDE_FROM_MODEL** |
| `minute` | 0 | 0.000% | variable | Clock minute extracted from the timestamp. | **EXCLUDE_FROM_MODEL** |
| `minute_of_day` | 0 | 0.000% | variable | Linear minute index within a day. | **EXCLUDE_FROM_MODEL** |
| `elapsed_minutes_from_start` | 0 | 0.000% | variable | Dataset-position trend measured from the first observation. | **EXCLUDE_FROM_MODEL** |
| `hour_sin` | 0 | 0.000% | variable | Cyclical sine encoding of clock hour. | **EXCLUDE_FROM_MODEL** |
| `hour_cos` | 0 | 0.000% | variable | Cyclical cosine encoding of clock hour. | **EXCLUDE_FROM_MODEL** |
| `minute_of_day_sin` | 0 | 0.000% | variable | Fine-grained cyclical sine encoding of time of day. | **KEEP** |
| `minute_of_day_cos` | 0 | 0.000% | variable | Fine-grained cyclical cosine encoding of time of day. | **KEEP** |

### Original Environmental Measurements

| Feature | Missing | Missing % | Variability | Intended interpretation | Recommendation |
|---|---:|---:|---|---|---|
| `Air_Temp` | 0 | 0.000% | variable | Original ambient air-temperature measurement. | **KEEP** |
| `Relative_Humidity` | 0 | 0.000% | variable | Original ambient relative-humidity measurement. | **KEEP** |
| `Wind_Speed` | 0 | 0.000% | variable | Original wind-speed measurement used as environmental context. | **KEEP** |
| `Wind_Direction` | 0 | 0.000% | variable | Original circular wind-direction measurement; values above 360 remain unresolved. | **UNRESOLVED** |
| `Solar_Radiation` | 0 | 0.000% | variable | Original solar-radiation measurement and primary available-light context. | **KEEP** |

### Original Electrical Measurements

| Feature | Missing | Missing % | Variability | Intended interpretation | Recommendation |
|---|---:|---:|---|---|---|
| `Array_Voltage` | 0 | 0.000% | variable | Original array-voltage measurement; nearly duplicates generated-power behavior here. | **EXCLUDE_FROM_MODEL** |
| `Array_Current` | 0 | 0.000% | near_constant | Original array-current measurement with a narrow observed range. | **OPTIONAL** |
| `Power_Generated` | 0 | 0.000% | variable | Original generated-power measurement and main operational output. | **KEEP** |

### Rtd Channels

| Feature | Missing | Missing % | Variability | Intended interpretation | Recommendation |
|---|---:|---:|---|---|---|
| `RTD_1` | 0 | 0.000% | variable | Original RTD channel 1; placement, physical meaning, and unit are unconfirmed. | **EXCLUDE_FROM_MODEL** |
| `RTD_2` | 0 | 0.000% | variable | Original RTD channel 2; placement, physical meaning, and unit are unconfirmed. | **EXCLUDE_FROM_MODEL** |
| `RTD_3` | 0 | 0.000% | variable | Original RTD channel 3; placement, physical meaning, and unit are unconfirmed. | **EXCLUDE_FROM_MODEL** |
| `RTD_4` | 0 | 0.000% | variable | Original RTD channel 4; placement, physical meaning, and unit are unconfirmed. | **EXCLUDE_FROM_MODEL** |
| `RTD_5` | 0 | 0.000% | variable | Original RTD channel 5; placement, physical meaning, and unit are unconfirmed. | **EXCLUDE_FROM_MODEL** |

### Basic Engineered Features

| Feature | Missing | Missing % | Variability | Intended interpretation | Recommendation |
|---|---:|---:|---|---|---|
| `rtd_mean` | 0 | 0.000% | variable | Row-wise mean of the five unverified RTD channels. | **KEEP** |
| `rtd_std` | 0 | 0.000% | variable | Row-wise population standard deviation across the five RTD channels. | **KEEP** |
| `rtd_min` | 0 | 0.000% | variable | Row-wise minimum of the five RTD channels. | **EXCLUDE_FROM_MODEL** |
| `rtd_max` | 0 | 0.000% | variable | Row-wise maximum of the five RTD channels. | **EXCLUDE_FROM_MODEL** |
| `rtd_range` | 0 | 0.000% | variable | Row-wise RTD maximum minus minimum, describing channel disagreement. | **OPTIONAL** |

### Rolling Features

| Feature | Missing | Missing % | Variability | Intended interpretation | Recommendation |
|---|---:|---:|---|---|---|
| `power_generated_roll_mean_10m` | 4 | 0.396% | variable | Complete trailing 10-minute mean of generated power; current and earlier observations only. | **OPTIONAL** |
| `power_generated_roll_std_10m` | 4 | 0.396% | variable | Complete trailing 10-minute std of generated power; current and earlier observations only. | **OPTIONAL** |
| `power_generated_roll_min_10m` | 4 | 0.396% | variable | Complete trailing 10-minute min of generated power; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `power_generated_roll_max_10m` | 4 | 0.396% | variable | Complete trailing 10-minute max of generated power; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `power_generated_roll_mean_30m` | 14 | 1.388% | variable | Complete trailing 30-minute mean of generated power; current and earlier observations only. | **OPTIONAL** |
| `power_generated_roll_std_30m` | 14 | 1.388% | variable | Complete trailing 30-minute std of generated power; current and earlier observations only. | **OPTIONAL** |
| `power_generated_roll_min_30m` | 14 | 1.388% | variable | Complete trailing 30-minute min of generated power; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `power_generated_roll_max_30m` | 14 | 1.388% | variable | Complete trailing 30-minute max of generated power; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `power_generated_roll_mean_60m` | 29 | 2.874% | variable | Complete trailing 60-minute mean of generated power; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `power_generated_roll_std_60m` | 29 | 2.874% | variable | Complete trailing 60-minute std of generated power; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `power_generated_roll_min_60m` | 29 | 2.874% | variable | Complete trailing 60-minute min of generated power; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `power_generated_roll_max_60m` | 29 | 2.874% | variable | Complete trailing 60-minute max of generated power; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `solar_radiation_roll_mean_10m` | 4 | 0.396% | variable | Complete trailing 10-minute mean of solar radiation; current and earlier observations only. | **OPTIONAL** |
| `solar_radiation_roll_std_10m` | 4 | 0.396% | variable | Complete trailing 10-minute std of solar radiation; current and earlier observations only. | **OPTIONAL** |
| `solar_radiation_roll_min_10m` | 4 | 0.396% | variable | Complete trailing 10-minute min of solar radiation; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `solar_radiation_roll_max_10m` | 4 | 0.396% | variable | Complete trailing 10-minute max of solar radiation; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `solar_radiation_roll_mean_30m` | 14 | 1.388% | variable | Complete trailing 30-minute mean of solar radiation; current and earlier observations only. | **OPTIONAL** |
| `solar_radiation_roll_std_30m` | 14 | 1.388% | variable | Complete trailing 30-minute std of solar radiation; current and earlier observations only. | **OPTIONAL** |
| `solar_radiation_roll_min_30m` | 14 | 1.388% | variable | Complete trailing 30-minute min of solar radiation; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `solar_radiation_roll_max_30m` | 14 | 1.388% | variable | Complete trailing 30-minute max of solar radiation; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `solar_radiation_roll_mean_60m` | 29 | 2.874% | variable | Complete trailing 60-minute mean of solar radiation; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `solar_radiation_roll_std_60m` | 29 | 2.874% | variable | Complete trailing 60-minute std of solar radiation; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `solar_radiation_roll_min_60m` | 29 | 2.874% | variable | Complete trailing 60-minute min of solar radiation; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `solar_radiation_roll_max_60m` | 29 | 2.874% | variable | Complete trailing 60-minute max of solar radiation; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `air_temp_roll_mean_30m` | 14 | 1.388% | variable | Complete trailing 30-minute mean of air temperature; current and earlier observations only. | **OPTIONAL** |
| `air_temp_roll_mean_60m` | 29 | 2.874% | variable | Complete trailing 60-minute mean of air temperature; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `relative_humidity_roll_mean_30m` | 14 | 1.388% | variable | Complete trailing 30-minute mean of relative humidity; current and earlier observations only. | **OPTIONAL** |
| `relative_humidity_roll_mean_60m` | 29 | 2.874% | variable | Complete trailing 60-minute mean of relative humidity; current and earlier observations only. | **EXCLUDE_FROM_MODEL** |
| `rtd_mean_roll_mean_30m` | 14 | 1.388% | variable | Complete trailing 30-minute mean of mean RTD level; current and earlier observations only. | **OPTIONAL** |
| `rtd_mean_roll_std_30m` | 14 | 1.388% | variable | Complete trailing 30-minute std of mean RTD level; current and earlier observations only. | **OPTIONAL** |

### Change/Rate Features

| Feature | Missing | Missing % | Variability | Intended interpretation | Recommendation |
|---|---:|---:|---|---|---|
| `power_generated_diff` | 1 | 0.099% | variable | Current minus previous power generated value over one two-minute interval. | **OPTIONAL** |
| `solar_radiation_diff` | 1 | 0.099% | variable | Current minus previous solar radiation value over one two-minute interval. | **OPTIONAL** |
| `air_temp_diff` | 1 | 0.099% | variable | Current minus previous air temp value over one two-minute interval. | **OPTIONAL** |
| `relative_humidity_diff` | 1 | 0.099% | variable | Current minus previous relative humidity value over one two-minute interval. | **OPTIONAL** |
| `array_voltage_diff` | 1 | 0.099% | variable | Current minus previous array voltage value over one two-minute interval. | **EXCLUDE_FROM_MODEL** |
| `array_current_diff` | 1 | 0.099% | variable | Current minus previous array current value over one two-minute interval. | **OPTIONAL** |
| `rtd_mean_diff` | 1 | 0.099% | variable | Current minus previous rtd mean value over one two-minute interval. | **OPTIONAL** |
| `power_generated_rate_per_min` | 1 | 0.099% | variable | First difference of power generated divided by the verified two-minute interval. | **EXCLUDE_FROM_MODEL** |
| `solar_radiation_rate_per_min` | 1 | 0.099% | variable | First difference of solar radiation divided by the verified two-minute interval. | **EXCLUDE_FROM_MODEL** |
| `air_temp_rate_per_min` | 1 | 0.099% | variable | First difference of air temp divided by the verified two-minute interval. | **EXCLUDE_FROM_MODEL** |
| `relative_humidity_rate_per_min` | 1 | 0.099% | variable | First difference of relative humidity divided by the verified two-minute interval. | **EXCLUDE_FROM_MODEL** |
| `array_voltage_rate_per_min` | 1 | 0.099% | variable | First difference of array voltage divided by the verified two-minute interval. | **EXCLUDE_FROM_MODEL** |
| `array_current_rate_per_min` | 1 | 0.099% | variable | First difference of array current divided by the verified two-minute interval. | **EXCLUDE_FROM_MODEL** |
| `rtd_mean_rate_per_min` | 1 | 0.099% | variable | First difference of rtd mean divided by the verified two-minute interval. | **EXCLUDE_FROM_MODEL** |
| `power_generated_relative_change` | 1 | 0.099% | variable | Safeguarded two-minute generated-power change divided by the previous power value. | **OPTIONAL** |
| `solar_radiation_relative_change` | 518 | 51.338% | variable | Safeguarded two-minute radiation change divided by the previous radiation value. | **OPTIONAL** |
| `power_deviation_from_30m_mean` | 14 | 1.388% | variable | Generated power minus its causal trailing 30-minute mean. | **OPTIONAL** |
| `solar_deviation_from_30m_mean` | 14 | 1.388% | variable | Solar radiation minus its causal trailing 30-minute mean. | **OPTIONAL** |
| `power_deviation_ratio_30m` | 14 | 1.388% | variable | Power deviation divided by its trailing mean when the denominator is safe. | **OPTIONAL** |
| `solar_deviation_ratio_30m` | 520 | 51.536% | variable | Radiation deviation divided by its trailing mean when the denominator is safe. | **OPTIONAL** |

### Diagnostic Features

| Feature | Missing | Missing % | Variability | Intended interpretation | Recommendation |
|---|---:|---:|---|---|---|
| `electrical_power_product` | 0 | 0.000% | variable | Array_Voltage multiplied by Array_Current; almost exactly recreates Power_Generated. | **EXCLUDE_FROM_MODEL** |
| `power_consistency_residual` | 0 | 0.000% | variable | Power_Generated minus the voltage-current product. | **EXCLUDE_FROM_MODEL** |
| `power_consistency_abs_error` | 0 | 0.000% | variable | Absolute value of the power-consistency residual. | **EXCLUDE_FROM_MODEL** |

### Context Flags

| Feature | Missing | Missing % | Variability | Intended interpretation | Recommendation |
|---|---:|---:|---|---|---|
| `low_light_context` | 0 | 0.000% | variable | Exploratory context flag for Solar_Radiation at or below 5 W/m^2. | **CONTEXT_ONLY** |

## Missingness review

| Missingness source | NaN cells | Policy |
|---|---:|---|
| Rolling-window warm-up | 546 | Apply a candidate-specific history rule: none for Core, 14 initial rows for Extended, and 29 only if a 60-minute feature is later selected; never backfill |
| First-difference warm-up | 14 | The Extended set's 14-row exclusion also covers this first row; do not fill from future data |
| Deliberate denominator safety | 1025 | Exclude high-missingness ratios from the general model; reserve them for daylight analysis |
| Unexpected | 0 | None observed; investigate rather than blanket-impute if this changes |
| **Total** | **1585** | No policy is applied during Day 11 |

The categories are cell counts and do not overlap. All current NaNs match the documented rolling, previous-row, or denominator-safety formulas.

## Redundancy analysis

Pairwise Pearson correlation identified **181** numerical pairs with absolute correlation ≥ 0.95. The complete pair list and paired-observation counts are in `outputs/model_feature_review.json`.

Highly correlated variables can make Euclidean distances or similar unsupervised objectives count one physical pattern several times. They may therefore dominate scores without adding independent evidence. Correlation is exploratory evidence, not an automatic deletion rule.

| Focused group | Members | Absolute-correlation range | Initial recommendation |
|---|---|---:|---|
| electrical power reconstruction | `Power_Generated`, `Array_Voltage`, `electrical_power_product` | 0.999990–1.000000 | Choose one representation for an initial model; the product nearly recreates generated power. |
| power and voltage transitions | `power_generated_diff`, `power_generated_rate_per_min`, `array_voltage_diff`, `array_voltage_rate_per_min` | 0.999975–1.000000 | Rates are fixed scalar multiples of differences, and voltage transitions mirror power transitions. |
| RTD level summaries | `RTD_1`, `RTD_2`, `RTD_3`, `RTD_4`, `RTD_5`, `rtd_mean`, `rtd_min`, `rtd_max` | 0.998165–0.999956 | Use the mean for level and a dispersion summary rather than all level channels initially. |
| RTD dispersion summaries | `rtd_std`, `rtd_range` | 0.986815–0.986815 | Both summarize disagreement among the same five RTD channels. |
| power rolling means | `power_generated_roll_mean_10m`, `power_generated_roll_mean_30m`, `power_generated_roll_mean_60m` | 0.968484–0.991200 | Overlapping trailing windows encode similar slow-moving power level information. |
| solar rolling means | `solar_radiation_roll_mean_10m`, `solar_radiation_roll_mean_30m`, `solar_radiation_roll_mean_60m` | 0.979623–0.995542 | Overlapping trailing windows encode similar radiation level information. |
| environmental rolling means | `air_temp_roll_mean_30m`, `air_temp_roll_mean_60m` | 0.993032–0.993032 | The two temperature windows are highly correlated in this short file. |
| humidity rolling means | `relative_humidity_roll_mean_30m`, `relative_humidity_roll_mean_60m` | 0.994833–0.994833 | The two humidity windows are highly correlated in this short file. |
| linear clock representations | `hour`, `minute_of_day` | 0.999210–0.999210 | Minute of day almost deterministically contains clock hour; cyclical time encoding is preferred. |

Every per-minute rate is exactly its two-minute difference divided by two and therefore has correlation 1.0 with that difference. Keep one representation, not both.

### Electrical strategy

For the initial unsupervised model, keep `Power_Generated` as the electrical level, exclude `Array_Voltage` and `electrical_power_product`, and make `Array_Current` an extended-only comparison. Keep `power_consistency_residual` and `power_consistency_abs_error` in the master dataset as diagnostics, not general model inputs. This prevents the voltage-current identity from receiving repeated weight.

For a future prediction-residual model, none of those electrical variables or any power-derived feature should predict `Power_Generated`, because that would leak or nearly reconstruct the target.

### RTD strategy

Use **Option B** initially: `rtd_mean` for shared level and `rtd_std` for channel disagreement; compare `rtd_range` only in the extended set. Retain all five raw RTD channels in the master dataset until their placement, units, and independent failure modes are known.

## Candidate model-input sets

### Core Unsupervised Features (9 features)

Compact coverage of output, illumination, environment, summarized RTD behavior, and cyclical time without deterministic electrical duplicates.

```text
Power_Generated
Solar_Radiation
Air_Temp
Relative_Humidity
Wind_Speed
rtd_mean
rtd_std
minute_of_day_sin
minute_of_day_cos
```

Limitations:

- The source covers only about 33.6 hours.
- Power and environmental variables still share a strong daily cycle.
- RTD physical meaning and units remain unconfirmed.
- Scaling must later be fitted on training data only.

### Extended Diagnostic Features (23 features)

Adds one current channel, RTD disagreement, selected 30-minute baselines, selected changes, and absolute deviations for later comparison.

```text
Power_Generated
Solar_Radiation
Air_Temp
Relative_Humidity
Wind_Speed
rtd_mean
rtd_std
minute_of_day_sin
minute_of_day_cos
Array_Current
rtd_range
power_generated_roll_mean_30m
power_generated_roll_std_30m
solar_radiation_roll_mean_30m
solar_radiation_roll_std_30m
rtd_mean_roll_mean_30m
rtd_mean_roll_std_30m
power_generated_diff
solar_radiation_diff
array_current_diff
rtd_mean_diff
power_deviation_from_30m_mean
solar_deviation_from_30m_mean
```

Limitations:

- More correlated inputs may influence distance-based methods unevenly.
- Rolling and deviation inputs require removal of the initial warm-up rows.
- Array_Current is near-constant under the documented exploratory rule.
- This set must be compared using time-aware evaluation, not assumed superior.

### Daylight-Specific Features (8 features)

Retains safeguarded solar and power ratios for a separately filtered daylight analysis where their denominators are interpretable.

```text
Power_Generated
Solar_Radiation
power_generated_diff
solar_radiation_diff
power_generated_relative_change
solar_radiation_relative_change
power_deviation_ratio_30m
solar_deviation_ratio_30m
```

Limitations:

- It is not a general day-and-night feature set.
- The 5 W/m^2 threshold is exploratory, not a manufacturer limit.
- Relative changes remain sensitive near the accepted denominator boundary.

Application policy: Use only after defining a causal daylight mask that satisfies the existing 5 W/m^2 radiation denominator safeguard and rolling-baseline availability.

## Recurring 17:00 transition review

Each event uses 21 observations from 20 minutes before through 20 minutes after the event, inclusive. The complete windows are stored in `outputs/recurring_transition_review.json`.

| Event | Power change | Radiation change | Voltage change | Current change | Air-temperature change | Humidity change | Low-light before → after |
|---|---:|---:|---:|---:|---:|---:|---|
| 2022-04-27 17:00:00 | -125.304960 | 14.622780 | -23.461776 | 0.007918 | -6.452032 | -0.509792 | 0 → 0 |
| 2022-04-28 17:00:00 | -103.485460 | 39.060360 | -19.401192 | 0.008334 | -1.548892 | 0.074580 | 0 → 0 |

The aligned ±20-minute profiles are strongly similar for generated power (r=0.998747) and voltage (r=0.998765). Both events show a step down in power and voltage while current rises slightly; power remains near a lower plateau afterward and the trailing means decay gradually.

Radiation rises at the event step on both dates (+14.622780 and +39.060360), and `low_light_context` stays 0 throughout both windows. The radiation profiles are directionally similar but less closely aligned (r=0.886562) and differ in level. Environmental changes are not identical. The evidence therefore supports only the neutral description **recurring power transition**; it does not establish a fault, anomaly, or cause.

## Decisions deferred until modelling

- Confirm the compact unsupervised approach and candidate set before implementing a model.
- Clarify RTD channel placement and units if source documentation is available.
- Resolve the encoding of the ten `Wind_Direction` values above 360 before circular encoding.
- Obtain more independent days and choose a time-aware train/validation/test policy.
- Confirm whether the exploratory 5 W/m² daylight threshold is acceptable for specialized analysis.

## Validation

All Day 11 checks passed: all 81 columns are inventoried; candidate names exist; all 1,585 NaN cells have documented causes; dataset hashes are unchanged; the models directory is unchanged; no anomaly labels, fitted preprocessing, model files, or modified feature datasets were created.

## Reproduction

```powershell
python src/data_processing/review_model_features.py
python -m pytest -q
```
