# Day 17 baseline score and distribution-shift investigation

## Scope

This investigation loads the saved Day 16 `StandardScaler` and Isolation
Forest. It calls only `transform` and `score_samples`; it does not fit or replace
either artifact. Reproduced scores differ from the saved CSV values by at most
`5.55e-17`, which is CSV floating-point precision.

No threshold, anomaly label, synthetic evaluation data, or classification
metric is created. All observations discussed below are untouched baseline
measurements. A high score means more isolated according to this particular
model, not a confirmed anomaly or photovoltaic fault.

## Core feature distributions

Values after `Z:` use the existing training-fitted scaler. Standard deviations
are population values. The JSON report contains the full-precision statistics.

| Feature / partition | Raw mean ± std; median; range | Training-scaled mean ± std; range |
|---|---|---|
| Power, training | 122.190 ± 157.708; 15.226; 12.483 to 438.557 | 0.000 ± 1.000; -0.696 to 2.006 |
| Power, calibration | 259.799 ± 142.855; 299.993; 13.969 to 403.881 | 0.873 ± 0.906; -0.686 to 1.786 |
| Power, evaluation | 167.512 ± 166.636; 15.298; 8.485 to 403.954 | 0.287 ± 1.057; -0.721 to 1.787 |
| Radiation, training | 83.656 ± 178.368; 0.067; -0.838 to 710.463 | 0.000 ± 1.000; -0.474 to 3.514 |
| Radiation, calibration | 412.742 ± 461.733; 163.446; -0.706 to 1132.128 | 1.845 ± 2.589; -0.473 to 5.878 |
| Radiation, evaluation | 175.899 ± 284.068; 0.282; -0.686 to 979.139 | 0.517 ± 1.593; -0.473 to 5.020 |
| Air temperature, training | 21.961 ± 8.409; 18.059; 14.837 to 44.327 | 0.000 ± 1.000; -0.847 to 2.660 |
| Air temperature, calibration | 27.930 ± 14.448; 22.401; 12.075 to 49.632 | 0.710 ± 1.718; -1.176 to 3.291 |
| Air temperature, evaluation | 26.169 ± 9.679; 20.537; 16.158 to 43.655 | 0.500 ± 1.151; -0.690 to 2.580 |
| Relative humidity, training | 31.915 ± 12.191; 35.025; 10.648 to 46.961 | 0.000 ± 1.000; -1.744 to 1.234 |
| Relative humidity, calibration | 34.617 ± 12.810; 35.126; 14.542 to 51.489 | 0.222 ± 1.051; -1.425 to 1.606 |
| Relative humidity, evaluation | 28.896 ± 11.235; 34.750; 11.708 to 44.301 | -0.248 ± 0.922; -1.657 to 1.016 |
| Wind speed, training | 0.151 ± 0.429; 0.000; 0.000 to 2.867 | 0.000 ± 1.000; -0.352 to 6.334 |
| Wind speed, calibration | 0.233 ± 0.469; 0.000; 0.000 to 2.933 | 0.192 ± 1.093; -0.352 to 6.489 |
| Wind speed, evaluation | 0.155 ± 0.398; 0.000; 0.000 to 2.333 | 0.010 ± 0.928; -0.352 to 5.090 |
| RTD mean, training | 36.315 ± 23.970; 25.865; 22.198 to 101.815 | 0.000 ± 1.000; -0.589 to 2.733 |
| RTD mean, calibration | 54.003 ± 34.092; 39.499; 20.613 to 112.205 | 0.738 ± 1.422; -0.655 to 3.166 |
| RTD mean, evaluation | 49.779 ± 34.183; 28.584; 24.533 to 111.691 | 0.562 ± 1.426; -0.492 to 3.145 |
| RTD std, training | 0.698 ± 0.583; 0.477; 0.353 to 3.776 | 0.000 ± 1.000; -0.591 to 5.282 |
| RTD std, calibration | 1.632 ± 1.537; 0.893; 0.178 to 8.331 | 1.603 ± 2.638; -0.892 to 13.100 |
| RTD std, evaluation | 0.883 ± 0.731; 0.464; 0.236 to 2.693 | 0.318 ± 1.254; -0.793 to 3.424 |
| Time sine, training | -0.465 ± 0.531; -0.685; -1.000 to 0.649 | 0.000 ± 1.000; -1.008 to 2.099 |
| Time sine, calibration | 0.557 ± 0.449; 0.743; -0.477 to 1.000 | 1.926 ± 0.846; -0.023 to 2.759 |
| Time sine, evaluation | -0.623 ± 0.368; -0.743; -1.000 to 0.292 | -0.298 ± 0.694; -1.008 to 1.426 |
| Time cosine, training | 0.494 ± 0.508; 0.728; -0.602 to 1.000 | 0.000 ± 1.000; -2.157 to 0.996 |
| Time cosine, calibration | -0.387 ± 0.581; -0.570; -1.000 to 0.755 | -1.734 ± 1.145; -2.941 to 0.513 |
| Time cosine, evaluation | 0.264 ± 0.638; 0.391; -0.875 to 1.000 | -0.452 ± 1.255; -2.694 to 0.996 |

### Counts outside training support

Each cell is `below training minimum / above training maximum / |z|>2 /
|z|>3 / |z|>5`. These are support diagnostics, not anomaly counts.

| Feature | Calibration | Evaluation baseline |
|---|---:|---:|
| `Power_Generated` | 0 / 0 / 0 / 0 / 0 | 67 / 0 / 0 / 0 / 0 |
| `Solar_Radiation` | 0 / 118 / 119 / 118 / 95 | 0 / 31 / 64 / 38 / 1 |
| `Air_Temp` | 121 / 104 / 115 / 59 / 0 | 0 / 0 / 57 / 0 / 0 |
| `Relative_Humidity` | 0 / 103 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |
| `Wind_Speed` | 0 / 1 / 27 / 11 / 3 | 0 / 0 / 15 / 10 / 1 |
| `rtd_mean` | 101 / 59 / 91 / 22 / 0 | 0 / 66 / 92 / 38 / 0 |
| `rtd_std` | 15 / 15 / 144 / 92 / 16 | 2 / 0 / 54 / 20 / 0 |
| `minute_of_day_sin` | 0 / 198 / 205 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |
| `minute_of_day_cos` | 164 / 0 / 175 / 0 / 0 | 48 / 0 / 60 / 0 / 0 |

Calibration therefore has broad training-support gaps in radiation,
temperature, RTD summaries, and cyclical clock position. Evaluation departures
are smaller but remain visible in radiation, RTD mean/spread, temperature, and
time cosine.

## Focused `rtd_std` review

| Partition | Raw range | Median | P90 | P95 | P97.5 | P99 | Scaled range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Training | 0.353–3.776 | 0.477 | 1.676 | 2.055 | 2.397 | 3.013 | -0.591–5.282 |
| Calibration | 0.178–8.331 | 0.893 | 2.880 | 3.540 | 6.297 | 8.190 | -0.892–13.100 |
| Evaluation | 0.236–2.693 | 0.464 | 2.223 | 2.457 | 2.558 | 2.625 | -0.793–3.424 |

The ten largest calibration values occur from 10:28 through 10:56. They are all
daylight-like and all have radiation above 100 W/m²; their radiation ranges
from 927.53 to 1019.78 W/m² and power from 399.50 to 403.88 W. The largest value
is 8.3309 at 10:54, equivalent to 13.100 training-scaled units. Calibration
`rtd_std` has a descriptive Pearson correlation of 0.623 with radiation, and
its mean is 2.079 in daylight-like rows versus 0.480 in low light.

This supports an association with the observed high-radiation/daylight period,
but not a physical fault explanation. RTD placement and units remain
unconfirmed. The JSON report includes each top timestamp, concurrent sensor
values, and nearby ±10-minute means.

## Clock-time coverage

| Clock hour | Training | Calibration | Evaluation |
|---:|---:|---:|---:|
| 00 | 30 | 0 | 30 |
| 01 | 30 | 0 | 5 |
| 02 | 22 | 8 | 0 |
| 03 | 0 | 30 | 0 |
| 04 | 0 | 30 | 0 |
| 05 | 0 | 30 | 0 |
| 06 | 0 | 30 | 0 |
| 07 | 0 | 30 | 0 |
| 08 | 0 | 30 | 0 |
| 09 | 0 | 30 | 0 |
| 10 | 0 | 30 | 0 |
| 11 | 0 | 30 | 0 |
| 12 | 0 | 30 | 0 |
| 13 | 0 | 28 | 2 |
| 14 | 0 | 0 | 30 |
| 15 | 14 | 0 | 30 |
| 16 | 30 | 0 | 30 |
| 17 | 30 | 0 | 30 |
| 18 | 30 | 0 | 30 |
| 19 | 30 | 0 | 30 |
| 20 | 30 | 0 | 30 |
| 21 | 30 | 0 | 30 |
| 22 | 30 | 0 | 30 |
| 23 | 30 | 0 | 30 |

Training represents hours 00–02 and 15–23. Calibration represents 02–13; hours
03–13 are completely absent from training. Evaluation represents 13–01; hours
13–14 are absent from training and hour 15 is only partially represented there.
Consequently, the calibration sine/cosine positions occupy clock regions the
model never saw during fitting. The time features can contribute to elevated
scores, but they are not removed during this investigation.

## Daylight and low-light score behavior

Low light remains the exploratory rule `Solar_Radiation <= 5 W/m²`; it is not a
physical fault threshold.

| Partition/context | Rows | Mean | Median | Std | Range | P95 |
|---|---:|---:|---:|---:|---:|---:|
| Training daylight-like | 100 | 0.538617 | 0.541451 | 0.049245 | 0.465951–0.666732 | 0.638598 |
| Training low light | 236 | 0.440476 | 0.434595 | 0.021820 | 0.412857–0.592586 | 0.475080 |
| Calibration daylight-like | 242 | 0.657304 | 0.668917 | 0.028342 | 0.592871–0.705487 | 0.698676 |
| Calibration low light | 94 | 0.542976 | 0.542871 | 0.024687 | 0.501809–0.593891 | 0.588203 |
| Evaluation daylight-like | 149 | 0.587517 | 0.594759 | 0.063444 | 0.487985–0.672845 | 0.669440 |
| Evaluation low light | 188 | 0.470621 | 0.466696 | 0.020730 | 0.434172–0.550405 | 0.501023 |

Later scores are elevated in both contexts relative to the matching training
context. Daylight-like observations have the larger absolute scores and account
for every top-ten score in all three partitions. Calibration's overall shift is
mainly shaped by its 242 daylight-like rows, but its low-light mean is also
0.1025 above the training low-light mean.

## Score relationship with feature extremeness

The anomaly score correlates positively with the largest absolute
training-scaled feature value: Pearson/Spearman correlations are 0.837/0.750 in
training, 0.671/0.859 in calibration, and 0.944/0.922 in evaluation. Counts of
features beyond |z|=2 also correlate with score at 0.747, 0.765, and 0.893
respectively. This shows model-score association, not causality or physical
importance.

## High-score baseline observations

All entries below are **high-score baseline observations**, not confirmed
anomalies or faults. `Largest |z|` names the most extreme Core feature under the
training-fitted scaler. The full JSON records all requested values plus counts
beyond |z|=2, 3, and 5.

### Training top ten

| Timestamp | Score | Power | Radiation | Temp | RH | Wind | RTD mean/std | Context | Largest |z| |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Apr 27 15:36 | 0.666732 | 430.25 | 695.51 | 43.29 | 11.05 | 2.87 | 93.41 / 2.865 | daylight | Wind 6.33 |
| Apr 27 15:32 | 0.657041 | 430.15 | 707.53 | 43.35 | 12.79 | 0.53 | 93.81 / 2.958 | daylight | RTD std 3.88 |
| Apr 27 15:40 | 0.646648 | 431.90 | 674.06 | 43.73 | 10.72 | 2.27 | 96.87 / 3.093 | daylight | Wind 4.93 |
| Apr 27 17:50 | 0.645365 | 349.34 | 84.05 | 28.77 | 14.14 | 0.00 | 38.79 / 3.776 | daylight | RTD std 5.28 |
| Apr 27 15:34 | 0.640236 | 430.71 | 710.46 | 43.43 | 11.47 | 1.87 | 93.93 / 2.866 | daylight | Wind 4.00 |
| Apr 27 15:38 | 0.638512 | 433.07 | 696.24 | 43.52 | 12.06 | 0.00 | 96.77 / 3.028 | daylight | RTD std 4.00 |
| Apr 27 15:48 | 0.632832 | 437.42 | 649.33 | 44.00 | 11.75 | 2.33 | 101.28 / 2.395 | daylight | Wind 5.09 |
| Apr 27 15:44 | 0.621137 | 432.27 | 657.94 | 43.48 | 10.86 | 0.93 | 96.76 / 3.108 | daylight | RTD std 4.14 |
| Apr 27 15:46 | 0.609378 | 438.56 | 660.14 | 44.08 | 11.73 | 1.20 | 101.82 / 2.284 | daylight | Radiation 3.23 |
| Apr 27 15:54 | 0.608359 | 434.66 | 610.61 | 44.07 | 12.80 | 0.00 | 100.60 / 2.230 | daylight | Radiation 2.95 |

### Calibration top ten

| Timestamp | Score | Power | Radiation | Temp | RH | Wind | RTD mean/std | Context | Largest |z| |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Apr 28 11:58 | 0.705487 | 382.72 | 1124.45 | 47.36 | 18.85 | 2.60 | 102.00 / 2.725 | daylight | Radiation 5.84 |
| Apr 28 10:54 | 0.701707 | 402.03 | 1008.40 | 48.73 | 24.89 | 0.07 | 84.49 / 8.331 | daylight | RTD std 13.10 |
| Apr 28 10:42 | 0.701171 | 403.20 | 974.26 | 49.45 | 23.29 | 0.00 | 77.14 / 6.366 | daylight | RTD std 9.73 |
| Apr 28 13:42 | 0.701023 | 374.17 | 1050.22 | 41.98 | 14.99 | 2.93 | 111.65 / 2.622 | daylight | Wind 6.49 |
| Apr 28 10:36 | 0.701013 | 400.60 | 941.32 | 47.84 | 24.47 | 0.00 | 76.73 / 6.053 | daylight | RTD std 9.19 |
| Apr 28 10:44 | 0.701013 | 402.46 | 965.13 | 48.95 | 24.16 | 0.00 | 79.99 / 7.676 | daylight | RTD std 11.98 |
| Apr 28 10:52 | 0.700894 | 402.83 | 1016.89 | 48.78 | 22.45 | 0.00 | 83.90 / 8.193 | daylight | RTD std 12.86 |
| Apr 28 10:56 | 0.700894 | 402.46 | 1019.78 | 47.68 | 22.86 | 0.00 | 84.60 / 8.301 | daylight | RTD std 13.05 |
| Apr 28 10:48 | 0.700696 | 403.88 | 995.52 | 48.61 | 23.11 | 0.00 | 79.89 / 8.185 | daylight | RTD std 12.85 |
| Apr 28 10:40 | 0.700600 | 402.80 | 957.57 | 49.36 | 23.87 | 0.07 | 76.85 / 6.183 | daylight | RTD std 9.41 |

### Evaluation-baseline top ten

| Timestamp | Score | Power | Radiation | Temp | RH | Wind | RTD mean/std | Context | Largest |z| |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Apr 28 14:16 | 0.672845 | 374.32 | 908.29 | 43.25 | 13.96 | 0.07 | 111.29 / 2.565 | daylight | Radiation 4.62 |
| Apr 28 14:04 | 0.672461 | 374.60 | 953.54 | 42.95 | 13.67 | 0.20 | 110.97 / 2.591 | daylight | Radiation 4.88 |
| Apr 28 14:14 | 0.671858 | 375.83 | 930.47 | 42.96 | 13.66 | 0.00 | 111.18 / 2.530 | daylight | Radiation 4.75 |
| Apr 28 14:02 | 0.671843 | 374.00 | 945.97 | 42.71 | 14.27 | 0.00 | 110.44 / 2.526 | daylight | Radiation 4.83 |
| Apr 28 14:24 | 0.671285 | 374.62 | 869.20 | 43.29 | 12.95 | 0.07 | 110.63 / 2.604 | daylight | Radiation 4.40 |
| Apr 28 14:06 | 0.670555 | 374.70 | 960.76 | 42.88 | 13.38 | 0.13 | 110.70 / 2.561 | daylight | Radiation 4.92 |
| Apr 28 14:18 | 0.670307 | 375.01 | 910.75 | 43.55 | 13.08 | 0.00 | 111.59 / 2.529 | daylight | Radiation 4.64 |
| Apr 28 14:46 | 0.669850 | 374.35 | 800.55 | 43.56 | 14.57 | 0.00 | 111.07 / 2.452 | daylight | Radiation 4.02 |
| Apr 28 14:28 | 0.668826 | 375.66 | 858.45 | 43.26 | 13.24 | 0.33 | 110.63 / 2.693 | daylight | Radiation 4.34 |
| Apr 28 14:30 | 0.668230 | 374.74 | 839.63 | 43.40 | 14.43 | 0.00 | 110.84 / 2.389 | daylight | Radiation 4.24 |

Training top scores cluster at the incomplete late-afternoon start and are
driven mainly by wind and RTD spread. Calibration top scores cluster in the
high-radiation 10:36–11:58 period and are mostly driven by RTD spread.
Evaluation top scores cluster at 14:02–14:46 and are all radiation-led.

## Recurring 17:00 transitions

| Timestamp | Partition | Score | Highest-score rank | Percentile | Largest absolute scaled feature | Nearby ±10-minute range / mean |
|---|---|---:|---:|---:|---|---:|
| Apr 27 17:00 | Training | 0.586922 | 13 of 336 | 96.43 | Wind speed, 1.669 | 0.551323–0.584814 / 0.570439 |
| Apr 28 17:00 | Evaluation | 0.571615 | 94 of 337 | 72.40 | Time cosine, 1.482 | 0.551180–0.607237 / 0.581193 |

The first transition is locally elevated; the second lies inside its nearby
score range. They remain recurring operational transitions and are not labelled
as faults.

![Baseline score context review](../outputs/figures/operational/day17_score_context_review.png)

## Recommendations for the next modelling stage

1. Keep the Day 16 model frozen as a reference and do not reinterpret high
   baseline scores as faults.
2. Prioritize acquiring independent days with representative daylight and
   low-light training coverage.
3. Predeclare and compare a model without `minute_of_day_sin` and
   `minute_of_day_cos` to quantify sensitivity to clock-time coverage.
4. Predeclare and compare a model without `rtd_std`; separately inspect the
   five RTD channels before assigning physical meaning.
5. Keep the current Core definition unchanged until those controlled models
   are compared on the same frozen partitions.

## Threshold planning—not selection

A final threshold should not be selected while untouched baseline scores remain
strongly dependent on the short record's operating-regime coverage. A threshold
chosen now could encode time coverage rather than a stable operational policy.
Future strategies to compare include a predeclared calibration percentile, a
robust calibration statistic such as median plus a multiple of MAD, or a
threshold designed to meet a predefined acceptable baseline false-positive
rate. None is selected or calculated here.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m src.anomaly_detection.investigate_baseline_scores
.\.venv\Scripts\python.exe -m pytest -q
```

