# Day 16 initial baseline detector review

## Scope and leakage controls

The initial detector is an Isolation Forest trained on the adopted Day 15 Core
training matrix. One `StandardScaler` was fitted on the 336 training rows only;
the same fitted scaler transformed calibration and evaluation baseline data.
`Timestamp` was retained for traceability but was never passed to the scaler or
model.

No synthetic data or labels were used. No final anomaly threshold, anomaly
labels, precision, recall, F1, or other detector-performance result was
created. The score review describes untouched baseline behavior only.

## Scaler validation

The scaler saw exactly 336 × 9 training values in the approved feature order.
Training population means and standard deviations before scaling were:

| Feature | Mean | Population standard deviation |
|---|---:|---:|
| `Power_Generated` | 122.190351 | 157.708422 |
| `Solar_Radiation` | 83.655515 | 178.367671 |
| `Air_Temp` | 21.960861 | 8.408721 |
| `Relative_Humidity` | 31.914778 | 12.191329 |
| `Wind_Speed` | 0.150794 | 0.428807 |
| `rtd_mean` | 36.315456 | 23.970220 |
| `rtd_std` | 0.697932 | 0.582661 |
| `minute_of_day_sin` | -0.464799 | 0.530846 |
| `minute_of_day_cos` | 0.494096 | 0.507970 |

After scaling, every training-feature mean is within `8.46e-17` of zero and
every population standard deviation is within `2.22e-16` of one. Scaling
introduced no missing or infinite values. The ranges obtained by applying this
same training-fitted scaler to later data are:

| Feature | Calibration scaled range | Evaluation scaled range |
|---|---:|---:|
| `Power_Generated` | -0.686 to 1.786 | -0.721 to 1.787 |
| `Solar_Radiation` | -0.473 to 5.878 | -0.473 to 5.020 |
| `Air_Temp` | -1.176 to 3.291 | -0.690 to 2.580 |
| `Relative_Humidity` | -1.425 to 1.606 | -1.657 to 1.016 |
| `Wind_Speed` | -0.352 to 6.489 | -0.352 to 5.090 |
| `rtd_mean` | -0.655 to 3.166 | -0.492 to 3.145 |
| `rtd_std` | -0.892 to 13.100 | -0.793 to 3.424 |
| `minute_of_day_sin` | -0.023 to 2.759 | -1.008 to 1.426 |
| `minute_of_day_cos` | -2.941 to 0.513 | -2.694 to 0.996 |

The large calibration `rtd_std` maximum is a consequence of applying the
training scale to a calibration value outside the training range. It was not
clipped, removed, or used to refit the scaler.

## Initial Isolation Forest

The configured estimator uses:

| Parameter | Value |
|---|---:|
| `n_estimators` | 300 |
| `max_samples` | `auto` |
| `max_features` | 1.0 |
| `bootstrap` | false |
| `contamination` | `auto` |
| `random_state` | 42 |
| `n_jobs` | -1 |

The model fit input was only the 336 × 9 scaled training matrix. `contamination`
does not encode a presumed anomaly fraction. Although Isolation Forest exposes
native prediction behavior, this stage does not use it as a final operational
decision rule.

The continuous score convention is:

```text
anomaly_score = -IsolationForest.score_samples(X_scaled)
```

Higher values therefore mean more isolated observations. The files contain
only `Timestamp` and `anomaly_score`, not anomaly labels.

## Baseline score distributions

| Partition | Minimum | Maximum | Mean | Median | Population std | P90 | P95 | P97.5 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Training | 0.412857 | 0.666732 | 0.469685 | 0.448086 | 0.055403 | 0.560395 | 0.582952 | 0.608996 | 0.643569 |
| Calibration | 0.501809 | 0.705487 | 0.625319 | 0.645260 | 0.058162 | 0.681459 | 0.695893 | 0.700660 | 0.701019 |
| Evaluation baseline | 0.434172 | 0.672845 | 0.522305 | 0.495549 | 0.073415 | 0.651213 | 0.664643 | 0.668588 | 0.671642 |

Calibration mean score is 0.155635 above training, or 2.81 training score
standard deviations. Evaluation mean is 0.052620 above training, or 0.95
training score standard deviations. The shift is noticeable, especially in
calibration.

This is consistent with the Day 15 finding that the chronological partitions
cover different operating and time-of-day regimes. The record contains only
approximately 33.6 hours and one complete calendar day. Scores may therefore
reflect regime coverage as well as unusual individual observations. This is
not fixed by training on calibration, changing the adopted split, or choosing
a threshold during Day 16; those actions would either introduce leakage or go
beyond this review.

## Recurring 17:00 transitions

| Timestamp | Partition | Score | Nearby ±10-minute range | Nearby mean / median |
|---|---|---:|---:|---:|
| 2022-04-27 17:00 | Training | 0.586922 | 0.551323–0.584814 | 0.570439 / 0.572252 |
| 2022-04-28 17:00 | Evaluation baseline | 0.571615 | 0.551180–0.607237 | 0.581193 / 0.580134 |

The first transition score is slightly above all ten neighboring scores; the
second lies within its neighboring range and below the nearby mean. Neither is
classified as a fault. Both remain known recurring operational transitions for
careful review during later false-positive analysis.

![Continuous baseline anomaly scores](../outputs/figures/operational/day16_baseline_anomaly_scores.png)

## Validation and reproducibility

The run verifies the training-only scaler count, exact feature order, exclusion
of `Timestamp`, finite transformed values, 336 × 9 model fit input,
deterministic scores for repeated seed-42 fits, timestamp-preserving score
files, artifact/metadata hashes, unchanged source datasets, and absence of a
new synthetic evaluation dataset. Reloading the saved scaler and model exactly
reproduces the in-memory training transformations and scores.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.anomaly_detection.train_baseline_detector
.\.venv\Scripts\python.exe -m pytest -q
```

