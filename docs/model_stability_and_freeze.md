# Day 19: Model stability review and prototype freeze

## Scope and safeguards

This review tests whether the Day 18 `NO_TIME_7` recommendation is materially
dependent on the Isolation Forest random seed. It uses only the unchanged 336
training rows and 336 calibration rows. The evaluation partition, synthetic
events, anomaly labels, thresholds, and performance metrics are not used.

Each review run uses the same feature order, a new `StandardScaler` fitted only
on the training partition, and the same Isolation Forest parameters. Only
`random_state` changes between the predeclared seeds 7, 21, 42, 84, and 123.
The existing seed-42 Day 18 scaler and model were reproduced successfully and
were not overwritten.

## Score stability

Scores are continuous Isolation Forest anomaly scores, defined as the negative
of scikit-learn's `score_samples` output so that larger values mean a more
unusual model response. They are not classifications.

| Seed | Train mean | Train SD | Train P95 | Calibration mean | Calibration SD | Calibration P95 | Mean shift | Standardized shift |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.455310 | 0.070243 | 0.594487 | 0.601987 | 0.093560 | 0.713049 | 0.146677 | 2.088126 |
| 21 | 0.454592 | 0.070599 | 0.594630 | 0.600053 | 0.091191 | 0.707807 | 0.145461 | 2.060397 |
| 42 | 0.453758 | 0.072521 | 0.596483 | 0.601381 | 0.089997 | 0.710033 | 0.147623 | 2.035586 |
| 84 | 0.456510 | 0.070031 | 0.593146 | 0.600141 | 0.089392 | 0.703778 | 0.143630 | 2.050960 |
| 123 | 0.455557 | 0.071794 | 0.599635 | 0.602801 | 0.089363 | 0.704319 | 0.147244 | 2.050932 |

All five standardized shifts are below the Day 18 `CORE_9` reference of
2.809118. The seed review does not make adjacent chronological partitions
independent and does not establish deployment performance.

## Calibration rank agreement

Across the ten seed pairs, calibration Spearman rank correlation ranges from
0.979839 to 0.993315 (median 0.987992; mean 0.987260). This is strong agreement
for the overall ordering.

The top-10 overlap averages 5.1 of 10 observations and ranges from 2 to 7. The
top-20 overlap is 20 of 20 for every pair. Therefore the broader high-score
group is stable, while membership and ordering at the very top retain visible
seed sensitivity. These observations are not called anomalies.

## Focused `rtd_std` review

Sixteen calibration rows meet the predeclared condition
`abs(training-scaled rtd_std) > 5`. Under the existing seed-42 `NO_TIME_7`
model, all 16 are in the calibration top 10%, 15 are in the top 5%, and 8 are
among the 10 highest scores. Thus this extreme scaled `rtd_std` condition does
imply a high score for these 16 rows under the top-10% definition, but it does
not account for every observation in the ten highest scores. This is model
sensitivity evidence only. RTD placement, units, and physical meaning remain
unconfirmed.

| Timestamp | Raw `rtd_std` | Scaled `rtd_std` | Score | Percentile | Rank | Solar radiation | Power generated | `rtd_mean` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-04-28 09:04 | 3.698247 | 5.149334 | 0.693384 | 93.155 | 24 | 191.47562 | 307.41102 | 50.250104 |
| 2022-04-28 10:28 | 6.653007 | 10.220484 | 0.717775 | 99.405 | 3 | 927.52808 | 399.49968 | 76.143714 |
| 2022-04-28 10:30 | 5.696349 | 8.578606 | 0.716523 | 98.810 | 5 | 944.27424 | 398.87332 | 67.459179 |
| 2022-04-28 10:32 | 5.624866 | 8.455922 | 0.718099 | 99.702 | 2 | 937.16408 | 400.05884 | 67.962164 |
| 2022-04-28 10:34 | 5.355397 | 7.993442 | 0.715740 | 98.512 | 6 | 940.28328 | 400.41896 | 67.894139 |
| 2022-04-28 10:36 | 6.052713 | 9.190220 | 0.714128 | 97.619 | 9 | 941.32448 | 400.59788 | 76.732802 |
| 2022-04-28 10:38 | 6.081662 | 9.239905 | 0.714426 | 97.917 | 8 | 962.92936 | 403.42476 | 76.580435 |
| 2022-04-28 10:40 | 6.183072 | 9.413951 | 0.711234 | 95.536 | 16 | 957.57144 | 402.80080 | 76.853406 |
| 2022-04-28 10:42 | 6.366040 | 9.727973 | 0.712037 | 97.024 | 11 | 974.26096 | 403.19980 | 77.142072 |
| 2022-04-28 10:44 | 7.675500 | 11.975352 | 0.712519 | 97.321 | 10 | 965.12576 | 402.46044 | 79.991803 |
| 2022-04-28 10:46 | 7.891725 | 12.346451 | 0.717272 | 99.107 | 4 | 984.44904 | 403.08684 | 80.280902 |
| 2022-04-28 10:48 | 8.184886 | 12.849593 | 0.712037 | 97.024 | 11 | 995.51872 | 403.88060 | 79.893958 |
| 2022-04-28 10:50 | 8.288538 | 13.027488 | 0.712037 | 97.024 | 11 | 1007.38860 | 403.48812 | 80.248019 |
| 2022-04-28 10:52 | 8.192809 | 12.863192 | 0.711399 | 96.131 | 14 | 1016.89360 | 402.82560 | 83.897819 |
| 2022-04-28 10:54 | 8.330871 | 13.100143 | 0.711395 | 95.833 | 15 | 1008.39740 | 402.03272 | 84.494138 |
| 2022-04-28 10:56 | 8.301016 | 13.048903 | 0.710753 | 95.238 | 17 | 1019.77620 | 402.46196 | 84.602771 |

## `NO_TIME_7` versus `COMPACT_6`

The existing Day 18 artifacts were reused. Across all calibration rows, score
rank correlation between the two variants is 0.946695. For the 16 focused rows,
removing `rtd_std` causes a median absolute rank change of 11 and a maximum of
134. The `COMPACT_6` model places 11 of these rows in the top 10%, 7 in the top
5%, and 3 in its ten highest scores.

Removing `rtd_std` therefore changes the focused rankings and shifts sensitivity
toward correlated operating variables. It does not improve the declared overall
stability measure: the standardized calibration shift worsens from 2.035586 for
`NO_TIME_7` to 2.149208 for `COMPACT_6`. This evidence does not justify reopening
the bounded Day 18 feature selection, but the RTD uncertainty remains documented.

## Recurring 17:00 transition

The training observation at 2022-04-27 17:00 remains prominent under every
seed. Its rank ranges only from 12 to 17 among 336 training rows.

| Seed | Score | Highest-score rank | Training percentile |
| ---: | ---: | ---: | ---: |
| 7 | 0.594871 | 17 | 95.238 |
| 21 | 0.600733 | 15 | 95.833 |
| 42 | 0.604274 | 13 | 96.429 |
| 84 | 0.609574 | 12 | 96.726 |
| 123 | 0.607759 | 13 | 96.429 |

It continues to be described as a recurring operational transition, not a
fault or anomaly label.

## Predeclared freeze decision

| Check | Predeclared gate | Result | Passed |
| --- | --- | ---: | :---: |
| Overall rank agreement | Minimum pairwise Spearman >= 0.95 | 0.979839 | Yes |
| Top-10 agreement | Mean overlap fraction >= 0.50 | 0.51 | Yes |
| Top-20 agreement | Mean overlap fraction >= 0.65 | 1.00 | Yes |
| Shift versus `CORE_9` | Every seed below 2.809118 | Maximum 2.088126 | Yes |
| RTD dominance | At most 9 of seed-42 top 10 are focused RTD rows | 8 | Yes |
| Sensitivity comparison | `NO_TIME_7` shift below `COMPACT_6` | 2.035586 < 2.149208 | Yes |
| Interpretability | Retain six base operational features | Six retained | Yes |

All predeclared criteria pass using training and calibration evidence only.
The prototype configuration is therefore frozen as:

- `NO_TIME_7`, in the documented seven-feature order;
- `StandardScaler`, fitted only on the 336 training rows;
- `IsolationForest(n_estimators=300, max_samples="auto", max_features=1.0,
  bootstrap=False, contamination="auto", random_state=42, n_jobs=-1)`.

Seed 42 remains the prototype seed for exact reproducibility. The freeze points
to the validated Day 18 seed-42 model and scaler; it does not create another
selected model. Artifact paths and SHA-256 hashes are recorded in
`models/frozen_anomaly_detector_metadata.json`.

## Limitations and next boundary

This is a prototype configuration freeze, not a threshold decision or evidence
of fault-detection performance. The source covers about 33.6 hours and one
complete calendar day, while training and calibration are adjacent portions of
the same short record. The RTD channels lack confirmed placement and units.
Threshold calibration, final synthetic evaluation generation, and performance
evaluation remain separate future stages.
