# Day 20: Anomaly threshold calibration

## Scope

The threshold was selected from the 336 untouched calibration-baseline scores
produced by the frozen seed-42 `NO_TIME_7` detector. The seven-feature order,
training-fitted `StandardScaler`, Isolation Forest, chronological partitions,
and score convention were not changed. No `fit` operation was called.

The saved scaler and model hashes were verified before scoring. Their reproduced
calibration mean, population standard deviation, 95th percentile, and ten
highest timestamp/score pairs exactly match the existing Day 18 results.

Higher scores mean more isolated model responses because the established score
is `-model.score_samples`. A row is flagged only when its score is strictly
greater than the threshold.

## Predeclared candidates

The candidate rules and selection gates were recorded in
`config/anomaly_threshold_calibration.yaml` before comparison. The raw median
absolute deviation (MAD), without a normal-distribution consistency multiplier,
is used for the robust candidates. The calibration score median is 0.630715119
and MAD is 0.055985144.

| Candidate | Threshold | Flagged rows | Alert rate | Events | Typical duration | Maximum duration | Isolated events | Daylight / low light | Extreme RTD / other | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Calibration P95 | 0.710033453 | 17 | 5.060% | 3 | 2 min | 30 min | 2 | 17 / 0 | 15 / 2 | No |
| Calibration P97.5 | 0.713524276 | 9 | 2.679% | 4 | 2 min | 12 min | 3 | 9 / 0 | 7 / 2 | Yes |
| Calibration P99 | 0.717010223 | 4 | 1.190% | 4 | 2 min | 2 min | 4 | 4 / 0 | 3 / 1 | No |
| Median + 1.5 MAD | 0.714692834 | 6 | 1.786% | 3 | 2 min | 8 min | 2 | 6 / 0 | 5 / 1 | No |
| Median + 2 MAD | 0.742685406 | 0 | 0.000% | 0 | 0 min | 0 min | 0 | 0 / 0 | 0 / 0 | No |
| Median + 2.5 MAD | 0.770677978 | 0 | 0.000% | 0 | 0 min | 0 min | 0 | 0 / 0 | 0 / 0 | No |

These counts describe baseline flagged observations, not formally established
false positives. The calibration partition is treated as baseline data but has
not been independently verified to be fault-free.

## Event grouping

Flagged rows exactly two minutes apart belong to the same alert event. Any gap
greater than two minutes starts a new event. No smoothing is applied and
nonconsecutive events are not merged. Event duration is the number of flagged
observations multiplied by the two-minute sampling interval, so a one-row event
has a two-minute duration.

For the selected rule, the four calibration events are:

| Start | End | Flagged observations | Duration |
| --- | --- | ---: | ---: |
| 2022-04-28 10:28 | 2022-04-28 10:38 | 6 | 12 min |
| 2022-04-28 10:46 | 2022-04-28 10:46 | 1 | 2 min |
| 2022-04-28 11:58 | 2022-04-28 11:58 | 1 | 2 min |
| 2022-04-28 13:42 | 2022-04-28 13:42 | 1 | 2 min |

## Context and RTD sensitivity

The existing exploratory context rule is retained: `Solar_Radiation <= 5 W/m²`
is low light; larger values are daylight-like. Every flagged calibration row
under every nonempty candidate is daylight-like. This concentration is reported
but does not trigger model modification at this stage. Calibration itself
contains substantially more daylight-like rows than training and covers
operating conditions not fully represented there.

For the selected threshold, 7 of 9 flagged rows (77.8%) belong to the 16-row
extreme-`rtd_std` group defined by `abs(training-scaled rtd_std) > 5`; two flagged
rows do not. The selected threshold is therefore strongly sensitive to this
group, although it does not act exclusively as an `rtd_std` rule. RTD placement,
units, and physical meaning remain unconfirmed, and no RTD observations were
removed.

## Predeclared selection criteria

A candidate had to:

- flag between 4 and 18 calibration rows;
- produce a calibration baseline alert rate between 1.0% and 5.5%;
- keep isolated events at or below 75% of all events;
- keep extreme-RTD rows at or below 80% of flagged rows;
- be deterministic, reproducible, and selected without evaluation data.

Eligible candidates were ranked by distance from a predeclared 2.5% target
baseline alert rate, followed by isolated-event fraction, RTD concentration,
and declaration order. This avoids automatically selecting the largest
threshold merely because it produces fewer alerts.

P95 failed the RTD-concentration gate. P99 failed the isolated-event gate.
Median + 1.5 MAD failed the RTD-concentration gate, while the two larger MAD
rules produced no alerts. P97.5 was the only candidate to pass every gate and
was also close to the declared 2.5% target.

## Frozen prototype threshold

The frozen threshold is:

- strategy: calibration 97.5th percentile;
- exact value: `0.7135242760182695`;
- comparison: `anomaly_score > threshold`;
- calibration baseline alerts: 9 of 336 rows (2.6786%);
- calibration alert events: 4;
- context: 9 daylight-like, 0 low-light;
- RTD review: 7 extreme-RTD rows and 2 other rows.

The configuration references, but does not modify, the frozen Day 19 detector.
The threshold metadata stores the calibration partition and score-vector hashes,
the frozen detector metadata hash, partition boundaries, and limitations. It
does not contain another trained model.

## Post-freeze evaluation-baseline sanity check

Only after the threshold configuration and metadata were written was the
337-row evaluation-baseline partition loaded and scored. The frozen threshold
flagged 0 rows (0%), producing 0 events in both daylight-like and low-light
contexts. The recurring 2022-04-28 17:00 transition scored 0.581515963 and was
not flagged.

The threshold was not changed in response. These results are descriptive only
and are not an anomaly-detection performance evaluation.

## Limitations

The threshold is calibrated for the current internship prototype and is not a
generally validated operational alarm limit. The available source spans only
about 33.6 hours and one complete calendar day. Training and calibration are
adjacent parts of that record rather than independent operating periods. All
selected calibration alerts occur in daylight-like conditions, and most are
associated with the uncertain extreme-`rtd_std` group. Zero evaluation-baseline
alerts do not demonstrate fault-detection capability. Synthetic evaluation and
precision, recall, or F1 analysis remain future work.
