# Final operational anomaly-module review

## Decision and scope

The operational anomaly-detection component is **prototype complete** and is
closed for further model tuning after Day 23. This review uses the existing Day
22 results; it does not refit the scaler, retrain the Isolation Forest, alter the
seven inputs, change the threshold, or regenerate the synthetic evaluation.

The frozen score is `-IsolationForest.score_samples`, so larger values are more
isolated. `ALERT` is assigned only when the score is strictly greater than
`0.7135242760182695`.

## Evidence from the three controlled events

The table covers directly modified rows only. "Training SD" means the absolute
input change divided by the saved training-only `StandardScaler.scale_`; it is a
scale comparison, not a probability or proof that the change is physically
small. The score distance is `threshold - maximum synthetic score`, so every
positive value remained below the alert boundary.

| Event | Baseline score range | Synthetic score range | Maximum score increase | Distance below threshold | Direct frozen input change | Largest change on training scale |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Gradual power degradation | 0.634509–0.674162 | 0.633461–0.674314 | +0.002411 | 0.039211 | `Power_Generated` only; −3.169 to −47.862 | 0.303 SD |
| Sustained power reduction | 0.532801–0.575588 | 0.547960–0.581482 | +0.015159 | 0.132042 | `Power_Generated` only; −47.477 to −47.849 | 0.303 SD |
| Relative-humidity sensor excursion | 0.458022–0.458022 | 0.483552–0.483552 | +0.025530 | 0.229973 | `Relative_Humidity` only; +4.491514 | 0.368 SD |

For the gradual event, the average absolute power change was 0.161 training SD;
its mean score change was actually −0.000482, with individual changes from
−0.002154 to +0.002411. For the sustained event, the average absolute power
change was 0.302 training SD and the mean score increase was +0.009517. The
single humidity perturbation was 0.368 training SD and raised its score by
+0.025530. Thus, none of these direct changes was larger than 0.37 saved
training-scale units in its affected input.

## Evidence-based interpretation of zero detection

Several observations plausibly explain the weak response, without establishing
one definitive cause:

- The perturbations were moderate on the saved training scale: at most 0.303 SD
  for generated power and 0.368 SD for relative humidity. Isolation Forest
  evaluates the joint seven-dimensional position, not the semantic label of a
  scenario, and all event scores remained below the frozen boundary.
- The threshold was intentionally conservative. It is the calibration score's
  97.5th percentile. The calibration baseline itself contained 9 of 336 rows
  above it and relatively high scores associated largely with extreme
  `rtd_std`; it was not independently verified fault-free.
- The gradual event began in an already high-scoring baseline region (up to
  0.674162), but the synthetic change increased the maximum by only 0.000152 at
  that timestamp and by at most 0.002411 on any direct row. High baseline score
  alone therefore did not create a crossing.
- The frozen `NO_TIME_7` inputs are the five raw measurements plus `rtd_mean`
  and `rtd_std`. Rolling, first-difference, rate-of-change, and deviation columns
  are not model inputs. The 64 documented propagated engineered-feature effects
  consequently produced exactly zero score change for this detector.
- Model coverage is narrow: 336 training rows drawn from a source record of only
  about 33.6 hours overall. That limits evidence about normal variability and
  generalization. It does not by itself prove why a particular event was missed.

These are controlled synthetic results, not real photovoltaic fault-validation
results. Day 22 correctly preserved zero predicted rows, 0/3 detected events,
and a zero false-positive rate on its strict direct-row reference labels.

## Intended prototype purpose

The current module can reasonably provide:

- a continuous isolation/anomaly score for rows containing all seven inputs;
- a reproducible threshold-based `NORMAL` or `ALERT` prototype status;
- paired comparison with recent or selected baseline operating behavior when a
  caller supplies that context; and
- deterministic grouping/logging of threshold crossings at the observed
  two-minute cadence.

It cannot currently claim:

- validated classification or diagnosis of real photovoltaic faults;
- reliable detection of every moderate abnormal change;
- robustness across seasons, sites, hardware configurations, or unseen
  operating regimes; or
- physical diagnosis of the RTD sensors, whose placement and units remain
  unconfirmed.

## Frozen inference interface

[`src/anomaly_detection/inference.py`](../src/anomaly_detection/inference.py)
provides `load_frozen_detector`, `score_observation`, `score_dataframe`, and
`group_alert_events`. Loading verifies the saved scaler, model, detector,
threshold, and metadata hashes. Scoring selects the exact seven features in the
frozen order, rejects missing/non-finite inputs, calls only `transform` and
`score_samples`, and returns score, threshold, signed margin, and strict status.

The example dashboard feed is produced from the untouched evaluation baseline.
It contains 337 real-pipeline rows, zero `ALERT` rows, and zero grouped events;
that outcome is preserved rather than replaced with fabricated alerts.

