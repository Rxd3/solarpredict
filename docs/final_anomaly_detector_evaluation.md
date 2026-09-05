# Day 22: Frozen detector evaluation on final synthetic data

## Frozen evaluation contract

This evaluation loads the existing seed-42 `NO_TIME_7` Isolation Forest and its
training-fitted `StandardScaler`. It uses the frozen seven-feature order:

1. `Power_Generated`
2. `Solar_Radiation`
3. `Air_Temp`
4. `Relative_Humidity`
5. `Wind_Speed`
6. `rtd_mean`
7. `rtd_std`

The anomaly score remains `-model.score_samples`, so a larger score is more
isolated. The frozen decision rule is strictly
`anomaly_score > 0.7135242760182695`. All detector, scaler, threshold, and Day 21
dataset hashes were verified. No fit, regeneration, tuning, feature change, or
threshold change occurred.

## Direct-label point evaluation

The strict reference label is `synthetic_anomaly`: it is 1 only on deliberately
modified original-measurement rows. Propagated-effect rows retain reference
label 0. Consequently, a predicted alert on a propagated row would count as a
false positive in this strict confusion matrix even though it could represent a
causal trailing response.

The detector produced no alerts among 337 evaluation rows:

| Reference / prediction | Predicted normal | Predicted alert |
| --- | ---: | ---: |
| Direct synthetic row | FN = 21 | TP = 0 |
| Reference-negative row | TN = 316 | FP = 0 |

Accuracy is 93.77% but is not emphasized because 316 of 337 reference labels
are negative.

## Concise performance table

| Evaluation level | Metric | Result |
| --- | --- | ---: |
| Point | Precision | 0.0000 |
| Point | Recall | 0.0000 |
| Point | F1 | 0.0000 |
| Point | False-positive rate | 0.0000 |
| Event | Detected / total | 0 / 3 |
| Event | Event-level recall | 0.0000 |

Zero false-positive rate here means that no reference-negative row exceeded the
frozen threshold in this short constructed evaluation. It does not establish a
general operational false-alarm rate.

## Paired baseline comparison

The untouched evaluation baseline was scored with the same already-fitted
scaler and detector for the same 337 timestamps. For each row,
`score_delta = synthetic_score - baseline_score`.

| Context | Rows | Mean delta | Minimum | Maximum | Positive / negative / zero |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 21 | 0.003137 | -0.002154 | 0.025530 | 11 / 10 / 0 |
| Propagated | 64 | 0.000000 | 0.000000 | 0.000000 | 0 / 0 / 64 |
| Unaffected | 252 | 0.000000 | 0.000000 | 0.000000 | 0 / 0 / 252 |

The 64 propagated rows have changed rolling/change features in the full
85-column dataset, but none of those engineered features belongs to the frozen
`NO_TIME_7` input. Their seven model inputs are unchanged, so their synthetic
and baseline scores match exactly. No propagated alert therefore occurs in this
specific frozen configuration.

## Strict event-level evaluation

An event is strictly detected only if at least one directly modified row has a
score strictly above the frozen threshold. All three events were missed, giving
0 detected of 3 (0% event-level recall).

| Scenario | Direct flagged / rows | Coverage | Direct score mean / max | Baseline mean | Mean score delta | First alert / delay | Propagated alerts |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Gradual power degradation | 0 / 15 | 0% | 0.656342 / 0.674314 | 0.656824 | -0.000482 | None | 0 |
| Sustained power reduction | 0 / 5 | 0% | 0.563579 / 0.581482 | 0.554062 | +0.009517 | None | 0 |
| Relative-humidity sensor excursion | 0 / 1 | 0% | 0.483552 / 0.483552 | 0.458022 | +0.025530 | None | 0 |

Because no related direct or propagated row was flagged, detection delays and
last-related-alert timestamps are undefined for every event.

## Per-scenario behavior

### Gradual power degradation

The event directly changes `Power_Generated` among the seven model inputs;
`Array_Voltage` also changes under the generator consistency policy but is not a
frozen input. Scores differ only slightly from the paired baseline: direct
deltas range from -0.002154 to +0.002411 and average -0.000482. The highest
direct score is 0.674314, below the threshold. Detection is neither immediate
nor delayed because no alert occurs.

### Sustained power reduction

This event also directly changes only `Power_Generated` among the frozen inputs.
All five paired deltas are positive, ranging from +0.003993 to +0.015159, with a
mean of +0.009517. The maximum direct score is still only 0.581482. No direct or
propagated alert occurs.

### Relative-humidity sensor excursion

The event directly changes `Relative_Humidity`. Its score rises by 0.025530,
from a paired baseline value of 0.458022 to 0.483552, but remains well below the
threshold. No alert occurs.

These responses show that the frozen model/threshold is not sensitive enough to
flag the three safely placed synthetic cases. The evaluation does not justify
changing the already-frozen configuration during this stage.

## Protected 17:00 transition

The 21 rows from 2022-04-28 16:40 through 17:20 contain no direct or propagated
synthetic effect. Synthetic and paired-baseline scores match exactly throughout
the interval, and no row exceeds the threshold. At exactly 17:00 the score is
0.581515963, the score delta is zero, and the row is not flagged. It remains a
recurring operational transition, not a synthetic event or verified fault.

## Detector alert events

The Day 20 grouping rule is retained: predicted rows exactly two minutes apart
belong to the same alert event, with no smoothing or distant merging. Because
there are zero predicted rows, there are zero detector alert events. Thus no
alert event overlaps a direct event, a propagated tail, or unaffected baseline
data.

## Limitations

The evaluation uses controlled synthetic anomalies and only three successfully
placed events. The results therefore demonstrate prototype behavior and do not
establish real-world photovoltaic fault-detection performance.

The three cases cover only two power-reduction patterns and one
relative-humidity excursion. Four configured scenarios were not represented.
The operational source covers about 33.6 hours, the threshold was calibrated on
an adjacent short baseline period, and the synthetic signatures are not verified
equipment faults. Zero detections are an important prototype finding, not a
reason to alter the frozen model or threshold inside this evaluation stage.
