# Day 21: Final synthetic evaluation dataset

## Scope and reproducibility

The final synthetic evaluation dataset uses random seed `2026`, deliberately
different from the Day 13 engineering-validation seed `42`. All seven existing
scenario families were attempted using the unchanged severity and duration
ranges from `config/synthetic_anomaly_scenarios.yaml`.

Direct modifications are restricted to the frozen evaluation period from
2022-04-28 13:56 through 2022-04-29 01:08, inclusive. No original or engineered
value in training or calibration was modified. Required measurements had to be
finite, direct windows had to have continuous two-minute sampling, and direct
events could not overlap. Power-related events retained the requirements
`Solar_Radiation > 100 W/m²` and `Power_Generated >= 50 W`, together with the
existing 60-minute inter-event gap and causal-tail guards.

The frozen scaler, detector, feature order, chronological split, and threshold
were not changed. The dataset was not scaled or scored, and no performance
metric was calculated.

## Protected recurring transition

The inclusive interval from 2022-04-28 16:40 through 17:20 contains 21 rows.
All original measurements are byte/numerically unchanged after generation, no
row has a direct synthetic label, and all 21 rows have
`synthetic_effect_context = none`. The later sustained reduction begins at
17:24, outside the protected interval.

## Full-history recomputation

Generation starts with the complete 1,009-row untouched feature sequence.
Original measurements are modified only at permitted evaluation timestamps,
then the existing basic, rolling, and change-feature functions are run causally
over the complete sequence. Only afterward is the 337-row evaluation region
extracted.

This preserves the history preceding 13:56. As a validation comparison,
recomputing from the isolated evaluation slice would change 1,847 derived cells,
largely because early rolling/change features would lose preceding observations.
That isolated result is not used in either final output.

The full recomputation matches the generated dataset, including 17 basic, 30
rolling, and 20 change features. Power and voltage use the existing equal-factor
scaling policy while current remains unchanged. RTD summary features remain
consistent with the unchanged RTD channels.

## Direct and propagated effects

The output contains four synthetic-context columns:

- `synthetic_anomaly`: 1 only when an original operational or sensor value was
  deliberately modified; otherwise 0.
- `synthetic_anomaly_type`: scenario name for direct rows and `none` otherwise.
- `synthetic_anomaly_id`: event identifier for direct rows and null otherwise.
- `synthetic_effect_context`: `direct`, `propagated`, or `none`.

A propagated row retains all original measurements and
`synthetic_anomaly = 0`, but at least one causal engineered feature differs due
to an earlier direct event. Across the evaluation output there are 21 direct
rows, 64 propagated rows, and 252 unaffected rows.

## Successfully placed events

| Event | Scenario | Start | End | Rows | Duration | Severity | Propagated rows |
| --- | --- | --- | --- | ---: | ---: | --- | ---: |
| `86b56a024699-01` | gradual power degradation | 2022-04-28 14:44 | 2022-04-28 15:12 | 15 | 30 min | final reduction 0.126840 | 29 |
| `86b56a024699-02` | sustained power reduction | 2022-04-28 17:24 | 2022-04-28 17:32 | 5 | 10 min | reduction 0.170090 | 29 |
| `86b56a024699-03` | relative-humidity sensor excursion | 2022-04-29 00:56 | 2022-04-29 00:56 | 1 | 2 min | +4.491514 percentage points | 6 |

The two power events account for 20 daylight-like direct rows. The sensor
excursion contributes one low-light direct row. Event metadata includes original
and modified summaries, complete direct timestamps and indices, context checks,
and propagated timestamps.

## Skipped scenario attempts

Four scenario types were not forced into the data:

- `radiation_power_inconsistency`;
- `sudden_power_drop`;
- `temporary_near_zero_power`;
- `sudden_power_spike`.

For each, the generator recorded
`no_window_meets_context_gap_exclusion_and_tail_rules`. After placing the longer
windows first, no remaining start satisfied the unchanged evaluation-only,
daylight/power, 60-minute gap, protected-window, and causal-tail requirements.
The row budget still had capacity, so these are placement-constraint skips, not
budget failures. Constraints and scenario ranges were not weakened to manufacture
seven successful events.

## Output files

`data/model_ready/final_synthetic_evaluation.csv` has 337 rows and 85 columns:
the 81 useful source/engineered columns plus the four synthetic label/context
columns.

`data/model_ready/final_synthetic_evaluation_no_time_7.csv` has 337 rows and 12
columns: timestamp, the exact frozen seven-feature order, and the four synthetic
label/context columns. It is intentionally unscaled and contains no detector
score or prediction.

Both outputs have exactly the same timestamps as
`data/model_ready/baseline_evaluation.csv`.

## Numerical and source validation

All three generated event severities and durations are within their unchanged
configuration ranges. Direct event windows do not overlap, duration is positive,
humidity remains between 0 and 100, and no infinity is present. No generated
value is outside the range observed in the available baseline dataset.

Expected NaNs occur only in features governed by the existing denominator
safeguards:

| Feature | NaN count |
| --- | ---: |
| `solar_radiation_relative_change` | 187 |
| `solar_deviation_ratio_30m` | 180 |

These NaNs are reproduced by the existing causal feature pipeline. They were
not imputed.

Every protected source and all frozen detector/scaler/threshold hashes remained
unchanged. Direct labels exactly match rows with modified original measurements,
while propagated and unaffected rows preserve their source measurements.

## Limitation

This dataset provides controlled synthetic evaluation cases. It does not
establish that the simulated signatures are equivalent to verified real
photovoltaic faults. Only three of seven attempted scenarios could be placed
under the fixed constraints in this short evaluation interval. Future detector
application must report that coverage limitation and must not generalize results
to unrepresented fault types.
