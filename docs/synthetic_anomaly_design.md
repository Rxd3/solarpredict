# Day 12 synthetic anomaly design and evaluation plan

## Scope and evidence

This is a design specification. No anomalies have been injected, no dataset has
been relabeled, and no preprocessing or detector has been fitted in this stage.
All seven example scenarios in
[`config/synthetic_anomaly_scenarios.yaml`](../config/synthetic_anomaly_scenarios.yaml)
are disabled, including the top-level switch. There is no injection command yet.

The Day 11 review established a 1,009-row, 81-column master feature dataset, an
exact 120-second cadence, and approximately 33.6 hours of coverage. The source
is treated as a normal-like prototype baseline, not certified fault-free data.
Timezone: not specified by dataset source; timestamps remain timezone-naive.
The proposed transformations are controlled measurement perturbations. They do
not establish that a specific physical photovoltaic fault has occurred, or
represent the full range of real faults.

This plan uses the existing [feature review](model_feature_review.md) and
[recurring-transition evidence](../outputs/recurring_transition_review.json).
All ranges below are configurable project design choices, not measured failure
distributions, validated detector thresholds, or equipment ratings.

## Scenario catalogue

Let P, V, I, and S denote untouched generated power, array voltage, array current,
and solar radiation at the same row. Each event draws one magnitude and an integer
sample count n from its configured ranges. The same drawn magnitude applies
throughout a step event; only the gradual scenario varies its factor over time.
Return to the untouched trajectory after the event. The recovery edge is part of
the evaluation context and must be recorded.

| Scenario ID and name | Directly affected variables | Proposed severity | Samples / duration | Transformation | Type, testing purpose, and limitation |
|---|---|---|---|---|---|
| `sudden_power_drop` — Sudden power drop | `Power_Generated`, `Array_Voltage` | 10–30% reduction | 1–3 / 2–6 min | Multiply P and V by 1-a | Operational hypothesis; tests brief downward departures and response time. Natural operating changes can resemble this shape. |
| `sustained_power_reduction` — Sustained power reduction | `Power_Generated`, `Array_Voltage` | 10–25% reduction | 5–15 / 10–30 min | Multiply P and V by 1-a | Operational hypothesis; tests persistent moderate departure. Same transformation as a drop, with a deliberately separate duration range; not proof of a distinct fault mechanism. |
| `temporary_near_zero_power` — Temporary zero or near-zero interval | `Power_Generated`, `Array_Voltage` | Retain 0–2% of baseline | 1–3 / 2–6 min | Multiply P and V by retained fraction q | Operational or sensor hypothesis; tests near-loss-of-output response. This is intentionally severe and must be reported separately so it does not dominate the benchmark. Exact zero can be an explicit configured case; a continuous random draw need not select zero. |
| `sudden_power_spike` — Sudden power spike | `Power_Generated`, `Array_Voltage` | 10–30% increase | 1–2 / 2–4 min | Multiply P and V by 1+a | Operational or sensor hypothesis; tests short upward excursions. Capacity limits are unavailable, so do not claim physical feasibility from the historical maximum. |
| `gradual_power_degradation` — Gradual power degradation | `Power_Generated`, `Array_Voltage` | Final reduction 10–25% | 15–30 / 30–60 min | Multiply by 1-a*j/n for j=1..n | Operational hypothesis; tests a slow ramp versus abrupt changes. A one-hour ramp is a prototype stress case, not a simulation of long-term panel ageing. Returning at the end introduces a recovery edge. |
| `radiation_power_inconsistency` — Radiation/power inconsistency | `Solar_Radiation` only | 10–25% radiation increase | 3–10 / 6–20 min | S'=S*(1+a); P, V, I unchanged | Sensor or relational hypothesis; tests a changed radiation/power relationship, distinct from another power reduction. Normal P/S behavior is already variable; no constant efficiency or causal power response is assumed. |
| `sensor_excursion` — Sensor spike or excursion | Exactly one of `Air_Temp`, `Relative_Humidity` per event | Signed offset: 1–3 °C or 3–8 RH percentage points | 1–3 / 2–6 min | x'=x+sign*magnitude | Sensor hypothesis; tests an isolated environmental measurement excursion. These can resemble real environmental movement; do not interpret detection as sensor-fault diagnosis. |

Durations use sample support: n rows at two-minute cadence represent n*2 minutes;
the first-to-last timestamp difference is (n-1)*2 minutes. Future metadata must
record both the inclusive end timestamp and the exclusive end (start+n*2 minutes).
Severity percentages refer to each row's untouched value, not the event's first
value or an already modified value. All ranges are inclusive for integer sample
counts. Sensor signs are sampled once per event with equal probability.

## Context and consistency rules for future injection

These rules are proposed only and are not applied during Day 12.

- Work in a new evaluation copy with a separate destination. Require distinct
  resolved input/output paths and verify source hashes before and after writing.
  Never overwrite the raw CSV, cleaned CSV, or master feature CSVs.
- Establish chronological training, calibration, and evaluation boundaries first.
  Events must fit entirely inside the evaluation period. Read prior history for
  causal feature computation only; do not fit on evaluation rows or labels.
- Check eligibility against the untouched baseline for the entire event window.
  Require finite affected inputs, unique ordered timestamps, and uninterrupted
  120-second sampling. Reject an incomplete or ineligible window; do not shorten
  it silently or move it into training data.
- For the six power/daylight scenarios, require baseline `low_light_context=0`,
  `Solar_Radiation > 100 W/m^2`, and `Power_Generated >= 50 W` on every event row.
  These conservative placement floors are intentionally stricter than the
  existing 5 W/m^2 low-light flag and are configurable. Do not recalculate
  eligibility from the perturbed measurements.
- Sensor excursions may occur in either lighting context. Only finite temperature
  or humidity readings are candidates. Reject a proposed humidity offset if any
  resulting value falls outside 0–100% RH; record that rejection rather than clip.
  RTD, wind-direction, and other uncertain sensor encodings are excluded initially.
- Exclude any window intersecting ±20 minutes around 17:00 on April 27 or April
  28. This separates the initial injection experiments from the two already
  documented recurring transitions. Retain those observations in untouched controls
  and report their alert behavior separately; they are not verified faults.
- Initially allow one scenario family per evaluation copy, no overlapping events,
  at least 60 minutes from one event's exclusive end to the next event's start,
  and at most 5% directly modified evaluation rows per copy. These are scheduling
  ceilings, not quotas. Long events may be impossible in the small held-out period.
  Skip and record the reason; do not relax rules or force every scenario to fit.
- Enumerate eligible start rows in chronological order and draw starts using the
  recorded seed. A manual start override must pass the same checks and be logged.
  Begin experiments with low severity values as well as larger values; do not
  select only the cases that produce easy detections.

The original power is almost equal to V*I. For the five power-changing scenarios,
multiply P and V by the same factor while keeping I fixed. This preserves the
existing approximate algebraic relationship (the small residual scales too),
so the first detector cannot exploit an accidental broken identity. This is a
bookkeeping choice, not a circuit simulation: actual voltage/current responses
need equipment-specific evidence. Independent power-sensor corruption could be
a separately named future experiment, but is not silently mixed into these cases.

Modify original measurement columns in a copy of the cleaned evaluation data;
then recompute the existing basic, trailing rolling, change, and deviation features
causally inside that copy. Never alter a derived feature independently or keep
stale rolling summaries after changing its source. Use unchanged earlier rows as
history where available. Time features and RTD summaries remain unchanged for
the initial scenarios. Keep existing denominator safeguards and their NaNs.

Later rolling/change experiments can have derived-feature effects after an event
ends, including the first recovery difference and trailing-window tails. Record
these influenced rows separately in event metadata; do not extend direct anomaly
labels to them. The initial Core set has no rolling features, so this complication
does not change its point-label semantics. For later extended comparisons, report
tail alerts separately as well as in the strict unmodified-row alert count.

## Future label schema

Only future evaluation copies receive these columns; they never enter the model
feature matrix. Keep a stable source-row reference in the evaluation manifest.

| Column | Type | Untouched row | Directly modified row |
|---|---|---|---|
| `synthetic_anomaly` | int8 | 0 | 1 |
| `synthetic_anomaly_type` | string | `none` | Exact scenario ID from the configuration |
| `synthetic_anomaly_id` | nullable string | null | Deterministic event ID shared by all rows of that event |

A positive label means at least one designated original measurement was actually
changed. Reject numerical no-ops. The gradual formula uses j=1..n so its first
row is changed too. Baseline label 0 means unmodified by the generator, not a
verified healthy physical system. No event overlaps are supported initially.
Use IDs such as a configuration/baseline/seed-derived run prefix plus a
chronological event number. Do not embed wall-clock time in deterministic IDs.

## Reproducibility contract

The future generator must accept a seed (example 42), keep an explicit local
NumPy Generator using PCG64, and record the bit-generator type and dependency
versions. Draw integer duration uniformly from the allowed inclusive sample range
and severity uniformly from its continuous range; document draw order and all
rejections. Store scenario processing order and eligible-start ordering so file
system order cannot change a run. Separate copies use recorded seeds/substreams.

Create a future JSON manifest recording baseline and configuration hashes,
generator version, dependency versions, seed/RNG, exact split boundaries,
scenario/event IDs, source row positions, timestamps, duration, severity, selected
variable, original and modified values (including coupled voltage), context checks,
rejected/skipped candidates, feature-influence rows, and output SHA-256. Preserve
the resolved configuration with each run. Canonical CSV formatting and the same
baseline/config/code/environment must reproduce the same values, labels, event
IDs, and output hash; a seed alone is not sufficient. No manifest or synthetic
dataset is produced during this design stage.

## Planned model-data workflow

1. Use the untouched normal-like observations as the baseline and record their
   identity. Define ordered, disjoint training, calibration, and final evaluation
   periods, checking that calibration/evaluation have relevant daylight windows.
   Do not randomly split adjacent rows or fit on a row that later becomes a test
   copy. Exact boundaries remain unset pending a feasibility review.
2. Prepare exactly the Core nine inputs: `Power_Generated`, `Solar_Radiation`,
   `Air_Temp`, `Relative_Humidity`, `Wind_Speed`, `rtd_mean`, `rtd_std`,
   `minute_of_day_sin`, `minute_of_day_cos`. Exclude all synthetic label/ID fields.
3. Fit any scaling or preprocessing on baseline training observations only.
   The current Core inputs have no missing values and require no rolling warm-up
   exclusion. Investigate future missingness before defining train-only imputation.
4. Train the initial unsupervised detector on baseline training data. Select any
   alert threshold using separate baseline calibration data and a declared alert
   budget. Freeze model, preprocessing, and threshold before final evaluation.
5. Generate synthetic events only in copies of the held-out evaluation period.
   Keep an untouched control over exactly those timestamps. Development scenarios,
   if used for tuning later, must use distinct periods/seeds from the final test.
6. Apply the frozen preprocessing and detector to each evaluation copy and the
   untouched control. Do not refit or select thresholds using final synthetic labels.
7. Compare scores/detections with the synthetic labels, with results broken down
   by scenario, severity, duration, seed, and lighting context.

The 33.6-hour record has only one complete daily cycle. It may not support three
representative periods, all seven scenarios, or meaningful generalization tests.
Record unavailable scenarios/splits explicitly. Obtain more independent days for
credible evaluation. If a future demonstration deliberately reuses training rows
as perturbed test copies, name it an in-sample sensitivity check and do not report
it as held-out performance. That is not the default evaluation plan.

## Evaluation definitions

For point-level evaluation, let y=1 denote a directly modified row and predicted
positive mean its score crossed the previously frozen threshold. TP, FP, TN, FN
are row counts against these synthetic labels:

- Precision = TP/(TP+FP).
- Recall = TP/(TP+FN).
- F1 = 2*TP/(2*TP+FP+FN).
- False-positive count = FP; false-positive rate = FP/(FP+TN).

Report confusion counts and metric denominators. If a denominator is zero,
report null/undefined and the reason rather than substituting a perfect score.
Alerts on y=0 are false positives relative to the injection labels only; untouched
data is not certified fault-free. On the paired unmodified control, report alert
count/rate and alerts around the recurring transitions. These controls help
distinguish baseline behavior from changes induced by injection.

For event-level evaluation, an event is detected if at least one positive
prediction occurs on one of its directly modified rows. Report detected count,
missed count, and event recall = detected/total injected events. With no injected
events, event recall is undefined. Use no early/late tolerance for the primary
result. Report first-hit delay from event start and the fraction of event rows
detected; any later tolerance experiment must be separately named. Count each
event once. Do not convert one hit into positives for every event row when
computing point metrics (no point adjustment). Event recall by itself can favor
long windows or overly frequent alerts, so always show point false positives,
event duration, and delay beside it.

Report per-scenario results and macro summaries alongside pooled point metrics;
otherwise long reductions or easy near-zero events could dominate the score.
Record synthetic prevalence and actual event counts for each copy. Repeated
seeds on the same baseline measure injection variability, not independent evidence
about many real days. Synthetic performance is not proven performance on real
photovoltaic faults; no performance numbers are claimed in this plan.

## Validation and deferred choices

Lightweight tests parse YAML safely, check all switches remain disabled, validate
parameter/duration ranges and affected columns, verify the Core list and label
schema, and ensure every configured scenario has a documentation entry. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

For the Day 12 handoff, compare source CSV hashes and the data/model/output file
inventory before and after the work. Existing schemas must contain no new labels;
`models/` must contain no trained model. No injector is implemented here.

No user decision is needed to complete this design. Before later experiments,
review the proposed placement floors and severity ranges against company context,
confirm equipment limits if available, and choose feasible chronological split
boundaries (or acquire more days). The example seed is 42; future runs should
include predeclared additional seeds. These are future experiment choices,
not permission to inject events during Day 12.
