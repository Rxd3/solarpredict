# Day 14 synthetic prototype review and chronological split planning

## Scope

This review compares the 1,009 x 84 Day 13 synthetic prototype with its
unchanged 1,009 x 81 baseline. It reviews all seven seed-42 events and proposes
chronological boundaries for later modelling. It does not create split datasets,
fit scaling, train a detector, calibrate a threshold, or calculate model
performance.

The operational record covers 2022-04-27 15:32 through 2022-04-29 01:08: a
33.6-hour timestamp span and approximately 33.63 hours of two-minute sample
support. The first and last calendar days are incomplete, and only April 28 is a
complete calendar day. Timestamps remain timezone-naive because the source does
not specify a timezone.

Synthetic events are controlled prototype cases, not verified photovoltaic
faults. The current prototype was created to validate the generator. It must not
automatically become the final model test dataset.

## Review of all seven events

Ranges cover the directly modified rows and all affected original variables.
All seven events affect downstream rolling or change features. Each has 29 later
propagated rows because the longest causal rolling window contains 30 samples;
the event remains in that window for the following 29 rows.

| Anomaly ID | Scenario and inclusive timestamps | Direct rows | Affected variables | Baseline range | Synthetic range | Changed derived columns / propagated rows | Special observation |
|---|---|---:|---|---|---|---:|---|
| `6c1a54b3359e-01` | `gradual_power_degradation`, Apr 28 10:38–11:26 | 25 | Power, voltage | P 391.508040–403.880600; V 73.078736–75.359576 | P 306.905734–399.937663; V 57.286903–74.625334 | 22 / 29 (11:28–12:24) | A gradual ramp followed by an artificial return to the baseline trajectory. |
| `6c1a54b3359e-02` | `sustained_power_reduction`, Apr 28 14:00–14:08 | 5 | Power, voltage | P 373.688160–375.325520; V 69.804824–70.113056 | P 288.192162–289.454911; V 53.834200–54.071912 | 21 / 29 (14:10–15:06) | Persistent five-row reduction followed by a recovery edge. |
| `6c1a54b3359e-03` | `radiation_power_inconsistency`, Apr 28 17:22–17:34 | 7 | Solar radiation | 113.150280–139.882440 | 126.063737–155.846748 | 17 / 29 (17:36–18:32) | Radiation changes while power, voltage, and current remain at baseline. |
| `6c1a54b3359e-04` | `sudden_power_drop`, Apr 28 12:38–12:42 | 3 | Power, voltage | P 373.854560–374.366960; V 69.821168–69.915712 | P 279.557994–279.941153; V 52.210319–52.281016 | 20 / 29 (12:44–13:40) | Power and voltage use the same reduction factor. |
| `6c1a54b3359e-05` | `temporary_near_zero_power`, Apr 28 08:22–08:26 | 3 | Power, voltage | P 299.899540–300.261840; V 55.624412–55.692528 | P 0.768424–0.769353; V 0.142525–0.142699 | 19 / 29 (08:28–09:24) | Produces three expected safeguarded relative-change NaNs. |
| `6c1a54b3359e-06` | `sudden_power_spike`, Apr 28 15:34 | 1 | Power, voltage | P 389.069400; V 72.665816 | P 456.829573; V 85.321266 | 19 / 29 (15:36–16:32) | The power value is above the maximum observed in the available baseline dataset. |
| `6c1a54b3359e-07` | `sensor_excursion`, Apr 28 02:26–02:28 | 2 | Relative humidity | 43.420420–43.560464 | 50.534228–50.674272 | 4 / 29 (02:30–03:26) | Seed 42 selected humidity; every other original measurement remains unchanged. |

`P` means `Power_Generated` and `V` means `Array_Voltage`. The complete JSON
lists every actually changed downstream column. Differences in the number of
changed columns arise from the affected source and whether an event changes a
rolling minimum or maximum, not from a model response.

## Direct and propagated effects

A **direct anomaly row** has at least one original sensor or operational
measurement deliberately modified. These 46 rows keep the existing
`synthetic_anomaly=1`, exact scenario name, and event ID.

A **propagated feature effect** is a later row whose 13 original measurements
and timestamp are unchanged but whose causal rolling or change features differ
because an earlier event remains in their history. There are 203 such rows. The
first is 2022-04-28 02:30 and the last is 2022-04-28 18:32. The remaining 760
rows have neither direct nor propagated changes.

| Effect context | Rows | Existing direct label |
|---|---:|---:|
| `none` | 760 | 0 |
| `direct` | 46 | 1 |
| `propagated` | 203 | 0 |

For a future evaluation copy, add a separate non-model field named
`synthetic_effect_context` with values `none`, `direct`, or `propagated`;
`direct` takes precedence. Do not change the meaning of `synthetic_anomaly` and
do not add this field to the current prototype during review. The machine-readable
report calculates this context in memory only.

Strict point metrics compare alerts with directly modified rows. A detector that
uses rolling/change features may reasonably react on propagated rows after a
direct event ends, so those alerts count as false positives under the strict
point label. Report them separately as propagated-context alerts. Do not relabel
them after observing scores. Event-level detection can still count an event once
when at least one direct row is detected.

## Synthetic spike review

At 2022-04-28 15:34, baseline power is 389.069400 W and the synthetic value is
456.829573 W. The recalculated increase is 17.415960%, equal to the sampled
severity within the configured 10–30% range.

The available baseline has mean 183.151609 W, median 276.001580 W, population
standard deviation 166.216591 W, 90th/95th/99th percentiles of 392.801640,
403.154616, and 432.269834 W, and a maximum of 438.556840 W. The synthetic value
is 18.272733 W, or 4.166560%, above that observed maximum. In the ten neighboring
baseline rows within ±10 minutes (excluding the event row), power ranges from
384.942000 to 390.544840 W with mean 388.215848 W.

The transformation is mathematically consistent. Its neutral interpretation is
**above the maximum observed in the available baseline dataset**. Equipment
capacity is unknown, so this review makes no capacity claim.

## Near-zero event and safeguarded NaNs

The baseline has one NaN in `power_generated_relative_change`; the prototype has
four. The three additions all follow the existing rule that relative change is
undefined when absolute previous power is below the 1 W denominator floor:

| Timestamp | Previous synthetic power | Current synthetic power | Direct label |
|---|---:|---:|---:|
| 2022-04-28 08:24 | 0.768906 W | 0.768424 W | 1 |
| 2022-04-28 08:26 | 0.768424 W | 0.769353 W | 1 |
| 2022-04-28 08:28 | 0.769353 W | 301.216400 W | 0 |

The final row is the recovery row: its measurement is untouched, but its
previous value is synthetic and below the floor. Returning NaN prevents a huge
or infinite ratio. The complete prototype contains zero infinities. This is the
documented Day 10 safety policy working as intended.

The first Core model should continue to exclude power/solar relative-change and
normalized-ratio features. None of the established Core nine inputs contains
these fields. A later daylight-specific experiment may assess them separately.

## Chronological split options

The intervals below are contiguous and half-open conceptually: the next region
begins at the listed next two-minute sample. Every option covers positions
0–1008 exactly once, with no overlap, duplicate, or omitted row. Hours are sample
support (`rows x 2 / 60`), while the inclusive timestamp span is two minutes less
than support.

### Option A: larger training period

| Region | Inclusive boundaries | Rows | Support hours | Daylight / low-light rows | Radiation >100 W/m² |
|---|---|---:|---:|---:|---:|
| Training baseline | Apr 27 15:32 – Apr 28 11:30 | 600 | 20.00 | 270 / 330 | 191 |
| Calibration baseline | Apr 28 11:32 – 14:30 | 90 | 3.00 | 90 / 0 | 90 |
| Synthetic evaluation | Apr 28 14:32 – Apr 29 01:08 | 319 | 10.63 | 131 / 188 | 94 |

Advantages: the scaler and detector would receive 600 rows spanning late-day,
night, dawn, and morning. Evaluation contains daylight, the second recurring
17:00 transition, and night.

Disadvantages: calibration has only 90 midday rows and no low-light context. A
single threshold chosen there may transfer poorly into nighttime evaluation.
The regions still share one short daily record. The first 17:00 transition is in
training and the second is in evaluation.

### Option B: approximately balanced thirds

| Region | Inclusive boundaries | Rows | Support hours | Daylight / low-light rows | Radiation >100 W/m² |
|---|---|---:|---:|---:|---:|
| Training baseline | Apr 27 15:32 – Apr 28 02:42 | 336 | 11.20 | 100 / 236 | 64 |
| Calibration baseline | Apr 28 02:44 – 13:54 | 336 | 11.20 | 242 / 94 | 199 |
| Synthetic evaluation | Apr 28 13:56 – Apr 29 01:08 | 337 | 11.23 | 149 / 188 | 112 |

Advantages: all three regions contain both daylight and low-light observations.
Calibration covers pre-dawn, sunrise, morning, and midday, which is more useful
for reviewing a single global alert threshold than Option A's daylight-only
calibration. Evaluation contains afternoon, the second 17:00 transition, and
night.

Disadvantages: detector training receives only 336 rows, its daylight portion is
limited to the first late afternoon, and the three adjacent regions are not
independent days. The first 17:00 transition remains in training and the second
in evaluation.

### Recommendation

Use **Option B** for the next prototype stage because threshold calibration and
evaluation both cover daytime and low-light regimes. The recommendation values
regime coverage over Option A's additional training rows. It is a planning
decision, not a claim that 336 rows are sufficient.

This record has only one complete daily cycle. Neither option gives credible
independence or robust generalization. More independent days remain the proper
solution. Do not move boundaries to accommodate the events already seen in the
seed-42 prototype. Choose/freeze the boundaries first, then regenerate anomaly
copies only inside the evaluation region with predeclared seeds. The current
prototype's event distribution is recorded in JSON for transparency but is not
a reason for either boundary choice. Separate scenario-family evaluation copies
may be necessary because the short test region cannot safely host every event
with all spacing rules in one copy.

![Proposed Option B timeline](../outputs/figures/operational/day14_split_timeline.png)

## Core feature readiness

Population variance is calculated independently within each proposed region.
The Day 11 rule marks a feature near-constant if one value occupies at least 95%
of non-missing rows or population standard deviation divided by absolute mean is
at most 1%. Every Core feature has zero missing values, a finite value on every
row, positive variance, and `variable` status in all six proposed regions. No
scaling is fitted in this review.

### Option A

| Region | Feature | Missing / finite | Population variance | Status |
|---|---|---:|---:|---|
| Training | `Power_Generated` | 0 / 600 | 26001.621299 | variable |
| Training | `Solar_Radiation` | 0 / 600 | 77219.552378 | variable |
| Training | `Air_Temp` | 0 / 600 | 104.581603 | variable |
| Training | `Relative_Humidity` | 0 / 600 | 144.120920 | variable |
| Training | `Wind_Speed` | 0 / 600 | 0.132232 | variable |
| Training | `rtd_mean` | 0 / 600 | 559.463445 | variable |
| Training | `rtd_std` | 0 / 600 | 1.512457 | variable |
| Training | `minute_of_day_sin` | 0 / 600 | 0.552511 | variable |
| Training | `minute_of_day_cos` | 0 / 600 | 0.411013 | variable |
| Calibration | `Power_Generated` | 0 / 90 | 21.618454 | variable |
| Calibration | `Solar_Radiation` | 0 / 90 | 5232.928139 | variable |
| Calibration | `Air_Temp` | 0 / 90 | 3.927077 | variable |
| Calibration | `Relative_Humidity` | 0 / 90 | 5.116195 | variable |
| Calibration | `Wind_Speed` | 0 / 90 | 0.448253 | variable |
| Calibration | `rtd_mean` | 0 / 90 | 15.478079 | variable |
| Calibration | `rtd_std` | 0 / 90 | 0.025625 | variable |
| Calibration | `minute_of_day_sin` | 0 / 90 | 0.046424 | variable |
| Calibration | `minute_of_day_cos` | 0 / 90 | 0.003929 | variable |
| Evaluation | `Power_Generated` | 0 / 319 | 26775.663838 | variable |
| Evaluation | `Solar_Radiation` | 0 / 319 | 52491.876608 | variable |
| Evaluation | `Air_Temp` | 0 / 319 | 82.054802 | variable |
| Evaluation | `Relative_Humidity` | 0 / 319 | 119.409215 | variable |
| Evaluation | `Wind_Speed` | 0 / 319 | 0.145103 | variable |
| Evaluation | `rtd_mean` | 0 / 319 | 1012.033130 | variable |
| Evaluation | `rtd_std` | 0 / 319 | 0.398095 | variable |
| Evaluation | `minute_of_day_sin` | 0 / 319 | 0.142827 | variable |
| Evaluation | `minute_of_day_cos` | 0 / 319 | 0.357354 | variable |

Option A's calibration features are technically variable, but power, RTD spread,
and cosine time context have much narrower variance than in training/evaluation.
This reinforces the regime-coverage concern; the binary status alone is not a
guarantee of representative calibration data.

### Option B

| Region | Feature | Missing / finite | Population variance | Status |
|---|---|---:|---:|---|
| Training | `Power_Generated` | 0 / 336 | 24871.946355 | variable |
| Training | `Solar_Radiation` | 0 / 336 | 31815.025950 | variable |
| Training | `Air_Temp` | 0 / 336 | 70.706595 | variable |
| Training | `Relative_Humidity` | 0 / 336 | 148.628510 | variable |
| Training | `Wind_Speed` | 0 / 336 | 0.183875 | variable |
| Training | `rtd_mean` | 0 / 336 | 574.571450 | variable |
| Training | `rtd_std` | 0 / 336 | 0.339494 | variable |
| Training | `minute_of_day_sin` | 0 / 336 | 0.281797 | variable |
| Training | `minute_of_day_cos` | 0 / 336 | 0.258033 | variable |
| Calibration | `Power_Generated` | 0 / 336 | 20407.471273 | variable |
| Calibration | `Solar_Radiation` | 0 / 336 | 213197.360867 | variable |
| Calibration | `Air_Temp` | 0 / 336 | 208.758043 | variable |
| Calibration | `Relative_Humidity` | 0 / 336 | 164.084665 | variable |
| Calibration | `Wind_Speed` | 0 / 336 | 0.219497 | variable |
| Calibration | `rtd_mean` | 0 / 336 | 1162.294553 | variable |
| Calibration | `rtd_std` | 0 / 336 | 2.362287 | variable |
| Calibration | `minute_of_day_sin` | 0 / 336 | 0.201754 | variable |
| Calibration | `minute_of_day_cos` | 0 / 336 | 0.338076 | variable |
| Evaluation | `Power_Generated` | 0 / 337 | 27767.597694 | variable |
| Evaluation | `Solar_Radiation` | 0 / 337 | 80694.737993 | variable |
| Evaluation | `Air_Temp` | 0 / 337 | 93.680249 | variable |
| Evaluation | `Relative_Humidity` | 0 / 337 | 126.229238 | variable |
| Evaluation | `Wind_Speed` | 0 / 337 | 0.158370 | variable |
| Evaluation | `rtd_mean` | 0 / 337 | 1168.449927 | variable |
| Evaluation | `rtd_std` | 0 / 337 | 0.534032 | variable |
| Evaluation | `minute_of_day_sin` | 0 / 337 | 0.135593 | variable |
| Evaluation | `minute_of_day_cos` | 0 / 337 | 0.406553 | variable |

## Planned modelling order

1. Choose and freeze chronological boundaries. Option B is the current planning
   recommendation; no split files exist yet.
2. Use untouched baseline rows from the training region.
3. Reserve a separate untouched baseline calibration region.
4. Fit the scaler using training rows only.
5. Train the unsupervised detector using training rows only.
6. Select and freeze the alert threshold using calibration rows only.
7. Regenerate synthetic copies with new predeclared seeds and inject only within
   the frozen evaluation region. Keep a paired untouched evaluation control.
8. Evaluate the frozen pipeline against direct labels and report alerts in
   propagated context separately.

No step in this sequence is executed during Day 14. Synthetic performance would
measure response to the configured perturbations, not proven detection of real
photovoltaic faults.

## Reproduction and validation

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.anomaly_detection.review_synthetic_prototype
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

The review validates direct/propagated accounting, event metadata, spike math,
near-zero safeguards, chronological non-overlap, exact one-time row coverage,
Core readiness, source hashes, and the empty model directory. It writes only
`outputs/chronological_split_review.json` and one timeline figure.

The decisions deferred to the user are whether to adopt Option B as the actual
split, whether more independent days can be acquired first, which predeclared
seeds/scenario copies will form the final test, and whether the 17:00 transition
should receive a separate operational investigation. No modelling should start
until the boundary choice is explicit.
