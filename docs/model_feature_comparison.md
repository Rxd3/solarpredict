# Day 18 controlled model-feature comparison

## Scope and predeclared design

This stage compares exactly four feature sets. It is a prototype model
comparison, not final anomaly evaluation. Every new variant uses the same 336
chronological training rows, a training-only `StandardScaler`, and the unchanged
Day 16 Isolation Forest parameters. `Timestamp` is excluded. No synthetic data,
anomaly label, threshold, or hyperparameter search is used.

| Variant | Features | Artifact treatment |
|---|---|---|
| `CORE_9` | Power, radiation, temperature, humidity, wind, RTD mean, RTD std, time sine, time cosine | Frozen Day 16 scaler/model/scores reused |
| `NO_TIME_7` | Power, radiation, temperature, humidity, wind, RTD mean, RTD std | New training-only comparison artifacts |
| `NO_RTD_STD_8` | Power, radiation, temperature, humidity, wind, RTD mean, time sine, time cosine | New training-only comparison artifacts |
| `COMPACT_6` | Power, radiation, temperature, humidity, wind, RTD mean | New training-only comparison artifacts |

All new models use `n_estimators=300`, `max_samples="auto"`,
`max_features=1.0`, `bootstrap=False`, `contamination="auto"`,
`random_state=42`, and `n_jobs=-1`. Continuous score orientation remains
higher = more isolated.

## Predeclared selection criteria

The criteria and weights were written to
`config/model_feature_comparison.yaml` before variant results were generated.
An eligible variant must retain non-trivial training score spread: population
standard deviation at least 0.01 and P99 minus P90 at least 0.01.

Eligible variants are ranked by this fixed weighted objective:

- 45% absolute calibration-versus-training mean shift measured in training
  score standard deviations;
- 35% mean absolute daylight/low-light train-to-calibration shift in training
  score standard deviations;
- 15% fraction of calibration rows where an included coverage-limited or
  unconfirmed feature exceeds |z|=5;
- 5% feature-count penalty relative to the six-feature minimum.

The rule retains six meaningful operational measurements, discourages a
trivial detector with collapsed training scores, and does not simply select the
lowest absolute calibration score. Ties favor fewer features and then
declaration order. Evaluation-baseline results are explicitly unavailable to
the selection function.

## Training and calibration scores

| Variant/partition | Mean | Median | Population std | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|
| CORE_9 training | 0.469685 | 0.448086 | 0.055403 | 0.560395 | 0.582952 | 0.643569 |
| CORE_9 calibration | 0.625319 | 0.645260 | 0.058162 | 0.681459 | 0.695893 | 0.701019 |
| NO_TIME_7 training | 0.453758 | 0.417276 | 0.072521 | 0.573525 | 0.596483 | 0.648829 |
| NO_TIME_7 calibration | 0.601381 | 0.630715 | 0.089997 | 0.689586 | 0.710033 | 0.717010 |
| NO_RTD_STD_8 training | 0.480915 | 0.459031 | 0.051663 | 0.560145 | 0.584016 | 0.635127 |
| NO_RTD_STD_8 calibration | 0.635459 | 0.648086 | 0.047195 | 0.683359 | 0.685102 | 0.694278 |
| COMPACT_6 training | 0.469097 | 0.437618 | 0.062715 | 0.570523 | 0.593358 | 0.633886 |
| COMPACT_6 calibration | 0.603883 | 0.606488 | 0.072676 | 0.682158 | 0.685673 | 0.694172 |

### Calibration shift and predeclared objective

| Variant | Calibration minus training mean | Shift / training score std | Mean context shift / training std | Limited-feature extreme fraction | Objective |
|---|---:|---:|---:|---:|---:|
| `CORE_9` | 0.155635 | 2.8091 | 1.9962 | 0.0476 | 2.0199 |
| `NO_TIME_7` | 0.147623 | 2.0356 | 1.1150 | 0.0476 | **1.3301** |
| `NO_RTD_STD_8` | 0.154544 | 2.9914 | 2.2397 | 0.0000 | 2.1634 |
| `COMPACT_6` | 0.134787 | 2.1492 | 1.2902 | 0.0000 | 1.4187 |

All variants retain non-trivial training spread. Removing time features produces
the largest improvement in standardized overall and context-specific baseline
stability. Removing only `rtd_std` does not address the clock-time support gap.
`COMPACT_6` has the smallest raw mean difference, but its standardized shift and
context component are worse than `NO_TIME_7`; this illustrates why absolute
calibration score alone is not the selection rule.

## Calibration daylight and low-light behavior

| Variant | Training daylight mean | Calibration daylight mean | Training low-light mean | Calibration low-light mean | Daylight / low-light context shift (training std units) |
|---|---:|---:|---:|---:|---:|
| `CORE_9` | 0.538617 | 0.657304 | 0.440476 | 0.542976 | 2.142 / 1.850 |
| `NO_TIME_7` | 0.551945 | 0.650249 | 0.412154 | 0.475573 | **1.356 / 0.874** |
| `NO_RTD_STD_8` | 0.545211 | 0.661256 | 0.453670 | 0.569044 | 2.246 / 2.233 |
| `COMPACT_6` | 0.550763 | 0.642425 | 0.434492 | 0.504659 | 1.462 / 1.119 |

Removing the clock features reduces both context shifts materially. Removing
`rtd_std` alone does not. Daylight calibration scores remain elevated for every
variant, so the short training coverage is not fully resolved by feature
ablation.

## High-score calibration baseline observations

These are high-score baseline observations, not anomalies or faults. All 40
top-ten entries are daylight-like.

- `CORE_9`: ten timestamps span 10:36–13:42; eight fit within one 60-minute
  window. Most cluster around high `rtd_std` near 10:36–10:56, with 11:58 and
  13:42 radiation/wind cases.
- `NO_TIME_7`: ten timestamps span 10:28–13:42; eight fit within one hour. Most
  remain in the high-radiation/high-RTD-spread 10:28–10:46 period.
- `NO_RTD_STD_8`: timestamps span 11:48–13:48; six fit within one hour. Removing
  RTD spread moves the list toward high-radiation noon/early-afternoon rows.
- `COMPACT_6`: timestamps span 10:52–13:48; four fit within one hour. It mixes
  high-radiation noon/afternoon rows with three 10:52–10:56 high-RTD-period
  rows, even though RTD spread is not itself an input.

Only 11:58 and 13:42 appear in all four top-ten lists. Pairwise overlap ranges
from 2/10 to 6/10; the largest overlap is between `NO_RTD_STD_8` and
`COMPACT_6`. The machine-readable report records every timestamp, score,
light context, power, radiation, temperature, humidity, wind, and RTD context.

## Recurring training transition

The 2022-04-27 17:00 observation is retained and not classified as a fault.

| Variant | Score | Highest-score rank in 336 training rows | Percentile |
|---|---:|---:|---:|
| `CORE_9` | 0.586922 | 13 | 96.43 |
| `NO_TIME_7` | 0.604274 | 13 | 96.43 |
| `NO_RTD_STD_8` | 0.597504 | 10 | 97.32 |
| `COMPACT_6` | 0.616746 | 8 | 97.92 |

Removing features does not suppress this transition's relative training rank;
it becomes slightly more prominent in the compact variants.

## Training/calibration recommendation

The predeclared rule recommends **`NO_TIME_7`** with objective 1.3301. It retains
the six environmental/electrical/RTD-mean measurements plus `rtd_std`, reduces
overall standardized baseline shift from 2.81 to 2.04, and reduces mean context
shift from 2.00 to 1.12. It is compact enough to interpret while preserving RTD
spread for a later controlled comparison.

This is a recommendation for the next prototype configuration, not a final
model freeze or general validation. `rtd_std` remains unconfirmed and still
produces 16 calibration rows beyond |z|=5, so its inclusion should be revisited
when more representative training data or RTD metadata becomes available.

The selection record was hashed before evaluation-baseline scoring. The
evaluation period was not consulted by the selection function.

![Training/calibration score comparison](../outputs/figures/operational/day18_model_feature_comparison.png)

## Optional evaluation-baseline documentation

These results were computed only after the `NO_TIME_7` recommendation was
locked. They do not change or support the selection.

| Variant | Mean | Median | Population std | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|
| `CORE_9` | 0.522305 | 0.495549 | 0.073415 | 0.651213 | 0.664643 | 0.671642 |
| `NO_TIME_7` | 0.509457 | 0.479312 | 0.090451 | 0.652622 | 0.668993 | 0.674571 |
| `NO_RTD_STD_8` | 0.532201 | 0.504879 | 0.072060 | 0.657482 | 0.672792 | 0.677045 |
| `COMPACT_6` | 0.521191 | 0.506748 | 0.085451 | 0.654266 | 0.667146 | 0.672819 |

## Limitations and next boundary

The source contains approximately 33.6 hours and one complete calendar day.
This comparison selects a prototype feature configuration; it does not validate
generalization. No threshold is selected, no synthetic evaluation data is
generated, and no performance metric is calculated. Final model configuration
freeze remains a later explicit decision.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.anomaly_detection.compare_detector_features
.\.venv\Scripts\python.exe -m pytest -q
```

