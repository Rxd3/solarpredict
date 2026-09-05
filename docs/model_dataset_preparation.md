# Day 15 model dataset preparation

## Adopted chronological split

Day 15 formally adopts the previously reviewed Option B split. The input is the
untouched `data/processed/operational_features_change.csv` dataset: 1,009 rows
and 81 columns. It is not the Day 13 synthetic prototype. Timestamps remain
timezone-naive because the dataset source does not specify a timezone.

The boundaries are inclusive observed timestamps. Rows were not shuffled.

| Partition | Inclusive timestamps | Rows | Full baseline shape | Core shape |
|---|---|---:|---:|---:|
| Training | 2022-04-27 15:32–2022-04-28 02:42 | 336 | 336 × 81 | 336 × 10 |
| Calibration | 2022-04-28 02:44–2022-04-28 13:54 | 336 | 336 × 81 | 336 × 10 |
| Evaluation baseline | 2022-04-28 13:56–2022-04-29 01:08 | 337 | 337 × 81 | 337 × 10 |

The Core shape includes `Timestamp` plus these nine approved, unscaled
features:

- `Power_Generated`
- `Solar_Radiation`
- `Air_Temp`
- `Relative_Humidity`
- `Wind_Speed`
- `rtd_mean`
- `rtd_std`
- `minute_of_day_sin`
- `minute_of_day_cos`

The partition files contain no synthetic labels. No scaler, model, threshold,
or final synthetic evaluation data was created.

## Coverage and preservation validation

The partition row sum is 336 + 336 + 337 = 1,009. Every source position and
timestamp occurs in exactly one partition. There are no overlaps, uncovered
rows, or duplicate timestamps; all partitions remain chronological. The full
baseline files preserve all 81 columns, and concatenating them in partition
order reconstructs the source within CSV floating-point serialization
precision. The source file's SHA-256 hash was unchanged before and after the
operation.

All nine Core features have zero missing values, zero infinite values, a finite
value on every row, and positive population variance in every partition. Exact
minimum, maximum, mean, median, population standard deviation, and variance
values are available in `outputs/model_dataset_preparation.json`.

## Daylight and low-light context

The existing exploratory context rule defines low light as
`Solar_Radiation <= 5 W/m²`. It is not an anomaly label or an equipment
specification. Recalculation from irradiance exactly matches the stored
`low_light_context` field.

| Partition | Daylight | Low light | Total |
|---|---:|---:|---:|
| Training | 100 | 236 | 336 |
| Calibration | 242 | 94 | 336 |
| Evaluation baseline | 149 | 188 | 337 |

## Descriptive distribution comparison

The report compares each later partition mean with the training mean in units
of the training population standard deviation. This is a descriptive indicator
only. It is not a hypothesis test and does not establish population-level
distribution differences.

The most visible differences are:

- Calibration occurs primarily from pre-dawn through midday. Its time-of-day
  sine and cosine means differ from training by +1.93 and -1.73 training
  standard deviations respectively. This is expected from the selected time
  boundaries.
- Mean `Solar_Radiation` is 83.66 W/m² in training, 412.74 W/m² in calibration,
  and 175.90 W/m² in evaluation. Calibration differs from training by +1.84
  training standard deviations.
- Mean `Power_Generated` is 122.19 W in training, 259.80 W in calibration, and
  167.51 W in evaluation. Calibration differs from training by +0.87 training
  standard deviations.
- Calibration `rtd_std` averages 1.632 versus 0.698 in training, a +1.60
  training-standardized difference. Mean `rtd_mean` is 54.00 in calibration
  versus 36.32 in training (+0.74).
- Mean `Air_Temp` is 21.96 in training, 27.93 in calibration, and 26.17 in
  evaluation. The calibration and evaluation standardized differences are
  +0.71 and +0.50.
- Evaluation mean `rtd_mean` and `Solar_Radiation` differ from training by
  +0.56 and +0.52 training standard deviations. Relative humidity and wind
  speed do not cross the report's absolute 0.5 descriptive threshold.

These differences reinforce the limitation of splitting a short continuous
record: each region represents a different portion of the operating cycle.
They must be considered when later interpreting calibration scores and false
positives, without treating them as anomaly evidence.

## Recurring 17:00 transitions

The 2022-04-27 17:00 observation is in training. The 2022-04-28 17:00
observation is in the evaluation baseline. Both are retained and neither is
labelled as anomalous. They are known recurring operational transitions that
should be interpreted carefully during later false-positive analysis.

## Future model preprocessing order

1. Load the Core training matrix.
2. Fit a scaler **only** on the nine training features.
3. Transform training using that fitted scaler.
4. Apply the already-fitted training scaler to calibration.
5. Train the unsupervised detector using training data only.
6. Use untouched calibration data to study normal anomaly-score behavior and
   freeze a threshold.
7. Generate a new synthetic copy only within the evaluation period.
8. Recompute dependent features for that evaluation copy.
9. Extract the same nine Core features.
10. Apply the training-fitted scaler.
11. Evaluate using the frozen detector and frozen threshold.

Calibration and evaluation information must never be used to fit the scaler.
Synthetic evaluation labels must never be used to train the detector or choose
the threshold.

## Prototype limitation

The complete source covers approximately 33.6 hours and contains only one
complete calendar day. These adjacent partitions are suitable only for the
current internship prototype; they are not independent daily samples and do
not establish generalization. The split should be revisited when more
independent operational data becomes available.

## Reproduction

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.anomaly_detection.create_chronological_split
.\.venv\Scripts\python.exe -m pytest -q
```

