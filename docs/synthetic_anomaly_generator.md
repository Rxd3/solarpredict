# Day 13 synthetic anomaly generator

## Scope and entry point

The generator creates one reproducible validation prototype from the existing
`data/processed/operational_features_change.csv` (1,009 rows x 81 columns).
Output is `data/processed/operational_synthetic_prototype.csv` (1,009 x 84).
It adds exactly three synthetic-label columns and preserves all source files.

Synthetic anomalies are controlled prototype test cases and should not be
interpreted as verified real photovoltaic fault signatures.

No model training, scaling, threshold selection, model evaluation, computer
vision, video processing, or dashboard is implemented in this stage. In
particular, this full-baseline prototype is **not a final held-out model test**.

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.anomaly_detection.generate_synthetic_anomalies --prototype --scenarios all --seed 42
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

`--config`, `--input`, `--output`, `--metadata`, and `--figure` select file paths.
Use module execution (`python -m ...`) so imports resolve from the repository
root. Repeating the command replaces only the prototype artifacts at those
destinations. Raw/cleaned/feature source CSV paths cannot be destinations.

## Configuration and explicit prototype overrides

The generator loads `config/synthetic_anomaly_scenarios.yaml` with safe YAML
parsing and validates the seven supported scenarios, variable names, fractional
severity ranges, sample/minute duration agreement, and required placement rules.
Transformation strings are documentation, never executable expressions.

The Day 12 template retains its disabled top-level and scenario switches. With
`--prototype` alone, only scenarios enabled by both switches are selected (none
with the current file). `--scenarios all` or a list of exact scenario names is an
explicit per-run enable override recorded in metadata. The fixed prototype uses
all seven names and seed 42; the template is not rewritten to enable them.

Day 13 specifically requests one combined prototype before a final split exists.
Consequently, explicit `--prototype` permits two recorded departures from the
future Day 12 evaluation plan: using the full baseline as the scheduling pool,
and combining scenario families in one copy. The final chronological split,
training, calibration, and evaluation rules remain unimplemented. The API rejects
generation without explicit prototype scope rather than implying that a valid
final evaluation partition has been created.

## Architecture and copy preservation

`load_config` validates the template. The existing `load_feature_dataset` reads
the 81-column schema and parses source timestamps without assigning a timezone.
`generate_prototype` works on deep copies and returns a DataFrame plus metadata;
it performs no file writes and does not mutate the caller's configuration.
`generate_file` provides protected file loading, saving, hash checks, and the
single requested validation figure.

The input must have a positional index, finite original measurements, and exact
120-second cadence. The generator rejects irregular input because the existing
rate features assume two-minute intervals; it does not sort, fill, or repair it.
The placement helper independently checks candidate sampling continuity too.

The CSV writer copies unchanged source field strings verbatim, including all
timestamps and unaffected original measurements, and serializes changed cells
and labels only. A read-back validation verifies the saved dataset. Input,
configuration, and all prior operational CSVs are protected by before/after
SHA-256 checks. Output destinations must be distinct, have appropriate extensions,
and cannot overwrite these inputs or write artifacts under `models/`.

## Placement and transformation rules

- Power/daylight scenarios require `Solar_Radiation > 100 W/m^2`,
  `Power_Generated >= 50 W`, and `low_light_context == 0` on every original
  baseline event row. Eligibility never uses already injected values.
- All event rows require finite original measurements and contiguous sampling.
  Sensor excursions may use either lighting context. Exactly one temperature or
  humidity channel is modified; reject a humidity candidate outside 0–100% RH.
- No events overlap. At least 60 minutes separate an event's exclusive end from
  another event's start. At most floor(1009 * 0.05) = 50 original-measurement rows
  may be modified in this prototype.
- The inclusive windows 2022-04-27 16:40–17:20 and 2022-04-28 16:40–17:20 are
  protected. Reject placements whose direct window **or subsequent 60-minute
  feature-influence guard** intersects either window. This preserves all 81
  original columns in both windows, including their rolling/change features.
- Process long minimum-duration scenarios first, breaking ties using the YAML
  scenario order. Reserve the remaining scenarios' minimum row budgets where
  feasible. Enumerate feasible durations and chronologically ordered eligible
  starts; sample uniformly among feasible durations, then among their starts.
  This conditions the Day 12 proposed sampling on actual placement feasibility.
  It is not a representative distribution of real faults.
- Draw a continuous severity once per event (and a sensor/sign where applicable).
  Record failed placements as skipped with a budget/context reason. Never force
  an event, silently relax constraints, or treat a numerical no-op as modified.
  A skipped event does not imply the detector missed anything.

For power changes, scale `Power_Generated` and `Array_Voltage` by the same factor;
preserve `Array_Current`. The existing approximate P = V*I relationship is retained,
including its scaled small consistency residual. A gradual event uses
factor(j) = 1 - severity*j/n for j=1..n, so all event rows are modified. The
near-zero scenario also accepts an explicit fixed retained fraction [0, 0].

Radiation/power inconsistency scales only `Solar_Radiation`; power, voltage,
and current retain their baseline behavior. Sensor excursions add a signed
offset to one environmental sensor. These transformations reuse the Day 12
severity/duration ranges and are not circuit or physical fault simulations.

## Dependent-feature recomputation and labels

The synthetic measurement copy passes through existing reusable functions:

1. `add_basic_features`: electrical product/residual and low-light context.
2. `add_rolling_features`: complete trailing means/std/min/max.
3. `add_change_features`: differences, per-minute rates, safe relative changes,
   and rolling-baseline deviations.

A dependency mask derived from the existing feature maps and window specifications
identifies each column's affected rows, including causal trailing history and
the recovery difference. Unaffected cells are restored from the baseline between
stages. This avoids propagating tiny floating-point recalculation drift into
otherwise untouched rows; no feature formulas are duplicated in the generator.
The tests independently verify selected differences, rates, rolling means,
electrical products, and safeguarded ratios against their mathematical definitions.

| Label | Untouched original measurements | Directly modified original measurements |
|---|---|---|
| `synthetic_anomaly` | 0 | 1 |
| `synthetic_anomaly_type` | `none` | Exact scenario name |
| `synthetic_anomaly_id` | null (empty CSV field) | Unique deterministic event ID |

IDs are unique within the prototype and share a deterministic run prefix. Events
are numbered in recorded placement order, not chronological order. Every ID's
rows are contiguous and have the same scenario name. Labels are never added to
the master baseline or used as model features.

After direct measurement changes end, trailing features can still depend on the
event. These rows correctly keep label 0 and are recorded as
`derived_only_tail_rows` per event. Across the seed-42 run, 46 rows have modified
original measurements and another 203 rows are within derived-feature influence
support. Changes to these recorded tails are intentional; no original measurements
are changed there. All cells outside their dependency support remain unchanged.

## Seed-42 before/after event review

All seven scenarios were placed once; none was skipped. The timestamps below are
timezone-naive. Ranges compare the primary affected variable over the same event
rows. Power uses W, radiation W/m^2, and humidity percentage points.

| Scenario | Inclusive timestamps on 2022-04-28 | Rows / support duration | Severity | Variable | Baseline range | Synthetic range |
|---|---|---|---|---|---:|---:|
| sensor_excursion | 02:26–02:28 | 2 / 4 min | +7.113808 RH points | Relative_Humidity | 43.420420–43.560464 | 50.534228–50.674272 |
| temporary_near_zero_power | 08:22–08:26 | 3 / 6 min | Retain 0.256227% | Power_Generated | 299.899540–300.261840 | 0.768424–0.769353 |
| gradual_power_degradation | 10:38–11:26 | 25 / 50 min | Ramp to 21.609341% reduction | Power_Generated | 391.508040–403.880600 | 306.905734–399.937663 |
| sudden_power_drop | 12:38–12:42 | 3 / 6 min | 25.222794% reduction | Power_Generated | 373.854560–374.366960 | 279.557994–279.941153 |
| sustained_power_reduction | 14:00–14:08 | 5 / 10 min | 22.878969% reduction | Power_Generated | 373.688160–375.325520 | 288.192162–289.454911 |
| sudden_power_spike | 15:34–15:34 | 1 / 2 min | 17.415960% increase | Power_Generated | 389.069400 | 456.829573 |
| radiation_power_inconsistency | 17:22–17:34 | 7 / 14 min | 11.412660% increase | Solar_Radiation | 113.150280–139.882440 | 126.063737–155.846748 |

A single-row event still has two minutes of sample support. For n rows, the
inclusive first/last timestamp difference is (n-1)*2 minutes; the exclusive end
is start+n*2 minutes. The radiation event starts strictly after the protected
window ends at 17:20.

The metadata includes original and modified values for every directly affected
variable, source positions, sampled parameters, full baseline context for each
event plus two neighboring rows on each side, and all derived influence rows.
The one validation plot shows the gradual power event, with labeled baseline and
synthetic power, its return to baseline, and a highlighted event interval:

![Synthetic power example](../outputs/figures/operational/day13_synthetic_anomaly_example.png)

## Reproducibility and artifacts

The fixed run uses seed 42 with a local NumPy Generator/PCG64. No global RNG state
or wall-clock event IDs are used. The run prefix depends on baseline values,
resolved configuration, selected scenarios, seed, and generator version.
Metadata records processing order, feasible duration/start counts, all parameters,
input/config/output hashes, generator source and feature-utility hashes, package
versions, disabled/skipped scenarios, and the explicit prototype scope overrides.

Identical baseline, configuration, selected scenarios, seed, code, and environment
reproduce identical CSV bytes and event records. File-path provenance may differ
between workspaces; the same seed alone is not sufficient across changed inputs
or software versions. Byte reproducibility is tested with two temporary outputs.

Created artifacts:

- `data/processed/operational_synthetic_prototype.csv` — ignored by Git.
- `outputs/synthetic_anomaly_metadata.json` — audit and before/after event records.
- `outputs/figures/operational/day13_synthetic_anomaly_example.png` — one ignored,
  reproducible engineering-validation figure.

## Validation and limitations

Both in-memory and saved-CSV validation check rows, column order, timestamps,
non-overlap, IDs/labels, context, duration, electrical scaling, denominator
safeguards, excluded windows, derived-feature consistency, and permitted change
support. The saved CSV also preserves unaffected original fields verbatim.
All raw/cleaned/existing feature CSVs retain their hashes and `models/` remains
unchanged. No scaling, detector, or performance evaluation runs in the generator.

The near-zero event makes three additional power-relative-change values NaN,
raising feature NaN cells from 1,585 to 1,588. This is the intended existing
1 W denominator safeguard; those values are not filled and no infinities appear.
Null event IDs on baseline rows are separate label nulls, not feature missingness.

The synthetic spike reaches 456.829573 W, above the baseline's observed maximum
438.556840 W. Equipment capacity is not established, so this is a controlled
measurement perturbation, not a capacity-validated event. The ramp recovery edge
is also artificial. Neither is evidence about detector quality.

Coverage remains approximately 33.6 hours. No final chronological split has been
chosen, and this prototype must not be reused as an independent final model test
after training on the same timestamps. Later work must define training/calibration/
evaluation boundaries, use independent evaluation copies, and then train and
evaluate a detector in a separately authorized stage.
