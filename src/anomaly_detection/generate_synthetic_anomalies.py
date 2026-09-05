"""Generate controlled synthetic prototype copies; never fit a detector.

Run from the project root with ``python -m src.anomaly_detection.
generate_synthetic_anomalies --prototype --scenarios all``. The explicit
prototype switch permits the unsplit baseline and mixed scenario families for
generator validation only. Disabled YAML switches otherwise remain effective.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data_processing.engineer_basic_features import (
    ORIGINAL_COLUMNS, ENGINEERED_FEATURES, add_basic_features,
)
from src.data_processing.engineer_rolling_features import (
    ROLLING_FEATURE_SPECS, add_rolling_features,
)
from src.data_processing.engineer_change_features import (
    CHANGE_FEATURES, DIFFERENCE_FEATURES, RATE_FEATURES,
    RELATIVE_CHANGE_FEATURES, DEVIATION_FEATURES, add_change_features,
)
from src.data_processing.review_model_features import (
    FEATURE_COLUMNS, load_feature_dataset, CORE_UNSUPERVISED_FEATURES,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/synthetic_anomaly_scenarios.yaml"
OUTPUT = ROOT / "data/processed/operational_synthetic_prototype.csv"
METADATA = ROOT / "outputs/synthetic_anomaly_metadata.json"
FIGURE = ROOT / "outputs/figures/operational/day13_synthetic_anomaly_example.png"
VERSION = "1.0.0"
LABELS = ("synthetic_anomaly", "synthetic_anomaly_type", "synthetic_anomaly_id")
SCENARIOS = (
    "sudden_power_drop", "sustained_power_reduction", "temporary_near_zero_power",
    "sudden_power_spike", "gradual_power_degradation",
    "radiation_power_inconsistency", "sensor_excursion",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_config(config: dict) -> None:
    """Reject unsupported or inconsistent executable settings, not just YAML syntax."""
    if config.get("schema_version") != 1 or config.get("sampling_interval_seconds") != 120:
        raise ValueError("Only schema 1 and 120-second sampling are supported.")
    if type(config.get("enabled")) is not bool:
        raise ValueError("enabled must be a YAML boolean.")
    if tuple(config.get("core_features", ())) != CORE_UNSUPERVISED_FEATURES:
        raise ValueError("Core features must match the Day 11 definition.")
    if tuple(config.get("scenarios", ())) != SCENARIOS:
        raise ValueError("Configuration must define the seven supported scenarios in order.")
    context = config["context"]
    placement = context.get("placement_window")
    if placement is not None:
        start = pd.Timestamp(placement.get("start"))
        end = pd.Timestamp(placement.get("end"))
        if pd.isna(start) or pd.isna(end) or start.tzinfo is not None or end.tzinfo is not None:
            raise ValueError("Placement boundaries must be valid timezone-naive timestamps.")
        if start > end:
            raise ValueError("Placement start must not follow placement end.")
    for key in ("power_event_min_radiation_w_m2", "power_event_min_power_w"):
        if not np.isfinite(context[key]) or context[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if context["known_transition_exclusions"]["half_window_minutes"] < 20:
        raise ValueError("Protect at least 20 minutes on each side of known transitions.")
    if (context["overlap_allowed"] is not False or
            context["require_entire_window_eligible"] is not True or
            context["require_finite_inputs"] is not True or
            context["require_contiguous_sampling"] is not True):
        raise ValueError("Finite, continuous, whole non-overlapping windows are required.")
    if (context["minimum_gap_minutes"] < 60 or
            not 0 < context["max_modified_row_fraction_per_copy"] <= 0.05):
        raise ValueError("Require at least 60 minutes between events and at most 5% modified rows.")
    if context["start_position"]["manual_override"] is not None:
        raise ValueError("Manual start override is not supported in prototype v1.")
    for key, value in {
        "eligibility_reference": "untouched_baseline",
        "power_consistency_policy": "scale_power_and_voltage_by_same_factor_keep_current",
        "derived_feature_policy": "recompute_causally_in_evaluation_copy",
    }.items():
        if context[key] != value:
            raise ValueError(f"Unsupported context setting: {key}")
    for name, spec in config["scenarios"].items():
        if type(spec["enabled"]) is not bool:
            raise ValueError(f"{name}: enabled must be boolean.")
        low, high = spec["duration_samples"]
        if not (type(low) is int and type(high) is int and 1 <= low <= high):
            raise ValueError(f"{name}: invalid duration range.")
        if spec["duration_minutes"] != [low * 2, high * 2]:
            raise ValueError(f"{name}: minutes and sample duration disagree.")
        expected_variables = (
            ["Air_Temp", "Relative_Humidity"] if name == "sensor_excursion" else
            ["Solar_Radiation"] if name == "radiation_power_inconsistency" else
            ["Power_Generated", "Array_Voltage"]
        )
        if spec["affected_variables"] != expected_variables:
            raise ValueError(f"{name}: unsupported affected variables.")
        expected_rule = "finite_sensor_any_light" if name == "sensor_excursion" else "power_daylight"
        if spec["context_rule"] != expected_rule:
            raise ValueError(f"{name}: unsupported context rule.")
        if name == "sensor_excursion":
            if spec["severity"]["direction_choices"] != [-1, 1]:
                raise ValueError("Sensor direction choices must be -1 and 1.")
            ranges = [x["range"] for x in spec["severity"]["magnitude_ranges"].values()]
        else:
            ranges = [spec["severity"]["range"]]
        for a, b in ranges:
            allow_exact_zero = name == "temporary_near_zero_power"
            if not (np.isfinite([a, b]).all() and 0 <= a <= b and (b > 0 or allow_exact_zero)):
                raise ValueError(f"{name}: invalid severity range.")
            if name != "sensor_excursion" and b >= 1:
                raise ValueError(f"{name}: fraction must remain below 1.")


def load_config(path: Path = CONFIG) -> dict:
    with Path(path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("Scenario YAML must be a mapping.")
    validate_config(config)
    return config


def excluded_rows(frame: pd.DataFrame, config: dict) -> np.ndarray:
    rule = config["context"]["known_transition_exclusions"]
    mask = np.zeros(len(frame), dtype=bool)
    delta = pd.Timedelta(minutes=rule["half_window_minutes"])
    for value in rule["timestamps"]:
        time = pd.Timestamp(value)
        mask |= frame.Timestamp.between(time - delta, time + delta).to_numpy()
    return mask


def eligible_starts(frame: pd.DataFrame, config: dict, scenario: str, n: int,
                    occupied: list[tuple[int, int]], variable: str | None = None,
                    offset: float = 0.0) -> list[int]:
    """Enumerate starts against the untouched baseline, including a 60-minute tail guard."""
    context = config["context"]
    finite = np.isfinite(frame[list(ORIGINAL_COLUMNS[1:])].to_numpy()).all(axis=1)
    placement = context.get("placement_window")
    if placement is not None:
        finite &= frame.Timestamp.between(
            pd.Timestamp(placement["start"]), pd.Timestamp(placement["end"])
        ).to_numpy()
    if scenario != "sensor_excursion":
        finite &= ((frame.Solar_Radiation > context["power_event_min_radiation_w_m2"])
                   & (frame.Power_Generated >= context["power_event_min_power_w"])
                   & (frame.low_light_context == 0)).to_numpy()
    elif variable == "Relative_Humidity":
        finite &= (frame.Relative_Humidity + offset).between(0, 100).to_numpy()
    forbidden = excluded_rows(frame, config)
    gap = int(np.ceil(context["minimum_gap_minutes"] / 2))
    valid = []
    for start in range(len(frame) - n + 1):
        end = start + n  # exclusive
        tail_end = min(len(frame), end + 30)
        if not finite[start:end].all() or forbidden[start:tail_end].any():
            continue
        if any(not (end + gap <= a or b + gap <= start) for a, b in occupied):
            continue
        # Include the previous row for differences and the tail for rolling history.
        times = frame.Timestamp.iloc[max(0, start - 1):tail_end]
        if not times.diff().dropna().eq(pd.Timedelta(seconds=120)).all():
            continue
        valid.append(start)
    return valid


def _spread(mask: pd.Series, samples: int) -> pd.Series:
    return mask.rolling(samples, min_periods=1).max().astype(bool)


def dependency_masks(primary_masks: pd.DataFrame) -> dict[str, pd.Series]:
    """Track dependency support, using existing window specifications and feature maps."""
    masks = {c: primary_masks[c] for c in ORIGINAL_COLUMNS[1:]}
    false = pd.Series(False, index=primary_masks.index)
    for column in ENGINEERED_FEATURES:
        masks[column] = false.copy()
    electrical = masks["Power_Generated"] | masks["Array_Voltage"] | masks["Array_Current"]
    for column in ("electrical_power_product", "power_consistency_residual", "power_consistency_abs_error"):
        masks[column] = electrical
    masks["low_light_context"] = masks["Solar_Radiation"]
    for spec in ROLLING_FEATURE_SPECS:
        masks[spec.name] = _spread(masks[spec.source_column], spec.window_samples)
    for source, name in DIFFERENCE_FEATURES.items():
        masks[name] = _spread(masks[source], 2)
    for source, name in RATE_FEATURES.items():
        masks[name] = masks[DIFFERENCE_FEATURES[source]]
    for source, name in RELATIVE_CHANGE_FEATURES.items():
        masks[name] = _spread(masks[source], 2)
    for source, (baseline, deviation, ratio, _) in DEVIATION_FEATURES.items():
        masks[deviation] = masks[source] | masks[baseline]
        masks[ratio] = masks[deviation]
    return masks


def recompute_copy(baseline: pd.DataFrame, measurements: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reuse each engineering stage; retain original cells outside exact dependency support.

    Restoring untouched cells between stages avoids propagating harmless float
    round-trip/rolling accumulator drift into otherwise unaffected observations.
    """
    changed = measurements[list(ORIGINAL_COLUMNS[1:])].ne(baseline[list(ORIGINAL_COLUMNS[1:])])
    masks = dependency_masks(changed)
    result = measurements.copy(deep=True)
    stages = (
        (add_basic_features, ENGINEERED_FEATURES),
        (add_rolling_features, tuple(s.name for s in ROLLING_FEATURE_SPECS)),
        (add_change_features, CHANGE_FEATURES),
    )
    for function, names in stages:
        computed = function(result)
        for column in names:
            result[column] = baseline[column].where(~masks[column], computed[column])
    return result, masks


def _records(frame: pd.DataFrame, rows: list[int], columns: list[str]) -> list[dict]:
    return [{"row_position": i, "timestamp": str(frame.Timestamp.iloc[i]),
             **{c: float(frame[c].iloc[i]) for c in columns}} for i in rows]


def generate_prototype(baseline: pd.DataFrame, config: dict, *, seed: int | None = None,
                       prototype: bool = False, scenarios: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Pure copy-based API. Explicit selection enables scenarios for this prototype run only."""
    validate_config(config)
    if not prototype:
        raise ValueError("Use explicit prototype=True; final held-out generation is not implemented.")
    if list(baseline.columns) != list(FEATURE_COLUMNS) or baseline.empty:
        raise ValueError("Expected a non-empty Day 10 81-column baseline.")
    if not baseline.index.equals(pd.RangeIndex(len(baseline))):
        raise ValueError("Expected positional RangeIndex without reordered rows.")
    if not pd.api.types.is_datetime64_any_dtype(baseline.Timestamp):
        raise ValueError("Parse baseline Timestamp with the existing loader first.")
    if baseline.Timestamp.dt.tz is not None or baseline.Timestamp.isna().any():
        raise ValueError("Expected valid timezone-naive timestamps.")
    if not baseline.Timestamp.diff().dropna().eq(pd.Timedelta(seconds=120)).all():
        raise ValueError("Baseline must have exact 120-second cadence for existing feature utilities.")
    if not np.isfinite(baseline[list(ORIGINAL_COLUMNS[1:])].to_numpy()).all():
        raise ValueError("Baseline original measurements must be finite; no imputation is performed.")
    seed = config["random_seed"] if seed is None else seed
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a non-negative integer.")
    selected = (list(scenarios) if scenarios is not None else
                [n for n, s in config["scenarios"].items() if config["enabled"] and s["enabled"]])
    if len(set(selected)) != len(selected) or not set(selected).issubset(SCENARIOS):
        raise ValueError("Unknown or duplicate scenario names.")
    # Place long windows first; reserve minimum durations for remaining scenarios.
    order = sorted(selected, key=lambda n: (-config["scenarios"][n]["duration_samples"][0], SCENARIOS.index(n)))
    rng = np.random.Generator(np.random.PCG64(seed))
    baseline_identity = hashlib.sha256(pd.util.hash_pandas_object(baseline, index=True).values.tobytes()).hexdigest()
    run_id = canonical_hash({"baseline": baseline_identity, "config": config, "seed": seed,
                             "selected": selected, "version": VERSION})[:12]
    measurements = baseline[list(ORIGINAL_COLUMNS)].copy(deep=True)
    budget = int(len(baseline) * config["context"]["max_modified_row_fraction_per_copy"])
    occupied: list[tuple[int, int]] = []
    events, skipped = [], []
    for position, name in enumerate(order):
        spec = config["scenarios"][name]
        variable, offset = None, 0.0
        if name == "sensor_excursion":
            variable = str(rng.choice(spec["affected_variables"]))
            severity = float(rng.uniform(*spec["severity"]["magnitude_ranges"][variable]["range"]))
            offset = int(rng.choice(spec["severity"]["direction_choices"])) * severity
            variables = [variable]
        else:
            severity = float(rng.uniform(*spec["severity"]["range"]))
            variables = list(spec["affected_variables"])
        low, high = spec["duration_samples"]
        remaining = budget - sum(b - a for a, b in occupied)
        reserved = sum(config["scenarios"][n]["duration_samples"][0] for n in order[position + 1:])
        limit = min(high, remaining - reserved if remaining >= low + reserved else remaining)
        candidates = {n: eligible_starts(baseline, config, name, n, occupied, variable, offset)
                      for n in range(low, limit + 1)}
        candidates = {n: starts for n, starts in candidates.items() if starts}
        if not candidates:
            skipped.append({"scenario": name, "reason": "row_budget" if limit < low else
                            "no_window_meets_context_gap_exclusion_and_tail_rules",
                            "remaining_row_budget": remaining, "sampled_severity": severity,
                            "selected_sensor": variable})
            continue
        n = int(rng.choice(sorted(candidates)))
        start = int(rng.choice(candidates[n]))
        rows = list(range(start, start + n))
        if name == "sensor_excursion":
            measurements.loc[rows, variable] = baseline.loc[rows, variable] + offset
        elif name == "radiation_power_inconsistency":
            measurements.loc[rows, "Solar_Radiation"] = baseline.loc[rows, "Solar_Radiation"] * (1 + severity)
        else:
            factor = (severity if name == "temporary_near_zero_power" else
                      1 + severity if name == "sudden_power_spike" else
                      1 - severity * np.arange(1, n + 1) / n if name == "gradual_power_degradation" else
                      1 - severity)
            for column in variables:
                measurements.loc[rows, column] = baseline.loc[rows, column] * factor
        changed = measurements.loc[rows, variables].ne(baseline.loc[rows, variables]).any(axis=1)
        if not changed.all():
            measurements.loc[rows, variables] = baseline.loc[rows, variables]
            skipped.append({"scenario": name, "reason": "numerical_no_op"})
            continue
        occupied.append((start, start + n))
        events.append({"anomaly_id": f"{run_id}-{len(events)+1:02d}", "anomaly_type": name,
                       "random_seed": seed, "start_timestamp": str(baseline.Timestamp.iloc[start]),
                       "end_timestamp": str(baseline.Timestamp.iloc[start+n-1]),
                       "end_exclusive": str(baseline.Timestamp.iloc[start+n-1] + pd.Timedelta(minutes=2)),
                       "row_positions": rows, "affected_row_count": n, "duration_minutes": n*2,
                       "severity": {"parameter": spec["severity"]["parameter"], "value": severity,
                                    "signed_offset": offset if variable else None},
                       "affected_variables": variables,
                       "original_values": _records(baseline, rows, variables),
                       "modified_values": _records(measurements, rows, variables),
                       "baseline_context": _records(baseline, list(range(max(0,start-2), min(len(baseline),start+n+2))),
                                                     ["Power_Generated", "Solar_Radiation", "Array_Current", "low_light_context"]),
                       "eligible_start_count_for_duration": len(candidates[n]),
                       "feasible_duration_samples": sorted(candidates)})
    result, masks = recompute_copy(baseline, measurements)
    result[LABELS[0]] = np.zeros(len(result), dtype=np.int8)
    result[LABELS[1]] = "none"
    result[LABELS[2]] = pd.Series(None, index=result.index, dtype=object)
    for event in events:
        rows = event["row_positions"]
        result.loc[rows, list(LABELS)] = [1, event["anomaly_type"], event["anomaly_id"]]
        event_primary = pd.DataFrame(False, index=baseline.index, columns=ORIGINAL_COLUMNS[1:])
        event_primary.loc[rows, event["affected_variables"]] = True
        event_masks = dependency_masks(event_primary)
        derived_mask = pd.DataFrame({c: event_masks[c] for c in FEATURE_COLUMNS[14:]}).any(axis=1)
        event["derived_feature_influence_rows"] = np.flatnonzero(derived_mask).tolist()
        event["derived_only_tail_rows"] = [i for i in event["derived_feature_influence_rows"] if i not in rows]
    report = {"generator_version": VERSION, "seed": seed, "rng": "PCG64", "run_id": run_id,
              "scope": "unsplit_generator_validation_prototype_not_final_model_test",
              "overrides": {"explicit_scenario_enable_list": selected if scenarios is not None else [],
                            "mixed_families_in_one_copy": True, "use_full_baseline_without_final_split": True},
              "configuration": copy.deepcopy(config), "configuration_content_sha256": canonical_hash(config),
              "baseline_frame_identity": baseline_identity,
              "dependency_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                      "pandas": pd.__version__, "pyyaml": yaml.__version__},
              "scenario_processing_order": order,
              "sampling_policy": "severity then uniform feasible duration then uniform sorted eligible start; reserve later minimum durations",
              "events": events, "skipped_scenarios": skipped,
              "disabled_scenarios": [n for n in SCENARIOS if n not in selected],
              "shape": list(result.shape), "event_count": len(events),
              "directly_modified_rows": int(result.synthetic_anomaly.sum()), "row_budget": budget,
              "derived_only_rows": int((pd.DataFrame({c: masks[c] for c in FEATURE_COLUMNS[14:]}).any(axis=1)
                                        & result.synthetic_anomaly.eq(0)).sum()),
              "interpretation": "Controlled prototype cases, not verified real photovoltaic fault signatures."}
    report["validation"] = validate_prototype(baseline, result, config, events)
    if not all(report["validation"].values()):
        raise RuntimeError(f"Prototype validation failed: {report['validation']}")
    return result, report


def validate_prototype(baseline: pd.DataFrame, result: pd.DataFrame, config: dict, events: list[dict]) -> dict:
    """Validate labels, placement, measurement support, and recomputed feature values."""
    permitted = pd.DataFrame(False, index=baseline.index, columns=ORIGINAL_COLUMNS[1:])
    counts = np.zeros(len(baseline), dtype=int)
    context_ok = durations_ok = labels_ok = electrical_ok = True
    occupied = []
    for event in events:
        rows, name = event["row_positions"], event["anomaly_type"]
        a, b = rows[0], rows[-1] + 1
        n = len(rows)
        spec = config["scenarios"][name]
        permitted.loc[rows, event["affected_variables"]] = True
        counts[rows] += 1
        variable = event["affected_variables"][0] if name == "sensor_excursion" else None
        offset = event["severity"]["signed_offset"] or 0.0
        context_ok &= a in eligible_starts(baseline, config, name, n, occupied, variable, offset)
        occupied.append((a, b))
        durations_ok &= (rows == list(range(a, b)) and n * 2 == event["duration_minutes"]
                         and spec["duration_samples"][0] <= n <= spec["duration_samples"][1])
        labels_ok &= bool(result.loc[rows, LABELS[1]].eq(name).all()
                          and result.loc[rows, LABELS[2]].eq(event["anomaly_id"]).all())
        if "Power_Generated" in event["affected_variables"]:
            p_factor = result.loc[rows, "Power_Generated"] / baseline.loc[rows, "Power_Generated"]
            v_expected = baseline.loc[rows, "Array_Voltage"] * p_factor
            electrical_ok &= bool(np.allclose(result.loc[rows, "Array_Voltage"], v_expected, atol=1e-12))
    original_changes = result[list(ORIGINAL_COLUMNS[1:])].ne(baseline[list(ORIGINAL_COLUMNS[1:])])
    recalculated, masks = recompute_copy(baseline, result[list(ORIGINAL_COLUMNS)])
    formulas_ok = all(np.allclose(result[c], recalculated[c], rtol=1e-12, atol=1e-10, equal_nan=True)
                      for c in FEATURE_COLUMNS[1:])
    untouched = result.synthetic_anomaly.eq(0)
    labels_ok &= bool(result.loc[untouched, LABELS[1]].eq("none").all() and
                      result.loc[untouched, LABELS[2]].isna().all())
    forbidden = excluded_rows(baseline, config)
    outside_support = True
    for c in FEATURE_COLUMNS[14:]:
        outside_support &= result.loc[~masks[c], c].equals(baseline.loc[~masks[c], c])
    return {
        "row_count_preserved": len(result) == len(baseline),
        "columns_preserved": list(result.columns) == [*FEATURE_COLUMNS, *LABELS],
        "timestamps_and_order_preserved": result.Timestamp.equals(baseline.Timestamp),
        "event_ids_unique_and_nonoverlapping": bool(counts.max(initial=0) <= 1 and len({e['anomaly_id'] for e in events}) == len(events)),
        "labels_match_direct_measurement_changes": bool(np.array_equal(result.synthetic_anomaly, original_changes.any(axis=1).astype(int))
                                                       and np.array_equal(counts, result.synthetic_anomaly) and labels_ok),
        "no_unintended_original_measurement_changes": not bool((original_changes & ~permitted).any().any()),
        "duration_and_cadence_valid": bool(durations_ok),
        "context_gap_and_tail_guards_valid": bool(context_ok),
        "excluded_windows_all_81_columns_unchanged": result.loc[forbidden, list(FEATURE_COLUMNS)].equals(baseline.loc[forbidden]),
        "electrical_scaling_consistent": bool(electrical_ok and result.Array_Current.equals(baseline.Array_Current)),
        "no_infinities": not bool(np.isinf(result.select_dtypes(include='number').to_numpy()).any()),
        "dependent_features_recomputed": formulas_ok,
        "cells_outside_dependency_support_unchanged": bool(outside_support),
        "modified_row_budget_respected": int(result.synthetic_anomaly.sum()) <= int(len(baseline)*config['context']['max_modified_row_fraction_per_copy']),
    }


def _write_csv(path: Path, source: Path, baseline: pd.DataFrame, result: pd.DataFrame) -> None:
    """Copy unchanged source fields verbatim; serialize changed cells and labels only."""
    with source.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.reader(handle))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         suffix=".csv.tmp", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow([*source_rows[0], *LABELS])
            for i, cells in enumerate(source_rows[1:]):
                for j, column in enumerate(FEATURE_COLUMNS[1:], start=1):
                    old, new = baseline[column].iloc[i], result[column].iloc[i]
                    if not ((pd.isna(old) and pd.isna(new)) or old == new):
                        cells[j] = "" if pd.isna(new) else repr(float(new))
                event_id = result[LABELS[2]].iloc[i]
                writer.writerow([*cells, int(result[LABELS[0]].iloc[i]), result[LABELS[1]].iloc[i],
                                 "" if pd.isna(event_id) else event_id])
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_example(baseline: pd.DataFrame, synthetic: pd.DataFrame, events: list[dict], destination: Path) -> None:
    """Save exactly one short before/after power example."""
    event = next((e for e in events if "Power_Generated" in e["affected_variables"]), None)
    if event is None:
        return
    a, b = event["row_positions"][0], event["row_positions"][-1]
    rows = slice(max(0, a-10), min(len(baseline), b+11))
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(baseline.Timestamp.iloc[rows], baseline.Power_Generated.iloc[rows], label="Untouched baseline", color="#286090")
    axis.plot(synthetic.Timestamp.iloc[rows], synthetic.Power_Generated.iloc[rows], label="Synthetic prototype", color="#ce5427", linestyle="--")
    axis.axvspan(baseline.Timestamp.iloc[a], baseline.Timestamp.iloc[b]+pd.Timedelta(minutes=2), color="#ce5427", alpha=.12)
    axis.set(title=f"Synthetic example: {event['anomaly_type']}", ylabel="Generated power (W)",
             xlabel="Source timestamp (timezone not specified)")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axis.grid(alpha=.2)
    axis.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def generate_file(*, config_path: Path = CONFIG, input_path: Path | None = None,
                  output_path: Path = OUTPUT, metadata_path: Path = METADATA,
                  figure_path: Path | None = FIGURE, seed: int | None = None,
                  prototype: bool = False, scenarios: list[str] | None = None) -> dict:
    config = load_config(config_path)
    input_path = Path(input_path) if input_path is not None else ROOT / config["baseline_path"]
    sources = {input_path.resolve(), Path(config_path).resolve()}
    sources.update(p.resolve() for p in (ROOT / "data").rglob("*.csv") if p.resolve() != OUTPUT.resolve())
    destinations = [Path(output_path).resolve(), Path(metadata_path).resolve()]
    if figure_path is not None:
        destinations.append(Path(figure_path).resolve())
    if len(set(destinations)) != len(destinations) or sources.intersection(destinations):
        raise ValueError("Output paths must be distinct and cannot overwrite baseline/configuration files.")
    if any((ROOT / "models").resolve() in p.parents for p in destinations):
        raise ValueError("Generator artifacts cannot be written to models/.")
    if Path(output_path).suffix != ".csv" or Path(metadata_path).suffix != ".json":
        raise ValueError("Use .csv dataset and .json metadata destinations.")
    if figure_path is not None and Path(figure_path).suffix != ".png":
        raise ValueError("Use a .png figure destination.")
    hashes = {str(p): digest(p) for p in sorted(sources)}
    model_hashes = {str(p): digest(p) for p in (ROOT / 'models').rglob('*') if p.is_file()}
    baseline = load_feature_dataset(input_path)
    synthetic, report = generate_prototype(baseline, config, seed=seed, prototype=prototype, scenarios=scenarios)
    _write_csv(Path(output_path), input_path, baseline, synthetic)
    saved = pd.read_csv(output_path)
    saved.Timestamp = pd.to_datetime(saved.Timestamp)
    report['saved_csv_validation'] = validate_prototype(baseline, saved, config, report['events'])
    # Verify exact source text for every unmodified original field, including Timestamp.
    source_text = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    output_text = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    exact_original = True
    for c in ORIGINAL_COLUMNS:
        unchanged = baseline[c].eq(synthetic[c])
        exact_original &= source_text.loc[unchanged, c].equals(output_text.loc[unchanged, c])
    report['saved_csv_validation']['unaffected_original_csv_fields_verbatim'] = bool(exact_original)
    if not all(report['saved_csv_validation'].values()):
        raise RuntimeError(f"Saved CSV failed validation: {report['saved_csv_validation']}")
    if figure_path is not None:
        save_example(baseline, synthetic, report['events'], Path(figure_path))
    report.update({'baseline_path': str(input_path), 'baseline_sha256': digest(input_path),
                   'config_file_sha256': digest(Path(config_path)), 'output_sha256': digest(Path(output_path)),
                   'generator_source_sha256': digest(Path(__file__)),
                   'feature_utility_sha256': {name: digest(ROOT / 'src/data_processing' / name)
                                             for name in ('engineer_basic_features.py', 'engineer_rolling_features.py', 'engineer_change_features.py')},
                   'protected_source_hashes': hashes,
                   'protected_sources_unchanged': all(digest(Path(p)) == h for p,h in hashes.items()),
                   'models_unchanged': model_hashes == {str(p): digest(p) for p in (ROOT/'models').rglob('*') if p.is_file()}})
    if not report['protected_sources_unchanged'] or not report['models_unchanged']:
        raise RuntimeError("Protected sources or models changed during generation.")
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prototype', action='store_true', help='Explicitly allow unsplit generator validation only.')
    parser.add_argument('--scenarios', nargs='+', help='Explicit per-run enable list, or all; YAML stays unchanged.')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--metadata', type=Path, default=METADATA)
    parser.add_argument('--figure', type=Path, default=FIGURE)
    args = parser.parse_args()
    selected = list(SCENARIOS) if args.scenarios == ['all'] else args.scenarios
    try:
        report = generate_file(config_path=args.config, input_path=args.input, output_path=args.output,
                               metadata_path=args.metadata, figure_path=args.figure,
                               seed=args.seed, prototype=args.prototype, scenarios=selected)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, yaml.YAMLError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Prototype shape: {report['shape']}; seed: {report['seed']}")
    print(f"Events: {report['event_count']}; directly modified rows: {report['directly_modified_rows']}")
    print(f"Skipped: {report['skipped_scenarios']}")
    print("Validation passed. No detector was trained.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
