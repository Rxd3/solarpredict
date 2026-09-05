"""Exercise the generator against the real baseline without overwriting project artifacts."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.anomaly_detection.generate_synthetic_anomalies import (
    CONFIG, ROOT, SCENARIOS, LABELS, load_config, generate_prototype,
    generate_file, digest, excluded_rows, eligible_starts, validate_prototype,
)
from src.data_processing.engineer_basic_features import ORIGINAL_COLUMNS
from src.data_processing.review_model_features import load_feature_dataset


@pytest.fixture(scope="module")
def baseline():
    return load_feature_dataset(ROOT / "data/processed/operational_features_change.csv")


@pytest.fixture(scope="module")
def generated(baseline):
    return generate_prototype(baseline, load_config(), seed=42, prototype=True, scenarios=list(SCENARIOS))


def test_config_loading_and_invalid_durations(tmp_path):
    config = load_config()
    assert config["enabled"] is False
    assert set(config["scenarios"]) == set(SCENARIOS)
    config["scenarios"][SCENARIOS[0]]["duration_minutes"] = [3, 7]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        load_config(path)


def test_determinism_and_no_dataframe_mutation(baseline, generated):
    snapshot = baseline.copy(deep=True)
    config = load_config()
    original_config = copy.deepcopy(config)
    repeated, metadata = generate_prototype(baseline, config, seed=42, prototype=True, scenarios=list(SCENARIOS))
    pd.testing.assert_frame_equal(repeated, generated[0], check_exact=True)
    assert metadata == generated[1]
    pd.testing.assert_frame_equal(baseline, snapshot, check_exact=True)
    assert config == original_config


def test_different_seed_produces_different_valid_output(baseline, generated):
    other, report = generate_prototype(baseline, load_config(), seed=43, prototype=True, scenarios=list(SCENARIOS))
    assert all(report["validation"].values())
    assert not other.equals(generated[0])
    assert report["run_id"] != generated[1]["run_id"]


def test_event_labels_durations_and_context(baseline, generated):
    result, report = generated
    assert report["event_count"] == 7
    assert report["skipped_scenarios"] == []
    assert result.shape == (len(baseline), baseline.shape[1] + 3)
    assert set(result.synthetic_anomaly.unique()) == {0, 1}
    assert result.Timestamp.equals(baseline.Timestamp)
    changed = result[list(ORIGINAL_COLUMNS[1:])].ne(baseline[list(ORIGINAL_COLUMNS[1:])]).any(axis=1)
    np.testing.assert_array_equal(changed, result.synthetic_anomaly.astype(bool))
    occupied = set()
    ordered = sorted(report['events'], key=lambda e: e['row_positions'][0])
    for i, event in enumerate(ordered):
        rows = event["row_positions"]
        assert not occupied.intersection(rows)
        occupied.update(rows)
        assert rows == list(range(rows[0], rows[0] + len(rows)))
        assert len(rows)*2 == event['duration_minutes']
        assert (pd.Timestamp(event['end_exclusive']) - pd.Timestamp(event['start_timestamp'])).total_seconds() == len(rows)*120
        assert result.loc[rows, 'synthetic_anomaly_id'].eq(event['anomaly_id']).all()
        assert result.loc[rows, 'synthetic_anomaly_type'].eq(event['anomaly_type']).all()
        if event['anomaly_type'] != 'sensor_excursion':
            assert baseline.loc[rows, 'Solar_Radiation'].gt(100).all()
            assert baseline.loc[rows, 'Power_Generated'].ge(50).all()
            assert baseline.loc[rows, 'low_light_context'].eq(0).all()
        if i:
            assert (pd.Timestamp(event['start_timestamp']) - pd.Timestamp(ordered[i-1]['end_exclusive'])).total_seconds() >= 3600
    assert len(occupied) == report['directly_modified_rows'] <= len(baseline)*.05
    assert result.loc[~changed, 'synthetic_anomaly_id'].isna().all()
    assert result.loc[~changed, 'synthetic_anomaly_type'].eq('none').all()


def test_excluded_windows_and_unaffected_originals_exact(baseline, generated):
    result, report = generated
    mask = excluded_rows(baseline, load_config())
    pd.testing.assert_frame_equal(result.loc[mask, baseline.columns], baseline.loc[mask], check_exact=True)
    permitted = pd.DataFrame(False, index=baseline.index, columns=ORIGINAL_COLUMNS[1:])
    for event in report['events']:
        permitted.loc[event['row_positions'], event['affected_variables']] = True
    for column in ORIGINAL_COLUMNS[1:]:
        pd.testing.assert_series_equal(result.loc[~permitted[column], column], baseline.loc[~permitted[column], column], check_exact=True)


def test_recomputed_features_against_independent_formulas(baseline, generated):
    result, report = generated
    np.testing.assert_allclose(result.electrical_power_product, result.Array_Voltage*result.Array_Current, atol=1e-10)
    for source, prefix in [('Power_Generated','power_generated'), ('Solar_Radiation','solar_radiation'),
                           ('Air_Temp','air_temp'), ('Relative_Humidity','relative_humidity')]:
        np.testing.assert_allclose(result[prefix+'_diff'], result[source].diff(), atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(result[prefix+'_rate_per_min'], result[source].diff()/2, atol=1e-10, equal_nan=True)
        for minutes in [30,60]:
            n=minutes//2
            np.testing.assert_allclose(result[f'{prefix}_roll_mean_{minutes}m'],
                                       result[source].rolling(n,min_periods=n).mean(), atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(result.power_deviation_from_30m_mean,
                               result.Power_Generated-result.power_generated_roll_mean_30m, atol=1e-10, equal_nan=True)
    previous = result.Power_Generated.shift()
    expected = result.Power_Generated.diff()/previous.where(previous.abs() >= 1)
    np.testing.assert_allclose(result.power_generated_relative_change, expected, atol=1e-10, equal_nan=True)
    assert not np.isinf(result.select_dtypes('number').to_numpy()).any()
    # A recovery/trailing-history row is deliberately updated without a direct label.
    power_event = next(e for e in report['events'] if e['anomaly_type']=='sudden_power_drop')
    after = power_event['row_positions'][-1]+1
    assert result.synthetic_anomaly.iloc[after] == 0
    assert result.power_generated_diff.iloc[after] != baseline.power_generated_diff.iloc[after]


def test_invalid_context_skips_without_forcing(baseline):
    config = load_config()
    config['context']['power_event_min_radiation_w_m2'] = 1e9
    result, report = generate_prototype(baseline, config, prototype=True, scenarios=['sudden_power_drop'])
    assert report['event_count'] == 0
    assert report['skipped_scenarios'][0]['reason'].startswith('no_window')
    pd.testing.assert_frame_equal(result[baseline.columns], baseline, check_exact=True)


def test_explicit_zero_power_has_safe_ratios(baseline):
    config = load_config()
    config['scenarios']['temporary_near_zero_power']['severity']['range'] = [0.0, 0.0]
    result, report = generate_prototype(baseline, config, seed=42, prototype=True,
                                        scenarios=['temporary_near_zero_power'])
    rows = report['events'][0]['row_positions']
    assert result.loc[rows, 'Power_Generated'].eq(0).all()
    assert result.loc[rows, 'Array_Voltage'].eq(0).all()
    assert result.Array_Current.equals(baseline.Array_Current)
    assert not np.isinf(result.select_dtypes('number').to_numpy()).any()


@pytest.mark.parametrize('seed,expected_sensor', [(0, 'Relative_Humidity'), (1, 'Air_Temp')])
def test_sensor_offsets_only_change_one_original_sensor(baseline, seed, expected_sensor):
    result, report = generate_prototype(baseline, load_config(), seed=seed, prototype=True,
                                        scenarios=['sensor_excursion'])
    event = report['events'][0]
    assert event['affected_variables'] == [expected_sensor]
    rows = event['row_positions']
    np.testing.assert_allclose(result.loc[rows, expected_sensor] - baseline.loc[rows, expected_sensor],
                               event['severity']['signed_offset'], atol=1e-10)
    assert result.Power_Generated.equals(baseline.Power_Generated)


def test_candidate_windows_reject_gaps_nonfinite_and_humidity_bounds(baseline):
    altered = baseline.iloc[:40].copy()
    altered.loc[10, 'Timestamp'] += pd.Timedelta(seconds=30)
    assert 10 not in eligible_starts(altered, load_config(), 'sensor_excursion', 2, [], 'Air_Temp', 2)
    altered = baseline.iloc[:40].copy()
    altered.loc[10, 'Air_Temp'] = np.nan
    assert 10 not in eligible_starts(altered, load_config(), 'sensor_excursion', 2, [], 'Air_Temp', 2)
    assert not eligible_starts(baseline, load_config(), 'sensor_excursion', 2, [], 'Relative_Humidity', 1000)


def test_disabled_defaults_and_explicit_prototype_scope(baseline):
    with pytest.raises(ValueError, match='prototype=True'):
        generate_prototype(baseline, load_config())
    result, report = generate_prototype(baseline, load_config(), prototype=True)
    assert report['event_count'] == 0
    assert report['disabled_scenarios'] == list(SCENARIOS)
    pd.testing.assert_frame_equal(result[baseline.columns], baseline, check_exact=True)


def test_validator_detects_bad_labels_and_stale_features(baseline, generated):
    result, report = generated
    broken = result.copy(deep=True)
    row = report['events'][0]['row_positions'][0]
    broken.loc[row, 'synthetic_anomaly'] = 0
    broken.loc[row, 'power_generated_roll_mean_30m'] = baseline.loc[row, 'power_generated_roll_mean_30m']
    checks = validate_prototype(baseline, broken, load_config(), report['events'])
    assert not checks['labels_match_direct_measurement_changes']
    assert not checks['dependent_features_recomputed']


def test_saved_bytes_reproducible_sources_protected(tmp_path):
    input_path = ROOT / 'data/processed/operational_features_change.csv'
    before = digest(input_path)
    first = tmp_path / 'first.csv'
    second = tmp_path / 'second.csv'
    kwargs = dict(input_path=input_path, figure_path=None, prototype=True, scenarios=['sudden_power_drop'], seed=42)
    report = generate_file(output_path=first, metadata_path=tmp_path/'first.json', **kwargs)
    repeated = generate_file(output_path=second, metadata_path=tmp_path/'second.json', **kwargs)
    assert first.read_bytes() == second.read_bytes()
    assert report['output_sha256'] == repeated['output_sha256']
    assert digest(input_path) == before
    assert all(report['saved_csv_validation'].values())
    assert report['protected_sources_unchanged'] and report['models_unchanged']
    with pytest.raises(ValueError, match='overwrite'):
        generate_file(output_path=input_path, metadata_path=tmp_path/'refused.json', **kwargs)
    assert digest(input_path) == before
    assert not (tmp_path/'refused.json').exists()
