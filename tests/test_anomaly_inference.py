"""Contract tests for the closed, frozen operational inference module."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.anomaly_detection.finalize_operational_module import (
    BASELINE_INPUT,
    DEFAULT_FEED,
    DEFAULT_STATUS,
    EXPECTED_DAY22_HASHES,
    FEED_COLUMNS,
    ROOT,
    build_outputs,
)
from src.anomaly_detection.inference import (
    EXPECTED_THRESHOLD,
    FEATURE_ORDER,
    FrozenDetector,
    group_alert_events,
    load_frozen_detector,
    score_dataframe,
    score_observation,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def detector() -> FrozenDetector:
    return load_frozen_detector()


@pytest.fixture(scope="module")
def baseline() -> pd.DataFrame:
    return pd.read_csv(BASELINE_INPUT)


def test_frozen_loader_uses_exact_feature_order_and_hashes(detector: FrozenDetector) -> None:
    assert detector.feature_order == FEATURE_ORDER
    assert detector.threshold == pytest.approx(EXPECTED_THRESHOLD, abs=1e-15)
    assert detector.verification["fit_called"] is False
    for name in ("scaler", "model", "detector_config", "detector_metadata",
                 "threshold_config", "threshold_metadata"):
        record = detector.verification[name]
        assert _sha256(ROOT / record["path"]) == record["sha256"]


def test_loading_and_scoring_never_call_fit(
    monkeypatch: pytest.MonkeyPatch, baseline: pd.DataFrame
) -> None:
    def forbidden_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fit() is forbidden in operational inference")

    monkeypatch.setattr(StandardScaler, "fit", forbidden_fit)
    monkeypatch.setattr(IsolationForest, "fit", forbidden_fit)
    active = load_frozen_detector()
    output = score_dataframe(baseline.head(3), active)
    assert len(output) == 3


def test_dataframe_uses_frozen_order_and_preserves_timestamp(
    detector: FrozenDetector, baseline: pd.DataFrame
) -> None:
    scrambled = baseline.loc[:, ["Timestamp", *reversed(FEATURE_ORDER)]].head(4)
    output = score_dataframe(scrambled, detector)
    reference = score_dataframe(baseline.head(4), detector)
    assert output.columns[:8].tolist() == ["Timestamp", *FEATURE_ORDER]
    assert output["Timestamp"].equals(scrambled["Timestamp"].reset_index(drop=True))
    assert np.allclose(output.anomaly_score, reference.anomaly_score)


def test_single_observation_and_optional_context(
    detector: FrozenDetector, baseline: pd.DataFrame
) -> None:
    result = score_observation(baseline.iloc[0], detector, include_context=False)
    assert set(result) == {"Timestamp", "anomaly_score", "threshold", "threshold_margin", "status"}
    assert result["threshold_margin"] == pytest.approx(
        result["anomaly_score"] - result["threshold"]
    )


@pytest.mark.parametrize(
    "change, match",
    [
        ({"Power_Generated": None}, "finite|missing"),
        ({"Solar_Radiation": np.inf}, "finite"),
        ({"Air_Temp": "not-a-number"}, "non-numeric|finite"),
    ],
)
def test_invalid_required_values_are_rejected(
    detector: FrozenDetector, baseline: pd.DataFrame, change: dict, match: str
) -> None:
    row = baseline.iloc[[0]].copy()
    for key, value in change.items():
        row[key] = pd.Series([value], index=row.index, dtype=object)
    with pytest.raises(ValueError, match=match):
        score_dataframe(row, detector)


def test_missing_feature_is_rejected_clearly(
    detector: FrozenDetector, baseline: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="Missing required.*Wind_Speed"):
        score_dataframe(baseline.drop(columns="Wind_Speed").head(1), detector)


def test_strict_threshold_status_and_margin() -> None:
    class IdentityScaler:
        def transform(self, values: np.ndarray) -> np.ndarray:
            return values

    class FixedModel:
        def score_samples(self, values: np.ndarray) -> np.ndarray:
            return -np.array([EXPECTED_THRESHOLD, EXPECTED_THRESHOLD + 0.01])[: len(values)]

    fake = FrozenDetector(
        IdentityScaler(), FixedModel(), EXPECTED_THRESHOLD, FEATURE_ORDER, {"fit_called": False}
    )
    frame = pd.DataFrame(np.zeros((2, len(FEATURE_ORDER))), columns=FEATURE_ORDER)
    output = score_dataframe(frame, fake, include_context=False)
    assert output.status.tolist() == ["NORMAL", "ALERT"]
    assert output.threshold_margin.iloc[0] == pytest.approx(0.0)
    assert output.threshold_margin.iloc[1] == pytest.approx(0.01)


def test_event_grouping_uses_exact_two_minute_rule() -> None:
    scored = pd.DataFrame({
        "Timestamp": pd.to_datetime([
            "2022-01-01 00:00", "2022-01-01 00:02", "2022-01-01 00:05",
            "2022-01-01 00:07", "2022-01-01 00:09", "2022-01-01 00:11",
        ]),
        "anomaly_score": [0.8, 0.9, 1.0, 0.1, 1.1, 1.2],
        "status": ["ALERT", "ALERT", "ALERT", "NORMAL", "ALERT", "ALERT"],
    })
    events = group_alert_events(scored)
    assert events.row_count.tolist() == [2, 1, 2]
    assert events.duration_minutes.tolist() == [4, 2, 4]
    assert events.maximum_anomaly_score.tolist() == [0.9, 1.0, 1.2]
    assert events.event_id.tolist() == ["alert-001", "alert-002", "alert-003"]


def test_empty_alert_set_returns_stable_event_schema() -> None:
    events = group_alert_events(pd.DataFrame({
        "Timestamp": ["2022-01-01"], "anomaly_score": [0.5], "status": ["NORMAL"]
    }))
    assert events.empty
    assert events.columns.tolist() == [
        "event_id", "start_timestamp", "end_timestamp", "row_count",
        "duration_minutes", "maximum_anomaly_score", "mean_anomaly_score",
    ]


def test_dashboard_feed_matches_inference_and_preserves_zero_alert_result(
    detector: FrozenDetector, baseline: pd.DataFrame
) -> None:
    feed = pd.read_csv(DEFAULT_FEED)
    expected = score_dataframe(baseline, detector).loc[:, FEED_COLUMNS]
    assert feed.columns.tolist() == list(FEED_COLUMNS)
    assert feed.Timestamp.tolist() == expected.Timestamp.tolist()
    assert np.allclose(feed.anomaly_score, expected.anomaly_score)
    assert np.allclose(feed.threshold_margin, expected.threshold_margin)
    assert feed.status.tolist() == expected.status.tolist()
    assert feed.status.eq("ALERT").sum() == 0


def test_finalizer_preserves_day22_and_model_artifacts(tmp_path: Path) -> None:
    protected = {relative: _sha256(ROOT / relative) for relative in EXPECTED_DAY22_HASHES}
    joblib_before = {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in (ROOT / "models").rglob("*.joblib")
    }
    status = build_outputs(tmp_path / "feed.csv", tmp_path / "status.json")
    assert protected == {relative: _sha256(ROOT / relative) for relative in EXPECTED_DAY22_HASHES}
    assert joblib_before == {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in (ROOT / "models").rglob("*.joblib")
    }
    assert status["validation"]["fit_called"] is False
    assert status["validation"]["new_training_performed"] is False
    assert status["status"] == "operational_module_complete_for_prototype"


def test_saved_status_records_feed_hash_and_no_training() -> None:
    status = json.loads(DEFAULT_STATUS.read_text(encoding="utf-8"))
    assert status["dashboard_feed"]["sha256"] == _sha256(DEFAULT_FEED)
    assert status["dashboard_feed"]["schema"] == list(FEED_COLUMNS)
    assert status["day22_evaluation_summary"]["predicted_alert_rows"] == 0
    assert status["validation"]["new_training_performed"] is False
