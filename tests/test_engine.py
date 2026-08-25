"""Fraud engine: artifact loading, inference, schema and malformed input.

These tests use the real persisted artifacts, so they exercise the same code
path the security layer will. They skip cleanly when nothing is trained yet.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.fraud_engine.engine import FraudDetectionEngine, ModelNotFoundError
from src.fraud_engine.schemas import CONTRACT_FIELDS, FraudAssessment

pytestmark = pytest.mark.artifacts


@pytest.fixture
def sample_transaction(trained_engine) -> dict:
    """A syntactically valid transaction in the training schema."""
    return {name: 0.0 for name in trained_engine.raw_features}


# -- loading ----------------------------------------------------------------

def test_engine_loads_from_disk(trained_engine):
    assert trained_engine.model is not None
    assert trained_engine.preprocessor is not None
    assert 0.0 < trained_engine.threshold < 1.0
    assert trained_engine.raw_features
    assert trained_engine.feature_names


def test_engine_reload_is_stable(sample_transaction, trained_engine):
    """A freshly loaded engine must reproduce the same probability."""
    other = FraudDetectionEngine.load()
    a = trained_engine.predict(sample_transaction, explain=False)
    b = other.predict(sample_transaction, explain=False)
    assert a.fraud_probability == pytest.approx(b.fraud_probability)
    assert a.risk_level == b.risk_level


def test_loading_an_unknown_model_raises(tmp_path):
    with pytest.raises(ModelNotFoundError):
        FraudDetectionEngine.load("no_such_model", models_dir=tmp_path)


def test_describe_reports_configuration(trained_engine):
    d = trained_engine.describe()
    for key in ("model_name", "model_version", "threshold", "risk",
                "explainability_method"):
        assert key in d


# -- single-transaction inference ------------------------------------------

def test_predict_returns_a_valid_assessment(trained_engine, sample_transaction):
    result = trained_engine.predict(sample_transaction, transaction_id="tx-1")
    assert isinstance(result, FraudAssessment)
    result.validate()
    assert result.transaction_id == "tx-1"
    assert 0.0 <= result.fraud_probability <= 1.0
    assert result.fraud_prediction in (0, 1)
    assert result.risk_level in ("LOW", "MEDIUM", "HIGH")
    assert 0.0 <= result.risk_score <= 100.0
    assert result.latency_ms >= 0.0


def test_response_contains_every_contract_field(trained_engine, sample_transaction):
    payload = trained_engine.predict(sample_transaction).to_dict()
    missing = [f for f in CONTRACT_FIELDS if f not in payload]
    assert not missing, f"contract fields missing: {missing}"


def test_response_is_json_serialisable(trained_engine, sample_transaction):
    payload = trained_engine.predict(sample_transaction).to_dict()
    assert json.loads(json.dumps(payload, default=str))["risk_level"] in (
        "LOW", "MEDIUM", "HIGH")


def test_prediction_agrees_with_threshold(trained_engine, sample_transaction):
    r = trained_engine.predict(sample_transaction)
    assert r.fraud_prediction == int(r.fraud_probability >= r.threshold)


def test_risk_score_tracks_probability(trained_engine, sample_transaction):
    r = trained_engine.predict(sample_transaction)
    assert r.risk_score == pytest.approx(round(r.fraud_probability * 100, 2), abs=0.01)


def test_flagged_transaction_is_never_low_risk(trained_engine, sample_transaction):
    r = trained_engine.predict(sample_transaction)
    if r.fraud_prediction == 1:
        assert r.risk_level in ("MEDIUM", "HIGH")


def test_explainability_returns_contributing_features(trained_engine,
                                                      sample_transaction):
    r = trained_engine.predict(sample_transaction, explain=True)
    if trained_engine.explainer.method == "unsupported":
        assert r.top_contributing_features == []
        return
    assert r.top_contributing_features, "expected top contributing features"
    for c in r.top_contributing_features:
        assert set(c) >= {"feature", "contribution", "direction"}
        assert c["feature"] in trained_engine.feature_names
        assert isinstance(c["contribution"], float)


def test_predictions_are_deterministic(trained_engine, sample_transaction):
    probs = {trained_engine.predict(sample_transaction, explain=False).fraud_probability
             for _ in range(5)}
    assert len(probs) == 1


def test_accepts_a_pandas_series(trained_engine, sample_transaction):
    trained_engine.predict(pd.Series(sample_transaction)).validate()


def test_extra_columns_are_ignored(trained_engine, sample_transaction):
    enriched = {**sample_transaction, "merchant_note": "irrelevant", "channel": "web"}
    trained_engine.predict(enriched).validate()


# -- batch inference --------------------------------------------------------

def test_batch_inference_matches_single(trained_engine, sample_transaction):
    batch = trained_engine.predict_batch([sample_transaction] * 3)
    assert len(batch) == 3
    single = trained_engine.predict(sample_transaction, explain=False)
    for r in batch:
        r.validate()
        assert r.fraud_probability == pytest.approx(single.fraud_probability)


def test_batch_accepts_a_dataframe(trained_engine, sample_transaction):
    df = pd.DataFrame([sample_transaction] * 4)
    results = trained_engine.predict_batch(df, transaction_ids=list("abcd"))
    assert [r.transaction_id for r in results] == list("abcd")


def test_batch_with_explanations(trained_engine, sample_transaction):
    results = trained_engine.predict_batch([sample_transaction] * 2, explain=True)
    if trained_engine.explainer.method != "unsupported":
        assert all(r.top_contributing_features for r in results)


# -- malformed input --------------------------------------------------------

def test_missing_features_raise_a_clear_error(trained_engine, sample_transaction):
    partial = dict(list(sample_transaction.items())[:-3])
    with pytest.raises(ValueError, match="missing .* required feature"):
        trained_engine.predict(partial)


def test_empty_batch_raises(trained_engine):
    with pytest.raises(ValueError, match="Empty transaction batch"):
        trained_engine.predict_batch([])


def test_wrong_type_raises(trained_engine):
    with pytest.raises(TypeError):
        trained_engine.predict_batch(42)


def test_non_dict_batch_elements_raise(trained_engine):
    with pytest.raises(TypeError, match="sequence of dicts"):
        trained_engine.predict_batch([1, 2, 3])


def test_nan_input_does_not_crash(trained_engine, sample_transaction):
    """Imputation is inside the pipeline, so a missing value must be survivable."""
    noisy = {**sample_transaction,
             list(sample_transaction)[0]: float("nan")}
    trained_engine.predict(noisy, explain=False).validate()


# -- schema contract --------------------------------------------------------

def test_assessment_validate_rejects_bad_values():
    def make(**kw):
        base = dict(transaction_id="t", fraud_probability=0.5, fraud_prediction=1,
                    risk_score=50.0, risk_level="MEDIUM", model_name="m",
                    model_version="v", threshold=0.3)
        base.update(kw)
        return FraudAssessment(**base)

    for bad in (dict(fraud_probability=1.5), dict(fraud_prediction=2),
                dict(risk_level="CRITICAL"), dict(risk_score=200.0),
                dict(latency_ms=-1.0)):
        with pytest.raises((ValueError, TypeError)):
            make(**bad).validate()


def test_latency_is_measured(trained_engine, sample_transaction):
    r = trained_engine.predict(sample_transaction)
    assert r.latency_ms > 0
    assert set(r.latency_breakdown_ms) >= {"preprocessing", "inference", "total"}


def test_inference_latency_is_realtime_capable(trained_engine, sample_transaction):
    """The classical path must be fast enough for an online API."""
    import time

    trained_engine.predict(sample_transaction, explain=False)   # warm up
    t0 = time.perf_counter()
    for _ in range(50):
        trained_engine.predict(sample_transaction, explain=False)
    mean_ms = 1000 * (time.perf_counter() - t0) / 50
    assert mean_ms < 100, f"mean single-transaction latency {mean_ms:.1f} ms"


def test_compare_models_works_without_qml(trained_engine, sample_transaction):
    r = trained_engine.compare_models(sample_transaction, transaction_id="cmp")
    r.validate()
    assert r.classical is not None
    assert r.classical["model"] == trained_engine.model_name
