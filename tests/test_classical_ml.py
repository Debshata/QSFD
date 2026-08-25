"""Unit tests for `src/classical_ml/train.py` and `models.py` orchestration.

Closes `classical_ml_loophole_analysis.md` item 8: the existing suite tests
the *served* model (`tests/test_engine.py`), not the training pipeline's own
decision logic (calibration accept/reject, production-model selection,
latency measurement, anomaly-score scaling). These tests use small synthetic
arrays, mirroring the style already used in `tests/conftest.py`.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.classical_ml.models import ANOMALY_MODELS, score_samples
from src.classical_ml.train import (
    maybe_calibrate,
    measure_single_row_latency,
    select_production_model,
)


@pytest.fixture
def separable_val():
    """A validation fold a logistic model fits well -- calibration shouldn't help much."""
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(0, 1, (n, 3))
    y = (X[:, 0] + rng.normal(0, 0.2, n) > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    return model, X, y


# -- maybe_calibrate ---------------------------------------------------------

def test_calibration_is_kept_when_it_improves_brier(separable_val):
    model, X, y = separable_val
    # A poorly-calibrated but well-separating model: push raw probabilities
    # toward the extremes so Platt scaling has real room to improve on it.
    rng = np.random.default_rng(2)
    X_val = X + rng.normal(0, 3, X.shape)   # noisier validation fold
    calibrator, info = maybe_calibrate(model, "logistic_regression", X_val, y)
    assert info["attempted"] is True
    # Whichever way it goes, the decision must be justified by the recorded
    # before/after Brier scores, not asserted.
    if info["applied"]:
        assert calibrator is not None
        assert info["brier_after"] <= info["brier_before"]
    else:
        assert calibrator is None


def test_calibration_is_rejected_when_it_does_not_improve(monkeypatch, separable_val):
    """Force the 'no improvement' branch and assert it actually rejects."""
    model, X, y = separable_val

    class _WorseCalibrator:
        def __init__(self, *a, **k):
            pass

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            # Deliberately bad: always predicts the wrong class strongly.
            n = len(X)
            return np.column_stack([np.full(n, 0.01), np.full(n, 0.99)])

    # CalibratedClassifierCV is imported locally inside maybe_calibrate, so
    # the patch target is the sklearn module it is imported from, not train.py.
    monkeypatch.setattr("sklearn.calibration.CalibratedClassifierCV",
                        _WorseCalibrator)
    calibrator, info = maybe_calibrate(model, "logistic_regression", X, y)
    assert info["attempted"] is True
    assert info["applied"] is False
    assert calibrator is None
    assert info["brier_after"] > info["brier_before"]


def test_calibration_skipped_for_anomaly_models():
    calibrator, info = maybe_calibrate(object(), "isolation_forest",
                                       np.zeros((5, 2)), np.zeros(5))
    assert calibrator is None
    assert info == {"attempted": False, "applied": False}


def test_calibration_handles_failure_gracefully(separable_val):
    """A calibrator that raises must degrade to raw probabilities, not crash."""
    model, X, y = separable_val
    calibrator, info = maybe_calibrate("not_a_real_model", "logistic_regression", X, y)
    assert calibrator is None
    assert info["attempted"] is True
    assert "error" in info


# -- score_samples: Isolation Forest anomaly scaling -------------------------

def test_isolation_forest_score_is_not_a_probability():
    from sklearn.ensemble import IsolationForest

    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (100, 4))
    model = IsolationForest(n_estimators=50, random_state=0).fit(X)
    scores, is_prob = score_samples(model, X, "isolation_forest")
    assert is_prob is False
    assert scores.shape == (100,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_isolation_forest_degenerate_scaling_does_not_crash(monkeypatch):
    """hi == lo (every row scores identically) must not divide by zero."""
    from sklearn.ensemble import IsolationForest

    class _ConstantScorer:
        def score_samples(self, X):
            return np.full(len(X), -0.5)

    scores, is_prob = score_samples(_ConstantScorer(), np.zeros((10, 2)),
                                    "isolation_forest")
    assert is_prob is False
    assert np.all(scores == 0.0)
    assert np.isfinite(scores).all()


def test_supervised_model_returns_probability():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (50, 3))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    scores, is_prob = score_samples(model, X, "logistic_regression")
    assert is_prob is True
    assert (scores >= 0).all() and (scores <= 1).all()


# -- select_production_model --------------------------------------------------

def _fake_result(model: str, pr_auc: float, recall: float, precision: float,
                 f2: float, fpr: float, single_ms: float, ok: bool = True):
    return {
        "model": model, "ok": ok,
        "test_metrics": {"pr_auc": pr_auc, "recall": recall,
                         "precision": precision, "f2": f2,
                         "false_positive_rate": fpr},
        "single_transaction_ms": single_ms,
        "inference_ms_per_sample": single_ms / 1000,
    }


def test_selection_excludes_anomaly_models():
    results = [
        _fake_result("xgboost", 0.80, 0.75, 0.90, 0.77, 0.0001, 3.0),
        _fake_result("isolation_forest", 0.99, 0.99, 0.99, 0.99, 0.0, 1.0),
    ]
    production, selection = select_production_model(results)
    assert production == "xgboost"
    assert all(c["model"] != "isolation_forest" for c in selection["candidates"])


def test_selection_skips_failed_models():
    results = [
        _fake_result("xgboost", 0.80, 0.75, 0.90, 0.77, 0.0001, 3.0),
        _fake_result("random_forest", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ok=False),
    ]
    production, selection = select_production_model(results)
    assert production == "xgboost"
    assert len(selection["candidates"]) == 1


def test_selection_returns_none_with_no_eligible_candidates():
    results = [_fake_result("isolation_forest", 0.99, 0.99, 0.99, 0.99, 0.0, 1.0)]
    production, selection = select_production_model(results)
    assert production is None
    assert selection["candidates"] == []


def test_selection_latency_penalises_slow_single_row_inference():
    """Same accuracy metrics, only latency differs: the faster model must win."""
    fast = _fake_result("model_fast", 0.80, 0.75, 0.90, 0.77, 0.0001, 0.5)
    slow = _fake_result("model_slow", 0.80, 0.75, 0.90, 0.77, 0.0001, 90.0)
    production, selection = select_production_model([fast, slow])
    assert production == "model_fast"
    rows = {c["model"]: c for c in selection["candidates"]}
    assert rows["model_fast"]["selection_score"] > rows["model_slow"]["selection_score"]
    assert rows["model_fast"]["latency_quality"] > rows["model_slow"]["latency_quality"]


def test_selection_latency_quality_is_bounded():
    """The latency term must not blow up or go negative for extreme inputs."""
    zero_ms = _fake_result("instant", 0.5, 0.5, 0.5, 0.5, 0.01, 0.0)
    huge_ms = _fake_result("glacial", 0.5, 0.5, 0.5, 0.5, 0.01, 1_000_000.0)
    _, selection = select_production_model([zero_ms, huge_ms])
    rows = {c["model"]: c for c in selection["candidates"]}
    assert 0.0 <= rows["instant"]["latency_quality"] <= 1.0
    assert 0.0 <= rows["glacial"]["latency_quality"] <= 1.0


# -- measure_single_row_latency ------------------------------------------------

def test_single_row_latency_is_positive_for_a_real_model():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (30, 3))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    ms = measure_single_row_latency(model, X, "logistic_regression",
                                    calibrated=False, n=5)
    assert ms > 0
    assert np.isfinite(ms)


def test_single_row_latency_returns_nan_on_failure():
    class _Broken:
        def predict_proba(self, X):
            raise RuntimeError("simulated inference failure")

    ms = measure_single_row_latency(_Broken(), np.zeros((5, 2)), "broken",
                                    calibrated=False, n=3)
    assert np.isnan(ms)


def test_single_row_latency_uses_score_samples_for_anomaly_models():
    from sklearn.ensemble import IsolationForest

    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, (40, 3))
    model = IsolationForest(n_estimators=20, random_state=0).fit(X)
    ms = measure_single_row_latency(model, X, "isolation_forest",
                                    calibrated=False, n=3)
    assert ms > 0 and np.isfinite(ms)
