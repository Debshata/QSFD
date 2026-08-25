"""Metrics, threshold selection and the risk engine."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics, select_threshold
from src.fraud_engine.risk import HIGH, LOW, MEDIUM, RiskConfig, RiskEngine


# -- metrics ----------------------------------------------------------------

def test_metrics_on_a_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.01, 0.02, 0.98, 0.99]), 0.5)
    assert m["recall"] == 1.0 and m["precision"] == 1.0
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert m["fp"] == 0 and m["fn"] == 0
    assert m["false_positive_rate"] == 0.0


def test_metrics_on_an_all_negative_predictor():
    """The failure mode the SRS warns about: high accuracy, zero detection."""
    y = np.concatenate([np.zeros(999, int), np.ones(1, int)])
    m = compute_metrics(y, np.zeros(1000), 0.5)
    assert m["accuracy"] > 0.99      # looks great
    assert m["recall"] == 0.0        # catches nothing
    assert m["f1"] == 0.0
    assert m["balanced_accuracy"] == pytest.approx(0.5)


def test_metrics_confusion_matrix_is_consistent():
    rng = np.random.default_rng(3)
    y = (rng.random(400) < 0.2).astype(int)
    s = rng.random(400)
    m = compute_metrics(y, s, 0.5)
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == 400
    assert m["tp"] + m["fn"] == int(y.sum())


def test_brier_omitted_for_non_probability_scores():
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, np.array([-3.0, 4.0, -1.0, 2.0]), 0.0,
                        score_is_probability=False)
    assert not m["score_is_probability"]
    assert np.isnan(m["brier_score"])


# -- threshold selection ----------------------------------------------------

def test_threshold_selection_beats_naive_half():
    """With a shifted score distribution, 0.5 is the wrong operating point."""
    rng = np.random.default_rng(11)
    y = (rng.random(2000) < 0.05).astype(int)
    score = np.clip(rng.normal(0.05, 0.03, 2000) + y * 0.15, 0, 1)
    t, info = select_threshold(y, score, objective="f2")
    assert 0.0 < t < 1.0
    assert compute_metrics(y, score, t)["f2"] >= compute_metrics(y, score, 0.5)["f2"]
    assert info["selected_on"] == "validation"


@pytest.mark.parametrize("objective", ["f1", "f2", "youden",
                                       "max_recall_at_precision",
                                       "max_precision_at_recall"])
def test_all_threshold_objectives_return_a_valid_point(objective):
    rng = np.random.default_rng(5)
    y = (rng.random(800) < 0.1).astype(int)
    score = np.clip(rng.random(800) * 0.6 + y * 0.35, 0, 1)
    t, info = select_threshold(y, score, objective=objective)
    assert score.min() < t <= score.max()
    assert info["objective"] == objective


def test_threshold_selection_is_deterministic():
    rng = np.random.default_rng(2)
    y = (rng.random(500) < 0.1).astype(int)
    s = rng.random(500)
    assert select_threshold(y, s)[0] == select_threshold(y, s)[0]


def test_threshold_handles_degenerate_scores():
    y = np.array([0, 1, 0, 1])
    t, info = select_threshold(y, np.full(4, 0.3))
    assert t == 0.5 and "reason" in info


def test_unknown_objective_raises():
    with pytest.raises(ValueError, match="Unknown threshold objective"):
        select_threshold(np.array([0, 1]), np.array([0.1, 0.9]), objective="bogus")


# -- risk engine ------------------------------------------------------------

@pytest.fixture
def engine():
    return RiskEngine(RiskConfig(low_upper_bound=0.3, medium_upper_bound=0.7,
                                 score_scale=100, enforce_threshold_floor=False))


@pytest.mark.parametrize("p,expected", [
    (0.00, LOW), (0.29, LOW), (0.30, LOW),
    (0.31, MEDIUM), (0.50, MEDIUM), (0.70, MEDIUM),
    (0.71, HIGH), (1.00, HIGH),
])
def test_risk_bands_including_boundaries(engine, p, expected):
    assert engine.level(p) == expected


def test_risk_score_is_probability_scaled(engine):
    assert engine.score(0.0) == 0.0
    assert engine.score(0.876) == 87.6
    assert engine.score(1.0) == 100.0


def test_risk_score_rejects_out_of_range(engine):
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match="must be in"):
            engine.score(bad)


def test_threshold_floor_lifts_a_flagged_low_probability():
    """A flagged transaction must never be reported as LOW risk."""
    e = RiskEngine(RiskConfig(0.3, 0.7, 100, enforce_threshold_floor=True))
    assert e.level(0.15, threshold=0.10) == MEDIUM     # flagged -> floored
    assert e.level(0.05, threshold=0.10) == LOW        # not flagged -> LOW


def test_assess_returns_the_full_contract(engine):
    out = engine.assess(0.82, threshold=0.4)
    assert out["fraud_prediction"] == 1
    assert out["risk_level"] == HIGH
    assert out["risk_score"] == 82.0
    assert set(out) == {"fraud_probability", "fraud_prediction", "risk_score",
                        "risk_level", "threshold"}


def test_risk_config_rejects_inverted_bands():
    with pytest.raises(ValueError, match="Risk bands must satisfy"):
        RiskConfig(0.8, 0.2, 100, True).validate()


def test_risk_config_loads_from_yaml():
    cfg = RiskConfig.from_config()
    assert 0.0 <= cfg.low_upper_bound < cfg.medium_upper_bound <= 1.0


def test_describe_documents_the_mapping():
    described = RiskEngine().describe()
    assert set(described["mapping"]) == {LOW, MEDIUM, HIGH}
