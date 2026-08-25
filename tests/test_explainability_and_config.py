"""Explainability, configuration and reproducibility."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.fraud_engine.explainability import Explainer, global_importance_report
from src.utils.config import get_seed, load_config
from src.utils.reproducibility import environment_info, set_global_seed


@pytest.fixture
def toy():
    rng = np.random.default_rng(13)
    X = rng.normal(0, 1, (300, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.4, 300) > 0.8).astype(int)
    return X, y, [f"f{i}" for i in range(5)]


# -- explainability ---------------------------------------------------------

def test_logistic_explanations_are_signed_contributions(toy):
    X, y, names = toy
    model = LogisticRegression(max_iter=1000).fit(X, y)
    ex = Explainer(model, "logistic_regression", names)
    assert ex.method == "coefficient"

    out = ex.explain(X[0])
    assert out
    assert len(out) <= ex.top_k
    for c in out:
        assert c["feature"] in names
        assert c["direction"] in ("increases_fraud_risk", "decreases_fraud_risk")
        assert (c["contribution"] > 0) == (c["direction"] == "increases_fraud_risk")


def test_explanations_are_ranked_by_absolute_contribution(toy):
    X, y, names = toy
    ex = Explainer(LogisticRegression(max_iter=1000).fit(X, y),
                   "logistic_regression", names)
    magnitudes = [abs(c["contribution"]) for c in ex.explain(X[0])]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_tree_explanations_are_produced(toy):
    X, y, names = toy
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    ex = Explainer(model, "random_forest", names)
    assert ex.method in ("shap", "native_importance")
    assert ex.explain(X[0])


def test_isolation_forest_claims_no_attribution(toy):
    X, _, names = toy
    ex = Explainer(IsolationForest(n_estimators=30, random_state=0).fit(X),
                   "isolation_forest", names)
    assert ex.method == "unsupported"
    assert ex.explain(X[0]) == []


def test_explainer_survives_a_wrong_shaped_row(toy):
    X, y, names = toy
    ex = Explainer(LogisticRegression(max_iter=1000).fit(X, y),
                   "logistic_regression", names)
    assert ex.explain(np.zeros(99)) == []      # logged and empty, not a crash


def test_global_importance_is_ranked(toy):
    X, y, names = toy
    df = global_importance_report(
        RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y),
        "random_forest", names, None, None)
    assert list(df["rank"]) == list(range(1, len(names) + 1))
    assert df["importance"].is_monotonic_decreasing
    assert set(df["feature"]) == set(names)


def test_global_importance_rejects_a_model_without_one(toy):
    X, _, names = toy
    with pytest.raises(TypeError, match="No global importance"):
        global_importance_report(IsolationForest().fit(X), "isolation_forest",
                                 names, None, None)


# -- configuration ----------------------------------------------------------

@pytest.mark.parametrize("name", ["base", "datasets", "classical", "qml", "risk"])
def test_every_config_loads(name):
    assert isinstance(load_config(name), dict)


def test_unknown_config_raises():
    with pytest.raises(KeyError, match="Unknown config"):
        load_config("not_a_config")


def test_config_returns_a_copy():
    """Mutating a returned config must not poison the cache."""
    a = load_config("risk")
    a["risk"]["low_upper_bound"] = 0.999
    assert load_config("risk")["risk"]["low_upper_bound"] != 0.999


def test_dataset_registry_is_well_formed():
    cfg = load_config("datasets")
    assert cfg["primary_dataset"] in cfg["datasets"]
    for name, entry in cfg["datasets"].items():
        assert entry.get("kaggle_id"), f"{name} has no kaggle_id"
        assert isinstance(entry.get("target_positive"), list)


def test_qml_stays_simulator_sized():
    qml = load_config("qml")
    assert 4 <= qml["feature_reduction"]["n_components"] <= 8
    assert qml["subset"]["train_samples"] <= 2000, "QML subset must stay tractable"


def test_classical_selection_weights_are_sane():
    weights = load_config("classical")["selection"]["weights"]
    assert weights["pr_auc"] == max(weights.values()), "PR-AUC must dominate selection"


def test_threshold_objective_is_supported():
    assert load_config("classical")["threshold"]["objective"] in (
        "f1", "f2", "youden", "max_recall_at_precision", "max_precision_at_recall")


# -- reproducibility --------------------------------------------------------

def test_seeding_makes_numpy_deterministic():
    set_global_seed(get_seed())
    a = np.random.rand(5)
    set_global_seed(get_seed())
    np.testing.assert_array_equal(a, np.random.rand(5))


def test_environment_info_records_versions():
    info = environment_info()
    assert info["python"] and info["platform"]
    for pkg in ("numpy", "pandas", "sklearn", "xgboost"):
        assert info["packages"][pkg] != "not installed"
