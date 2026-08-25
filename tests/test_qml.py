"""QML feature reduction, circuit construction and a genuine smoke run.

The heavy simulation tests are marked ``slow`` and are skipped automatically if
Qiskit is not installed, so the suite stays green in a classical-only
environment.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qml.feature_reduction import build_qml_subset, stratified_subsample

qiskit = pytest.importorskip("qiskit", reason="Qiskit not installed")
pytest.importorskip("qiskit_machine_learning",
                    reason="qiskit-machine-learning not installed")

from src.qml.circuits import build_ansatz, build_feature_map, circuit_stats  # noqa: E402


@pytest.fixture
def folds():
    """Imbalanced train/val/test arrays standing in for a preprocessed dataset."""
    rng = np.random.default_rng(19)

    def make(n, rate):
        y = (rng.random(n) < rate).astype(int)
        X = rng.normal(0, 1, (n, 12)) + y[:, None] * 0.9
        return X, y

    return {"train": make(4000, 0.02), "val": make(1000, 0.02),
            "test": make(1200, 0.02)}


# -- subsampling ------------------------------------------------------------

def test_subsample_hits_the_requested_fraud_ratio(folds):
    X, y = folds["train"]
    idx = stratified_subsample(X, y, 200, 0.3, seed=1)
    assert len(idx) == 200
    assert abs(y[idx].mean() - 0.3) < 0.05


def test_subsample_is_reproducible(folds):
    X, y = folds["train"]
    a = stratified_subsample(X, y, 150, 0.3, seed=42)
    b = stratified_subsample(X, y, 150, 0.3, seed=42)
    np.testing.assert_array_equal(a, b)


def test_subsample_never_exceeds_available_positives():
    y = np.array([1] * 3 + [0] * 200)
    X = np.zeros((len(y), 4))
    idx = stratified_subsample(X, y, 100, 0.5, seed=0)
    assert y[idx].sum() <= 3


# -- feature reduction ------------------------------------------------------

def test_subset_has_the_configured_qml_dimensionality(folds):
    from src.utils.config import load_config

    n_expected = int(load_config("qml")["feature_reduction"]["n_components"])
    subset = build_qml_subset(*folds["train"], *folds["val"], *folds["test"])
    assert subset.n_features == n_expected
    for X in (subset.X_train, subset.X_val, subset.X_test):
        assert X.shape[1] == n_expected
    assert 4 <= n_expected <= 8, "QML input space must stay simulator-sized"


def test_reduced_features_are_angle_bounded(folds):
    """Angle encoding requires inputs inside [0, pi], including val and test."""
    subset = build_qml_subset(*folds["train"], *folds["val"], *folds["test"])
    for X in (subset.X_train, subset.X_val, subset.X_test):
        assert X.min() >= -1e-9
        assert X.max() <= np.pi + 1e-9
        assert np.isfinite(X).all()


def test_both_classes_survive_in_every_qml_fold(folds):
    subset = build_qml_subset(*folds["train"], *folds["val"], *folds["test"])
    for y in (subset.y_train, subset.y_val, subset.y_test):
        assert 0 < y.sum() < len(y)


def test_reduction_is_fitted_on_training_data_only(folds):
    """Changing only the test fold must not change the transform of train."""
    a = build_qml_subset(*folds["train"], *folds["val"], *folds["test"])
    rng = np.random.default_rng(99)
    Xte, yte = folds["test"]
    b = build_qml_subset(*folds["train"], *folds["val"],
                         Xte + rng.normal(0, 5, Xte.shape), yte)
    np.testing.assert_allclose(a.X_train, b.X_train)


def test_subset_summary_records_the_experiment(folds):
    s = build_qml_subset(*folds["train"], *folds["val"], *folds["test"]).summary()
    for key in ("qml_training_samples", "qml_validation_samples",
                "qml_test_samples", "qml_features", "reduction_method"):
        assert key in s


# -- circuits ---------------------------------------------------------------

@pytest.mark.parametrize("name", ["ZFeatureMap", "ZZFeatureMap"])
def test_feature_map_qubits_match_features(name):
    circuit = build_feature_map(name, 4, reps=1)
    assert circuit.num_qubits == 4
    assert circuit.num_parameters == 4


def test_ansatz_is_parameterised():
    ansatz = build_ansatz("RealAmplitudes", 4, reps=2)
    assert ansatz.num_qubits == 4
    assert ansatz.num_parameters > 0


def test_circuit_stats_reports_depth_and_parameters():
    stats = circuit_stats(build_feature_map("ZZFeatureMap", 4, reps=2),
                          build_ansatz("RealAmplitudes", 4, reps=2))
    assert stats["n_qubits"] == 4
    assert stats["circuit_depth"] > 0
    assert stats["n_parameters"] > 0


def test_unknown_circuit_name_raises():
    with pytest.raises(ValueError, match="Unsupported circuit"):
        build_feature_map("NotARealFeatureMap", 4)


# -- end-to-end simulation --------------------------------------------------

@pytest.mark.slow
def test_vqc_smoke_run_produces_scores():
    """A real (tiny) variational training run on a simulator."""
    from src.qml.vqc import train_vqc

    rng = np.random.default_rng(4)

    class Tiny:
        n_features = 2
        y_train = np.array([0, 0, 0, 1, 1, 1])
        y_val = np.array([0, 1, 0, 1])
        y_test = np.array([0, 1, 0, 1])
        X_train = np.clip(rng.random((6, 2)) * np.pi, 0, np.pi)
        X_val = np.clip(rng.random((4, 2)) * np.pi, 0, np.pi)
        X_test = np.clip(rng.random((4, 2)) * np.pi, 0, np.pi)

    from src.utils.config import load_config

    cfg = load_config("qml")
    cfg["vqc"]["maxiter"] = 3            # keep the smoke test to a few seconds

    # vqc.py imported load_config by name, so patch the module-level reference.
    import src.qml.vqc as vqc_mod

    original = vqc_mod.load_config
    vqc_mod.load_config = lambda name, **kw: (
        cfg if name == "qml" else original(name, **kw))
    try:
        out = train_vqc(Tiny())
    finally:
        vqc_mod.load_config = original

    assert out["model"] == "vqc"
    assert out["n_qubits"] == 2
    assert len(out["test_score"]) == 4
    assert np.isfinite(out["test_score"]).all()
    assert out["train_seconds"] > 0


@pytest.mark.slow
@pytest.mark.artifacts
def test_persisted_qml_metadata_is_honest(project_root):
    """A model must not claim its score is a probability unless it is."""
    import json

    path = project_root / "models" / "metadata" / "qml_registry.json"
    if not path.exists():
        pytest.skip("QML branch has not been trained")
    registry = json.loads(path.read_text(encoding="utf-8"))

    for name in (registry.get("saved_models") or {}):
        meta = json.loads((project_root / "models" / "qml" / name /
                           "metadata.json").read_text(encoding="utf-8"))
        assert meta["n_qubits"] >= 1
        assert meta["subset"]["qml_features"] == meta["n_qubits"]
        # QSVM emits a signed SVM margin and QNN an expectation value; neither
        # is a probability, and the metadata must say so.
        if name in ("qsvm", "qnn"):
            assert meta["score_is_probability"] is False
        assert meta["test_metrics"]["n_samples"] > 0
