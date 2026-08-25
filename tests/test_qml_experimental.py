"""Tests for the QML experimental research layer.

Expensive training is never run here -- the quantum tests use tiny circuits and
handfuls of samples. The point is to verify the *machinery* (mutual
information, spanning trees, leakage prevention, checkpointing, readiness
scoring), not to reproduce the experiments.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.qml_experimental.chow_liu.discretization import fit_discretizer
from src.qml_experimental.chow_liu.mutual_information import (
    mutual_information_matrix,
    plug_in_mutual_information,
)
from src.qml_experimental.chow_liu.tree import (
    ChowLiuTree,
    build_chow_liu_tree,
    maximum_weight_spanning_tree,
)

qiskit = pytest.importorskip("qiskit", reason="Qiskit not installed")
pytest.importorskip("qiskit_machine_learning")

from src.qml_experimental.chow_liu.feature_structure import (  # noqa: E402
    learn_feature_structure,
)
from src.qml_experimental.chow_liu.quantum_mapping import (  # noqa: E402
    QubitMapping,
    baseline_entangling_pairs,
    build_qubit_mapping,
)
from src.qml_experimental.circuits import (  # noqa: E402
    build_circuit,
    circuit_metrics,
    entangling_pairs_for,
)


@pytest.fixture
def dependent_data():
    """Features with a KNOWN dependency chain: f0 -> f1 -> f2, f3 independent."""
    rng = np.random.default_rng(11)
    n = 800
    f0 = rng.normal(0, 1, n)
    f1 = f0 + rng.normal(0, 0.3, n)         # strongly depends on f0
    f2 = f1 + rng.normal(0, 0.3, n)         # strongly depends on f1
    f3 = rng.normal(0, 1, n)                # independent of everything
    X = np.column_stack([f0, f1, f2, f3])
    y = (f0 + f2 + rng.normal(0, 0.5, n) > 1.0).astype(int)
    return X, y, ["f0", "f1", "f2", "f3"]


# --------------------------------------------------------------------------
# discretisation
# --------------------------------------------------------------------------

def test_discretizer_fitted_on_train_only(dependent_data):
    """Test rows must be binned with TRAINING edges, not their own quantiles."""
    X, _, _ = dependent_data
    train, test = X[:600], X[600:]
    disc = fit_discretizer(train, strategy="quantile", n_bins=8)

    shifted = test + 100.0                  # far outside the training range
    out = disc.transform(shifted)
    assert out.shape == shifted.shape
    # Everything lands in the top bin because the edges came from training.
    assert out.max() <= max(disc.cardinalities) - 1
    assert (out == max(disc.cardinalities) - 1).all()


def test_discretizer_bin_indices_in_range(dependent_data):
    X, _, _ = dependent_data
    disc = fit_discretizer(X, n_bins=8)
    out = disc.transform(X)
    assert out.min() >= 0
    for j, card in enumerate(disc.cardinalities):
        assert out[:, j].max() < card


@pytest.mark.parametrize("strategy", ["quantile", "uniform"])
def test_discretization_strategies(strategy, dependent_data):
    X, _, _ = dependent_data
    disc = fit_discretizer(X, strategy=strategy, n_bins=6)
    assert disc.strategy == strategy
    assert np.isfinite(disc.transform(X)).all()


# --------------------------------------------------------------------------
# mutual information
# --------------------------------------------------------------------------

def test_mi_of_independent_variables_is_near_zero():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 5, 4000)
    b = rng.integers(0, 5, 4000)
    assert plug_in_mutual_information(a, b, 5, 5) < 0.02


def test_mi_of_identical_variables_equals_entropy():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 4, 4000)
    entropy = -sum(p * np.log(p) for p in np.bincount(a) / len(a) if p > 0)
    assert plug_in_mutual_information(a, a, 4, 4) == pytest.approx(entropy, abs=0.02)


def test_mi_is_symmetric_and_non_negative():
    rng = np.random.default_rng(5)
    a = rng.integers(0, 4, 1000)
    b = (a + rng.integers(0, 2, 1000)) % 4
    ab = plug_in_mutual_information(a, b, 4, 4)
    ba = plug_in_mutual_information(b, a, 4, 4)
    assert ab == pytest.approx(ba, abs=1e-9)
    assert ab >= 0


def test_mi_matrix_shape_symmetry_and_zero_diagonal(dependent_data):
    X, _, _ = dependent_data
    disc = fit_discretizer(X, n_bins=8)
    mi = mutual_information_matrix(disc.transform(X), disc.cardinalities)
    assert mi.shape == (4, 4)
    np.testing.assert_allclose(mi, mi.T)
    np.testing.assert_allclose(np.diag(mi), 0.0)
    assert (mi >= 0).all()


def test_mi_detects_the_known_dependency(dependent_data):
    """f0-f1 must score far above f0-f3, which is independent by construction."""
    X, _, _ = dependent_data
    disc = fit_discretizer(X, n_bins=8)
    mi = mutual_information_matrix(disc.transform(X), disc.cardinalities)
    assert mi[0, 1] > mi[0, 3] * 3
    assert mi[1, 2] > mi[1, 3] * 3


# --------------------------------------------------------------------------
# spanning tree
# --------------------------------------------------------------------------

def test_spanning_tree_has_n_minus_one_edges():
    rng = np.random.default_rng(0)
    for n in (2, 4, 6, 8, 10):
        w = rng.random((n, n))
        w = (w + w.T) / 2
        np.fill_diagonal(w, 0)
        edges, weights = maximum_weight_spanning_tree(w)
        assert len(edges) == n - 1
        assert len(weights) == n - 1


def test_spanning_tree_picks_maximum_weight():
    # A clear optimum: 0-1 and 1-2 are heavy, everything else is light.
    w = np.array([[0.0, 9.0, 0.1, 0.1],
                  [9.0, 0.0, 8.0, 0.1],
                  [0.1, 8.0, 0.0, 0.2],
                  [0.1, 0.1, 0.2, 0.0]])
    edges, weights = maximum_weight_spanning_tree(w)
    assert (0, 1) in edges and (1, 2) in edges
    assert sum(weights) == pytest.approx(9.0 + 8.0 + 0.2)


def test_spanning_tree_is_connected_and_acyclic():
    rng = np.random.default_rng(7)
    n = 8
    w = rng.random((n, n))
    w = (w + w.T) / 2
    np.fill_diagonal(w, 0)
    edges, _ = maximum_weight_spanning_tree(w)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        assert ra != rb, "a spanning tree must not contain a cycle"
        parent[ra] = rb
    assert len({find(i) for i in range(n)}) == 1, "tree must be connected"


def test_spanning_tree_is_deterministic():
    rng = np.random.default_rng(1)
    w = rng.random((6, 6))
    w = (w + w.T) / 2
    np.fill_diagonal(w, 0)
    assert maximum_weight_spanning_tree(w)[0] == maximum_weight_spanning_tree(w)[0]


def test_tree_rejects_wrong_edge_count():
    with pytest.raises(ValueError, match="must have exactly"):
        ChowLiuTree(n_nodes=4, edges=[(0, 1)], weights=[1.0],
                    feature_names=list("abcd"))


def test_tree_recovers_the_known_chain(dependent_data):
    """The CLT should find f0-f1-f2 and attach f3 somewhere weakly."""
    X, _, names = dependent_data
    disc = fit_discretizer(X, n_bins=8)
    tree = build_chow_liu_tree(disc.transform(X), names,
                               cardinalities=disc.cardinalities)
    edge_set = {tuple(sorted(e)) for e in tree.edges}
    assert (0, 1) in edge_set
    assert (1, 2) in edge_set
    assert tree.n_edges == 3


def test_tree_orientation_gives_every_node_a_parent(dependent_data):
    X, _, names = dependent_data
    disc = fit_discretizer(X, n_bins=6)
    tree = build_chow_liu_tree(disc.transform(X), names,
                               cardinalities=disc.cardinalities)
    assert tree.parents[tree.root] is None
    assert len(tree.parents) == tree.n_nodes
    for node, parent in tree.parents.items():
        if node != tree.root:
            assert parent is not None


# --------------------------------------------------------------------------
# class-conditional classifier
# --------------------------------------------------------------------------

def test_clt_classifier_learns_and_scores(dependent_data):
    from src.qml_experimental.chow_liu.classifier import ChowLiuClassifier

    X, y, names = dependent_data
    clf = ChowLiuClassifier(n_bins=6).fit(X[:600], y[:600], names)
    llr = clf.log_likelihood_ratio(X[600:])
    assert llr.shape == (200,)
    assert np.isfinite(llr).all(), "smoothing must prevent -inf log-likelihoods"


def test_clt_classifier_beats_chance(dependent_data):
    from sklearn.metrics import roc_auc_score

    from src.qml_experimental.chow_liu.classifier import ChowLiuClassifier

    X, y, names = dependent_data
    clf = ChowLiuClassifier(n_bins=6).fit(X[:600], y[:600], names)
    auc = roc_auc_score(y[600:], clf.log_likelihood_ratio(X[600:]))
    assert auc > 0.7


def test_clt_probabilities_are_in_range(dependent_data):
    from src.qml_experimental.chow_liu.classifier import ChowLiuClassifier

    X, y, names = dependent_data
    clf = ChowLiuClassifier(n_bins=6).fit(X[:600], y[:600], names)
    proba = clf.predict_proba(X[600:])
    assert proba.shape == (200, 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_smoothing_prevents_zero_frequency_failure():
    """An unseen bin combination must not send the log-likelihood to -inf."""
    from src.qml_experimental.chow_liu.classifier import ChowLiuClassifier

    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (120, 4))
    y = (rng.random(120) < 0.3).astype(int)
    clf = ChowLiuClassifier(n_bins=8, smoothing=1.0).fit(X, y)
    extreme = np.full((5, 4), 99.0)          # every value in an unseen top bin
    assert np.isfinite(clf.log_likelihood_ratio(extreme)).all()


def test_clt_classifier_requires_both_classes():
    from src.qml_experimental.chow_liu.classifier import ChowLiuClassifier

    X = np.random.default_rng(0).normal(0, 1, (50, 3))
    with pytest.raises(ValueError, match="Both classes"):
        ChowLiuClassifier().fit(X, np.zeros(50, dtype=int))


def test_clt_structure_is_serialisable(dependent_data):
    from src.qml_experimental.chow_liu.classifier import ChowLiuClassifier

    X, y, names = dependent_data
    clf = ChowLiuClassifier(n_bins=6).fit(X[:400], y[:400], names)
    assert json.loads(json.dumps(clf.structure(), default=str))


# --------------------------------------------------------------------------
# qubit mapping
# --------------------------------------------------------------------------

def test_qubit_mapping_is_a_bijection(dependent_data):
    X, _, names = dependent_data
    structure = learn_feature_structure(X, names, n_bins=6)
    mapping = build_qubit_mapping(structure, "cz")
    assert mapping.n_qubits == 4
    assert sorted(mapping.feature_to_qubit.values()) == [0, 1, 2, 3]
    assert len(mapping.entangling_pairs) == 3


def test_qubit_mapping_rejects_a_bad_assignment():
    bad = QubitMapping(feature_to_qubit={"a": 0, "b": 0},
                       qubit_to_feature={0: "a", 1: "b"},
                       entangling_pairs=[(0, 1)], edge_weights=[1.0],
                       entangling_gate="cz", n_qubits=2)
    with pytest.raises(ValueError, match="not a bijection"):
        bad.validate()


def test_qubit_mapping_rejects_self_entanglement():
    bad = QubitMapping(feature_to_qubit={"a": 0, "b": 1},
                       qubit_to_feature={0: "a", 1: "b"},
                       entangling_pairs=[(1, 1)], edge_weights=[1.0],
                       entangling_gate="cz", n_qubits=2)
    with pytest.raises(ValueError, match="Self-entangling"):
        bad.validate()


def test_qubit_mapping_rejects_unknown_gate():
    bad = QubitMapping(feature_to_qubit={"a": 0, "b": 1},
                       qubit_to_feature={0: "a", 1: "b"},
                       entangling_pairs=[(0, 1)], edge_weights=[1.0],
                       entangling_gate="toffoli", n_qubits=2)
    with pytest.raises(ValueError, match="Unsupported entangling gate"):
        bad.validate()


def test_mapping_is_stable_across_rebuilds(dependent_data):
    """A silently changed mapping would leave a trained circuit misread."""
    X, _, names = dependent_data
    a = build_qubit_mapping(learn_feature_structure(X, names, n_bins=6), "cz")
    b = build_qubit_mapping(learn_feature_structure(X, names, n_bins=6), "cz")
    assert a.feature_to_qubit == b.feature_to_qubit
    assert a.entangling_pairs == b.entangling_pairs


@pytest.mark.parametrize("pattern,expected", [
    ("linear", 3), ("circular", 4), ("full", 6), ("none", 0),
])
def test_baseline_topology_pair_counts(pattern, expected):
    assert len(baseline_entangling_pairs(4, pattern)) == expected


def test_linear_control_has_the_same_pair_count_as_a_tree():
    """The head-to-head control must not differ in gate budget."""
    for n in (4, 6, 8):
        assert len(baseline_entangling_pairs(n, "linear")) == n - 1


# --------------------------------------------------------------------------
# circuits
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_qubits", [4, 6, 8])
def test_circuit_builds_at_every_configured_size(n_qubits):
    pairs = entangling_pairs_for("linear", n_qubits)
    built = build_circuit(n_qubits, pairs)
    assert built["circuit"].num_qubits == n_qubits
    assert len(built["input_params"]) == n_qubits
    metrics = circuit_metrics(built["circuit"])
    assert metrics["logical_depth"] > 0
    assert metrics["logical_two_qubit_gates"] > 0


def test_structured_and_standard_have_equal_parameter_counts():
    """Capacity must be held constant so the ablation isolates topology."""
    for n in (4, 6, 8):
        linear = build_circuit(n, entangling_pairs_for("linear", n))
        tree = build_circuit(n, [(0, i) for i in range(1, n)])   # a star tree
        assert (circuit_metrics(linear["circuit"])["trainable_parameters"]
                == circuit_metrics(tree["circuit"])["trainable_parameters"])


def test_more_entangling_pairs_means_more_two_qubit_gates():
    linear = circuit_metrics(build_circuit(6, entangling_pairs_for("linear", 6))["circuit"])
    full = circuit_metrics(build_circuit(6, entangling_pairs_for("full", 6))["circuit"])
    assert full["logical_two_qubit_gates"] > linear["logical_two_qubit_gates"]
    assert full["logical_depth"] > linear["logical_depth"]


def test_circuit_metrics_reports_every_required_field():
    metrics = circuit_metrics(build_circuit(4, entangling_pairs_for("linear", 4))["circuit"])
    for field in ("n_qubits", "logical_depth", "logical_total_gates",
                  "logical_single_qubit_gates", "logical_two_qubit_gates",
                  "trainable_parameters"):
        assert metrics[field] is not None


# --------------------------------------------------------------------------
# transpilation
# --------------------------------------------------------------------------

def test_transpilation_reports_measured_metrics():
    from src.qml_experimental.transpilation import transpile_metrics

    built = build_circuit(4, entangling_pairs_for("linear", 4))
    out = transpile_metrics(built["circuit"], "line")
    assert out["status"] == "ok"
    assert out["transpiled_depth"] > 0
    assert out["transpiled_two_qubit_gates"] >= 0
    assert out["swap_count"] >= 0


def test_constrained_coupling_costs_at_least_as_much_as_full():
    """Routing on a line can only add work relative to all-to-all."""
    from src.qml_experimental.transpilation import transpile_metrics

    built = build_circuit(6, entangling_pairs_for("full", 6))
    full = transpile_metrics(built["circuit"], "full")
    line = transpile_metrics(built["circuit"], "line")
    assert line["transpiled_two_qubit_gates"] >= full["transpiled_two_qubit_gates"]


@pytest.mark.parametrize("coupling", ["full", "line", "ring", "heavy_hex_like"])
def test_all_coupling_maps_are_buildable(coupling):
    from src.qml_experimental.transpilation import build_coupling_map

    result = build_coupling_map(coupling, 6)
    assert result is None or len(list(result.get_edges())) > 0


# --------------------------------------------------------------------------
# backends / noise
# --------------------------------------------------------------------------

def test_backend_precision_matches_shot_count():
    from src.qml_experimental.backends import BackendSpec

    assert BackendSpec(name="ideal").precision is None
    assert BackendSpec(name="s", shots=1024).precision == pytest.approx(1 / 32)
    assert BackendSpec(name="s", shots=256).precision == pytest.approx(1 / 16)


def test_noise_levels_build_from_config():
    from src.qml_experimental.backends import build_noise_model

    assert build_noise_model("none") is None
    for level in ("low", "moderate"):
        assert build_noise_model(level) is not None


def test_unknown_noise_level_raises():
    from src.qml_experimental.backends import build_noise_model

    with pytest.raises(ValueError, match="Unknown noise level"):
        build_noise_model("catastrophic")


def test_ideal_spec_is_ideal():
    from src.qml_experimental.backends import ideal_spec

    spec = ideal_spec()
    assert spec.is_ideal
    assert spec.describe()["simulator"] == "ideal"


# --------------------------------------------------------------------------
# QBoost
# --------------------------------------------------------------------------

def test_qubo_is_square_and_symmetric_off_diagonal():
    from src.qml_experimental.qboost import build_qubo

    rng = np.random.default_rng(4)
    H = rng.choice([-1.0, 1.0], size=(60, 6))
    y = rng.choice([-1.0, 1.0], size=60)
    qubo = build_qubo(H, y, 0.05)
    assert qubo.Q.shape == (6, 6)
    off = qubo.Q - np.diag(np.diag(qubo.Q))
    np.testing.assert_allclose(off, off.T)


def test_qboost_selects_a_non_empty_ensemble():
    from src.qml_experimental.qboost import QBoostClassifier

    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, (200, 5))
    y = (X[:, 0] + X[:, 1] > 0.5).astype(int)
    model = QBoostClassifier(n_learners=6, reads=20, sweeps=100).fit(X, y)
    assert model.ensemble_size >= 1
    assert model.info()["is_quantum_solver"] is False


def test_qboost_sa_is_labelled_classical():
    """A QUBO solved classically must never be reported as quantum."""
    from src.qml_experimental.qboost import QUBO, solve_simulated_annealing

    rng = np.random.default_rng(0)
    Q = rng.normal(0, 1, (5, 5))
    Q = (Q + Q.T) / 2
    out = solve_simulated_annealing(QUBO(Q=Q, n_variables=5, lambda_penalty=0.0),
                                    reads=5, sweeps=50)
    assert out["is_quantum"] is False
    assert "not quantum" in out["solver_note"].lower()


def test_qboost_scores_are_in_range():
    from src.qml_experimental.qboost import QBoostClassifier

    rng = np.random.default_rng(8)
    X = rng.normal(0, 1, (150, 4))
    y = (X[:, 0] > 0).astype(int)
    model = QBoostClassifier(n_learners=5, reads=10, sweeps=50).fit(X, y)
    proba = model.predict_proba(X)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.abs(model.decision_function(X)).max() <= 1.0 + 1e-9


@pytest.mark.slow
def test_qaoa_solver_runs_on_a_tiny_qubo():
    from src.qml_experimental.qboost import QUBO, solve_qaoa

    rng = np.random.default_rng(1)
    Q = rng.normal(0, 1, (4, 4))
    Q = (Q + Q.T) / 2
    out = solve_qaoa(QUBO(Q=Q, n_variables=4, lambda_penalty=0.0),
                     reps=1, maxiter=5)
    assert out["is_quantum"] is True
    assert "simulator" in out["solver_note"].lower()
    assert len(out["w"]) == 4
    assert set(np.unique(out["w"])).issubset({0.0, 1.0})


# --------------------------------------------------------------------------
# fusion leakage
# --------------------------------------------------------------------------

def test_fusion_is_fitted_on_validation_only():
    """Changing test scores must not change the fitted fusion model."""
    from src.qml_experimental.hybrid import build_hybrid

    rng = np.random.default_rng(9)
    y_val = (rng.random(80) < 0.3).astype(int)
    comp_a = {"val": rng.random(80) + y_val * 0.5, "test": rng.random(60)}
    comp_b = {"val": rng.random(80) + y_val * 0.3, "test": rng.random(60)}

    first = build_hybrid({"a": comp_a, "b": comp_b}, y_val)
    comp_a2 = {**comp_a, "test": rng.random(60) * 10}
    second = build_hybrid({"a": comp_a2, "b": comp_b}, y_val)
    assert first["fusion_info"]["weights"] == second["fusion_info"]["weights"]


def test_fusion_output_is_bounded():
    from src.qml_experimental.hybrid import build_hybrid

    rng = np.random.default_rng(10)
    y_val = (rng.random(60) < 0.4).astype(int)
    out = build_hybrid(
        {"a": {"val": rng.random(60), "test": rng.random(40)},
         "b": {"val": rng.random(60), "test": rng.random(40)}}, y_val)
    assert (out["test_score"] >= 0).all() and (out["test_score"] <= 1).all()


def test_weighted_fusion_ignores_a_useless_component():
    from src.qml_experimental.hybrid import fit_fusion

    rng = np.random.default_rng(12)
    y = (rng.random(200) < 0.4).astype(int)
    good = y + rng.normal(0, 0.1, 200)
    noise = rng.random(200)
    fusion = fit_fusion(np.column_stack([good, noise]), y, ["good", "noise"],
                        method="weighted")
    assert fusion.weights[0] > fusion.weights[1]


# --------------------------------------------------------------------------
# checkpointing / timeout / schema
# --------------------------------------------------------------------------

def test_checkpoint_round_trip(tmp_path, monkeypatch):
    import src.qml_experimental.metrics as m

    monkeypatch.setattr(m, "raw_dir", lambda: tmp_path)
    result = m.ExperimentResult(experiment="unit_test", model="dummy", qubits=4,
                                pr_auc=0.5, status=m.STATUS_OK)
    m.save_result(result, "unit_test")
    loaded = m.load_result("unit_test")
    assert loaded["experiment"] == "unit_test"
    assert loaded["pr_auc"] == 0.5
    assert m.is_complete("unit_test")


def test_unmeasured_fields_stay_null():
    """A measurement that was not taken must be null, never zero."""
    from src.qml_experimental.metrics import ExperimentResult

    result = ExperimentResult(experiment="x", model="y")
    payload = result.to_dict()
    for field in ("pr_auc", "transpiled_depth", "swap_count", "shots",
                  "logical_depth", "training_seconds"):
        assert payload[field] is None


def test_non_finite_metrics_become_null():
    from src.qml_experimental.metrics import ExperimentResult, attach_metrics

    result = ExperimentResult(experiment="x", model="y")
    # A single-class fold makes ROC-AUC undefined.
    attach_metrics(result, np.zeros(10, dtype=int), np.random.default_rng(0).random(10),
                   threshold=0.5, score_is_probability=True)
    assert result.roc_auc is None


def test_run_guarded_isolates_failure(tmp_path, monkeypatch):
    import src.qml_experimental.metrics as m

    monkeypatch.setattr(m, "raw_dir", lambda: tmp_path)

    def boom():
        raise RuntimeError("simulated failure")

    result = m.run_guarded("failing", boom, resume=False)
    assert result.status == m.STATUS_FAILED
    assert "simulated failure" in result.error
    assert (tmp_path / "failing.json").exists()


def test_run_guarded_marks_over_budget_as_timeout(tmp_path, monkeypatch):
    import time as _time

    import src.qml_experimental.metrics as m

    monkeypatch.setattr(m, "raw_dir", lambda: tmp_path)
    monkeypatch.setattr(m, "timeout_for", lambda kind: 0.01)

    def slow():
        _time.sleep(0.05)
        return m.ExperimentResult(experiment="slow", model="x", status=m.STATUS_OK)

    result = m.run_guarded("slow", slow, resume=False)
    assert result.status == m.STATUS_TIMEOUT
    assert result.total_seconds is not None      # measurements are retained


def test_run_guarded_resumes_from_checkpoint(tmp_path, monkeypatch):
    import src.qml_experimental.metrics as m

    monkeypatch.setattr(m, "raw_dir", lambda: tmp_path)
    calls = {"n": 0}

    def work():
        calls["n"] += 1
        return m.ExperimentResult(experiment="once", model="x", pr_auc=0.9,
                                  status=m.STATUS_OK)

    m.run_guarded("once", work, resume=True)
    m.run_guarded("once", work, resume=True)
    assert calls["n"] == 1, "a completed experiment must not re-run under --resume"


# --------------------------------------------------------------------------
# hardware readiness
# --------------------------------------------------------------------------

def test_readiness_is_bounded_and_ranked():
    from src.qml_experimental.hardware_readiness import rank_candidates

    ranked = rank_candidates([
        {"model": "good", "qubits": 4, "pr_auc": 0.95, "ideal_f1": 0.9,
         "noisy_f1": 0.88, "transpiled_depth": 40,
         "transpiled_two_qubit_gates": 10, "swap_count": 0, "shot_std": 0.01,
         "runtime_seconds": 20},
        {"model": "bad", "qubits": 8, "pr_auc": 0.55, "ideal_f1": 0.5,
         "noisy_f1": 0.2, "transpiled_depth": 390,
         "transpiled_two_qubit_gates": 118, "swap_count": 58, "shot_std": 0.14,
         "runtime_seconds": 590},
    ])
    assert ranked[0]["model"] == "good"
    assert 0.0 <= ranked[0]["readiness_score"] <= 1.0
    assert 0.0 <= ranked[1]["readiness_score"] <= 1.0
    assert ranked[0]["readiness_score"] > ranked[1]["readiness_score"]


def test_readiness_records_unmeasured_axes():
    from src.qml_experimental.hardware_readiness import compute_readiness

    out = compute_readiness({"model": "partial", "qubits": 4, "pr_auc": 0.8})
    assert "noise_robustness" in out["missing_axes"]
    assert out["axes"]["noise_robustness"] is None
    assert out["max_possible_with_available_axes"] < 1.0


def test_readiness_never_credits_a_missing_measurement():
    from src.qml_experimental.hardware_readiness import compute_readiness

    full = compute_readiness({"model": "m", "pr_auc": 0.8, "ideal_f1": 1.0,
                              "noisy_f1": 1.0, "transpiled_depth": 0,
                              "transpiled_two_qubit_gates": 0, "swap_count": 0,
                              "shot_std": 0.0, "runtime_seconds": 0})
    partial = compute_readiness({"model": "m", "pr_auc": 0.8})
    assert full["readiness_score"] > partial["readiness_score"]


def test_readiness_weights_sum_to_one():
    from src.utils.config import load_config

    weights = load_config("qml_experimental")["hardware_readiness"]["weights"]
    assert sum(weights.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# production isolation
# --------------------------------------------------------------------------

def test_experimental_layer_does_not_alter_production_config():
    """The research layer must not change the shipped production model."""
    from src.utils.config import load_config

    assert load_config("classical")["models"]["xgboost"]["enabled"] is True
    assert load_config("qml")["subset"]["train_samples"] == 250


def test_production_registry_still_names_xgboost(project_root):
    registry = project_root / "models" / "metadata" / "classical_registry.json"
    if not registry.exists():
        pytest.skip("production model not trained")
    assert json.loads(registry.read_text(encoding="utf-8"))["production_model"] \
        == "xgboost"
