"""Quantum Neural Network classifier.

An ``EstimatorQNN`` (feature map + ansatz, expectation-value readout) wrapped
in ``NeuralNetworkClassifier``. Trained against +/-1 labels with a squared
error loss, which is the standard formulation for a single-observable QNN.

The expectation value lives in [-1, 1] and is a decision score, not a
probability. It is mapped to [0, 1] monotonically for plotting only, and the
result is flagged so nothing downstream calls it a probability.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.qml.circuits import build_ansatz, build_feature_map, build_optimizer, circuit_stats
from src.utils.config import get_seed, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def train_qnn(subset) -> dict[str, Any]:
    from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
    from qiskit_machine_learning.neural_networks import EstimatorQNN

    cfg = load_config("qml")["qnn"]
    feature_map = build_feature_map(
        cfg.get("feature_map", "ZFeatureMap"), subset.n_features,
        reps=int(cfg.get("feature_map_reps", 1)),
    )
    ansatz = build_ansatz(cfg.get("ansatz", "RealAmplitudes"), subset.n_features,
                          reps=int(cfg.get("ansatz_reps", 2)))
    circuit = feature_map.compose(ansatz)
    stats = circuit_stats(feature_map, ansatz)

    qnn = EstimatorQNN(circuit=circuit,
                       input_params=list(feature_map.parameters),
                       weight_params=list(ansatz.parameters))

    objective_trace: list[float] = []

    def callback(_weights, value):
        objective_trace.append(float(value))

    optimizer = build_optimizer(cfg.get("optimizer", "COBYLA"),
                                int(cfg.get("maxiter", 60)))
    model = NeuralNetworkClassifier(neural_network=qnn, optimizer=optimizer,
                                    loss="squared_error", callback=callback)

    # EstimatorQNN emits a single expectation value, so labels must be +/-1.
    y_train_pm = (2 * subset.y_train.astype(int) - 1).astype(float)

    t0 = time.perf_counter()
    model.fit(subset.X_train, y_train_pm)
    train_seconds = time.perf_counter() - t0
    LOGGER.info("QNN trained on %d samples in %.1fs (%d objective evaluations)",
                len(subset.X_train), train_seconds, len(objective_trace))

    def raw_scores(X: np.ndarray) -> np.ndarray:
        return np.asarray(qnn.forward(X, model.weights), dtype=float).ravel()

    # Monotonic map from expectation value [-1, 1] to [0, 1]. Ranking is
    # unchanged, so ROC-AUC / PR-AUC are unaffected; this is presentation only.
    val_score = (raw_scores(subset.X_val) + 1.0) / 2.0
    t0 = time.perf_counter()
    test_raw = raw_scores(subset.X_test)
    test_seconds = time.perf_counter() - t0
    test_score = (test_raw + 1.0) / 2.0

    return {
        "model": "qnn",
        "estimator": model,
        "val_score": val_score,
        "test_score": test_score,
        "score_is_probability": False,
        "default_threshold": 0.5,
        "train_seconds": round(train_seconds, 3),
        "inference_seconds_total": round(test_seconds, 3),
        "inference_ms_per_sample": round(1000 * test_seconds / max(len(subset.X_test), 1), 3),
        "n_objective_evaluations": len(objective_trace),
        "final_objective": round(objective_trace[-1], 6) if objective_trace else None,
        "config": {"feature_map": cfg.get("feature_map"),
                   "ansatz": cfg.get("ansatz"), "ansatz_reps": cfg.get("ansatz_reps"),
                   "optimizer": cfg.get("optimizer"), "maxiter": cfg.get("maxiter"),
                   "readout": "single-observable expectation value in [-1, 1]",
                   "seed": get_seed(), "simulator": "statevector (default primitives)"},
        **stats,
    }
