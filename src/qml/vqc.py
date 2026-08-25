"""Variational Quantum Classifier.

    reduced features -> feature map -> parameterised ansatz -> measurement
                     -> parity/one-hot readout -> class

Optimiser iterations are bounded (COBYLA, maxiter from config): the point is a
legitimate variational experiment at simulator scale, not a converged model.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.qml.circuits import build_ansatz, build_feature_map, build_optimizer, circuit_stats
from src.utils.config import get_seed, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def train_vqc(subset) -> dict[str, Any]:
    from qiskit_machine_learning.algorithms import VQC

    cfg = load_config("qml")["vqc"]
    feature_map = build_feature_map(
        cfg.get("feature_map", "ZZFeatureMap"), subset.n_features,
        reps=int(cfg.get("feature_map_reps", 1)),
    )
    ansatz = build_ansatz(cfg.get("ansatz", "RealAmplitudes"), subset.n_features,
                          reps=int(cfg.get("ansatz_reps", 2)))
    optimizer = build_optimizer(cfg.get("optimizer", "COBYLA"),
                                int(cfg.get("maxiter", 60)))
    stats = circuit_stats(feature_map, ansatz)

    objective_trace: list[float] = []

    def callback(_weights, value):
        objective_trace.append(float(value))

    model = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer,
                callback=callback)

    t0 = time.perf_counter()
    model.fit(subset.X_train, subset.y_train)
    train_seconds = time.perf_counter() - t0
    LOGGER.info("VQC trained on %d samples in %.1fs (%d objective evaluations)",
                len(subset.X_train), train_seconds, len(objective_trace))

    def probabilities(X: np.ndarray) -> np.ndarray:
        """VQC uses a soft-max readout, so class 1 output is a genuine probability."""
        raw = np.asarray(model.neural_network.forward(X, model.weights), dtype=float)
        if raw.ndim == 2 and raw.shape[1] == 2:
            return raw[:, 1]
        return raw.ravel()

    val_score = probabilities(subset.X_val)
    t0 = time.perf_counter()
    test_score = probabilities(subset.X_test)
    test_seconds = time.perf_counter() - t0

    return {
        "model": "vqc",
        "estimator": model,
        "val_score": val_score,
        "test_score": test_score,
        "score_is_probability": True,
        "default_threshold": 0.5,
        "train_seconds": round(train_seconds, 3),
        "inference_seconds_total": round(test_seconds, 3),
        "inference_ms_per_sample": round(1000 * test_seconds / max(len(subset.X_test), 1), 3),
        "n_objective_evaluations": len(objective_trace),
        "final_objective": round(objective_trace[-1], 6) if objective_trace else None,
        "config": {"feature_map": cfg.get("feature_map"),
                   "feature_map_reps": cfg.get("feature_map_reps"),
                   "ansatz": cfg.get("ansatz"), "ansatz_reps": cfg.get("ansatz_reps"),
                   "optimizer": cfg.get("optimizer"), "maxiter": cfg.get("maxiter"),
                   "seed": get_seed(), "simulator": "statevector (default primitives)"},
        **stats,
    }
