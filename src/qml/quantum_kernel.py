"""Quantum-kernel SVM (QSVM).

    reduced features -> quantum feature map -> fidelity kernel -> SVC

The kernel entry K(x, x') is the fidelity between the two encoded states, so
fitting costs O(n^2) circuit evaluations and inference costs O(n_train) per
sample. That quadratic cost -- not accuracy -- is what bounds the subset size.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.qml.circuits import build_feature_map, circuit_stats
from src.utils.config import get_seed, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def train_qsvm(subset) -> dict[str, Any]:
    """Fit a quantum-kernel SVC and score the QML val/test subsets."""
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    cfg = load_config("qml")["quantum_kernel"]
    feature_map = build_feature_map(
        cfg.get("feature_map", "ZZFeatureMap"), subset.n_features,
        reps=int(cfg.get("reps", 2)), entanglement=cfg.get("entanglement", "linear"),
    )
    stats = circuit_stats(feature_map)

    kernel = FidelityQuantumKernel(feature_map=feature_map)
    # probability=True would need an internal CV refit, multiplying the kernel
    # cost. We report the decision function and label it as a score, not a
    # probability, rather than paying for a poorly-supported one.
    model = QSVC(quantum_kernel=kernel, C=float(cfg.get("svc_C", 1.0)),
                 random_state=get_seed())

    t0 = time.perf_counter()
    model.fit(subset.X_train, subset.y_train)
    train_seconds = time.perf_counter() - t0
    LOGGER.info("QSVM trained on %d samples in %.1fs", len(subset.X_train), train_seconds)

    t0 = time.perf_counter()
    val_score = np.asarray(model.decision_function(subset.X_val), dtype=float).ravel()
    val_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_score = np.asarray(model.decision_function(subset.X_test), dtype=float).ravel()
    test_seconds = time.perf_counter() - t0

    return {
        "model": "qsvm",
        "estimator": model,
        "val_score": val_score,
        "test_score": test_score,
        # An SVM decision function is a signed margin, not a probability.
        "score_is_probability": False,
        "default_threshold": 0.0,
        "train_seconds": round(train_seconds, 3),
        "val_seconds": round(val_seconds, 3),
        "inference_seconds_total": round(test_seconds, 3),
        "inference_ms_per_sample": round(1000 * test_seconds / max(len(subset.X_test), 1), 3),
        "config": {"feature_map": cfg.get("feature_map"), "reps": cfg.get("reps"),
                   "entanglement": cfg.get("entanglement"), "C": cfg.get("svc_C"),
                   "simulator": "statevector (FidelityQuantumKernel default)"},
        **stats,
    }
