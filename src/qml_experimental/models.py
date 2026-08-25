"""QNN, VQC and QSVM arms of the experiment.

Every model here takes an explicit ``entangling_pairs`` list. The standard arm
passes a generic topology, the CLT arm passes the Chow-Liu tree edges, and
nothing else differs -- same encoding, same ansatz depth, same parameter
count, same optimiser, same iteration budget, same seed.

Readout honesty:

* **QNN** -- an ``EstimatorQNN`` expectation value in [-1, 1]. Mapped
  monotonically to [0, 1] for ranking (AUCs are unaffected) and flagged as a
  score, not a probability.
* **VQC** -- a ``SamplerQNN`` with parity readout normalised over two
  outcomes, which *is* a probability.
* **QSVM** -- a signed SVM margin. Not a probability.
"""
from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from src.qml_experimental.backends import BackendSpec
from src.qml_experimental.circuits import build_circuit, circuit_metrics
from src.utils.config import get_seed, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _optimizer(name: str, maxiter: int):
    from qiskit_machine_learning import optimizers as opt

    return getattr(opt, name.upper())(maxiter=int(maxiter))


def _circuit_cfg() -> dict[str, Any]:
    return load_config("qml_experimental")["circuits"]


# --------------------------------------------------------------------------
# QNN
# --------------------------------------------------------------------------

def initial_point(n_params: int, init_seed: int) -> np.ndarray:
    """Reproducible variational starting point.

    COBYLA is deterministic given a start, so repeating an experiment with
    different ``init_seed`` values is what produces an honest spread. Without
    it a single run's standard-vs-CLT gap cannot be distinguished from the
    luck of one initialisation.
    """
    rng = np.random.default_rng(init_seed)
    return rng.uniform(-np.pi, np.pi, size=n_params)


def train_qnn(subset, entangling_pairs: Sequence[tuple[int, int]],
              *, backend: BackendSpec, structure: str,
              maxiter: int | None = None,
              init_seed: int | None = None) -> dict[str, Any]:
    """EstimatorQNN + NeuralNetworkClassifier on +/-1 labels."""
    from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
    from qiskit_machine_learning.neural_networks import EstimatorQNN

    from src.qml_experimental.backends import build_estimator

    cfg = _circuit_cfg()
    n = subset.n_features
    built = build_circuit(n, entangling_pairs,
                          feature_map_reps=int(cfg["feature_map_reps"]),
                          ansatz_reps=int(cfg["ansatz_reps"]),
                          entangling_gate=cfg["entangling_gate"])
    metrics = circuit_metrics(built["circuit"])

    estimator, pass_manager = build_estimator(backend)
    qnn_kwargs: dict[str, Any] = {
        "circuit": built["circuit"],
        "input_params": built["input_params"],
        "weight_params": built["weight_params"],
    }
    if estimator is not None:
        qnn_kwargs["estimator"] = estimator
        qnn_kwargs["pass_manager"] = pass_manager
    if backend.precision is not None:
        # The only lever that actually controls shot noise in this stack.
        qnn_kwargs["default_precision"] = backend.precision

    qnn = EstimatorQNN(**qnn_kwargs)
    trace: list[float] = []
    model = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=_optimizer(cfg["optimizer"], maxiter or cfg["maxiter"]),
        loss="squared_error",
        initial_point=(None if init_seed is None
                       else initial_point(qnn.num_weights, init_seed)),
        callback=lambda _w, v: trace.append(float(v)),
    )

    y_pm = (2 * subset.y_train.astype(int) - 1).astype(float)
    t0 = time.perf_counter()
    model.fit(subset.X_train, y_pm)
    train_seconds = time.perf_counter() - t0

    def score(X):
        raw = np.asarray(qnn.forward(X, model.weights), dtype=float).ravel()
        return (raw + 1.0) / 2.0     # monotonic; ranking metrics unaffected

    val_score = score(subset.X_val)
    t0 = time.perf_counter()
    test_score = score(subset.X_test)
    infer_seconds = time.perf_counter() - t0

    return {
        "model": "clt_qnn" if structure == "chow_liu" else "qnn",
        "structure": structure,
        "estimator": model,
        "val_score": val_score,
        "test_score": test_score,
        "score_is_probability": False,
        "train_seconds": round(train_seconds, 3),
        "inference_ms_per_transaction": round(
            1000 * infer_seconds / max(len(subset.X_test), 1), 4),
        "n_objective_evaluations": len(trace),
        "final_objective": round(trace[-1], 6) if trace else None,
        "circuit": built,
        "circuit_metrics": metrics,
    }


# --------------------------------------------------------------------------
# VQC
# --------------------------------------------------------------------------

def train_vqc(subset, entangling_pairs: Sequence[tuple[int, int]],
              *, backend: BackendSpec, structure: str,
              maxiter: int | None = None,
              init_seed: int | None = None) -> dict[str, Any]:
    """SamplerQNN with parity readout -- a genuine two-class probability."""
    from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
    from qiskit_machine_learning.neural_networks import SamplerQNN

    from src.qml_experimental.backends import build_sampler

    cfg = _circuit_cfg()
    n = subset.n_features
    built = build_circuit(n, entangling_pairs,
                          feature_map_reps=int(cfg["feature_map_reps"]),
                          ansatz_reps=int(cfg["ansatz_reps"]),
                          entangling_gate=cfg["entangling_gate"])
    metrics = circuit_metrics(built["circuit"])

    sampler, pass_manager = build_sampler(backend)
    qnn_kwargs: dict[str, Any] = {
        "circuit": built["circuit"],
        "input_params": built["input_params"],
        "weight_params": built["weight_params"],
        "interpret": lambda bits: int(bin(bits).count("1") % 2),   # parity
        "output_shape": 2,
    }
    if sampler is not None:
        qnn_kwargs["sampler"] = sampler
        qnn_kwargs["pass_manager"] = pass_manager

    qnn = SamplerQNN(**qnn_kwargs)
    trace: list[float] = []
    model = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=_optimizer(cfg["optimizer"], maxiter or cfg["maxiter"]),
        loss="cross_entropy",
        one_hot=True,
        initial_point=(None if init_seed is None
                       else initial_point(qnn.num_weights, init_seed)),
        callback=lambda _w, v: trace.append(float(v)),
    )

    # one_hot=True expects a (n, 2) target.
    y = subset.y_train.astype(int)
    y_onehot = np.zeros((len(y), 2), dtype=float)
    y_onehot[np.arange(len(y)), y] = 1.0

    t0 = time.perf_counter()
    model.fit(subset.X_train, y_onehot)
    train_seconds = time.perf_counter() - t0

    def score(X):
        probs = np.asarray(qnn.forward(X, model.weights), dtype=float)
        return probs[:, 1] if probs.ndim == 2 else probs.ravel()

    val_score = score(subset.X_val)
    t0 = time.perf_counter()
    test_score = score(subset.X_test)
    infer_seconds = time.perf_counter() - t0

    return {
        "model": "clt_vqc" if structure == "chow_liu" else "vqc",
        "structure": structure,
        "estimator": model,
        "val_score": val_score,
        "test_score": test_score,
        # Parity readout normalised over two outcomes is a real probability.
        "score_is_probability": True,
        "train_seconds": round(train_seconds, 3),
        "inference_ms_per_transaction": round(
            1000 * infer_seconds / max(len(subset.X_test), 1), 4),
        "n_objective_evaluations": len(trace),
        "final_objective": round(trace[-1], 6) if trace else None,
        "circuit": built,
        "circuit_metrics": metrics,
    }


# --------------------------------------------------------------------------
# QSVM
# --------------------------------------------------------------------------

def train_qsvm(subset, entangling_pairs: Sequence[tuple[int, int]],
               *, backend: BackendSpec, structure: str,
               init_seed: int | None = None) -> dict[str, Any]:
    """Quantum-kernel SVM over the same encoding circuit.

    Cost is O(n^2) fidelity evaluations to fit and O(n_train) per prediction,
    which is what bounds the subset size -- not accuracy.
    """
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    from src.qml_experimental.circuits import build_feature_encoding

    cfg = _circuit_cfg()
    n = subset.n_features
    feature_map, _ = build_feature_encoding(
        n, entangling_pairs, reps=int(cfg["feature_map_reps"]),
        entangling_gate=cfg["entangling_gate"])
    metrics = circuit_metrics(feature_map)

    kernel = FidelityQuantumKernel(feature_map=feature_map)
    model = QSVC(quantum_kernel=kernel, C=1.0, random_state=get_seed())

    t0 = time.perf_counter()
    model.fit(subset.X_train, subset.y_train)
    train_seconds = time.perf_counter() - t0
    LOGGER.info("QSVM(%s, %dq) trained in %.1fs", structure, n, train_seconds)

    val_score = np.asarray(model.decision_function(subset.X_val), dtype=float).ravel()
    t0 = time.perf_counter()
    test_score = np.asarray(model.decision_function(subset.X_test), dtype=float).ravel()
    infer_seconds = time.perf_counter() - t0

    return {
        "model": "clt_qsvm" if structure == "chow_liu" else "qsvm",
        "structure": structure,
        "estimator": model,
        "val_score": val_score,
        "test_score": test_score,
        "score_is_probability": False,     # signed SVM margin
        "train_seconds": round(train_seconds, 3),
        "inference_ms_per_transaction": round(
            1000 * infer_seconds / max(len(subset.X_test), 1), 4),
        "circuit": {"feature_map": feature_map,
                    "entangling_pairs": [list(p) for p in entangling_pairs]},
        "circuit_metrics": metrics,
    }


TRAINERS = {"qnn": train_qnn, "vqc": train_vqc, "qsvm": train_qsvm}
