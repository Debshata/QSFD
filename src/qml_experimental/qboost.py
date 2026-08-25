"""QBoost: ensemble selection as a QUBO.

See ``docs/qboost_design.md`` for the derivation and for the explicit
statement of which solver is quantum and which is not.

    Q_ij = (1/N^2) sum_m h_i(x_m) h_j(x_m)                      (i != j)
    Q_ii = (1/N^2) sum_m h_i(x_m)^2 - (2/N) sum_m h_i(x_m) y_m + lambda

Two solvers:

* ``simulated_annealing`` -- CLASSICAL Metropolis heuristic. Labelled as
  classical in every result row. Solving a QUBO is not quantum computing.
* ``qaoa`` -- a genuine gate-model quantum algorithm on a **simulator**.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.config import get_seed, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class QUBO:
    """The quadratic form plus the pieces needed to interpret it."""

    Q: np.ndarray
    n_variables: int
    lambda_penalty: float

    def energy(self, w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        return float(w @ self.Q @ w)


def fit_weak_learners(X_train: np.ndarray, y_train: np.ndarray,
                      n_learners: int, seed: int) -> list[Any]:
    """Depth-1 decision stumps on bootstrapped feature subsets.

    Stumps are individually weak on purpose -- QBoost's contribution is the
    *selection*, so strong learners would mask the effect being measured.
    """
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(seed)
    n_features = X_train.shape[1]
    learners = []
    for i in range(n_learners):
        # Each stump sees a random single feature, so the pool is diverse and
        # the off-diagonal QUBO terms are meaningful.
        feature = int(rng.integers(0, n_features))
        stump = DecisionTreeClassifier(max_depth=1, random_state=seed + i)
        stump.fit(X_train[:, [feature]], y_train)
        learners.append({"model": stump, "feature": feature})
    return learners


def predict_matrix(learners: list[Any], X: np.ndarray) -> np.ndarray:
    """(n_samples, n_learners) matrix of +/-1 votes."""
    cols = []
    for entry in learners:
        pred = entry["model"].predict(X[:, [entry["feature"]]])
        cols.append(2 * np.asarray(pred, dtype=int) - 1)
    return np.column_stack(cols).astype(float)


def build_qubo(H: np.ndarray, y_pm: np.ndarray, lambda_penalty: float) -> QUBO:
    """Construct Q from the training prediction matrix."""
    M, N = H.shape
    Q = (H.T @ H) / (N ** 2)                       # correlation between learners
    linear = -(2.0 / N) * (H.T @ y_pm)             # agreement with the label
    Q = Q.copy()
    np.fill_diagonal(Q, np.diag(Q) + linear + lambda_penalty)
    LOGGER.info("QUBO built: %d variables from %d training samples, lambda=%.4f",
                N, M, lambda_penalty)
    return QUBO(Q=Q, n_variables=N, lambda_penalty=lambda_penalty)


# --------------------------------------------------------------------------
# solvers
# --------------------------------------------------------------------------

def solve_simulated_annealing(qubo: QUBO, *, reads: int = 200,
                              sweeps: int = 1000, seed: int = 42
                              ) -> dict[str, Any]:
    """CLASSICAL simulated annealing. Not a quantum solver."""
    rng = np.random.default_rng(seed)
    n = qubo.n_variables
    best_w, best_e = None, np.inf

    for _ in range(reads):
        w = rng.integers(0, 2, size=n).astype(float)
        energy = qubo.energy(w)
        for step in range(sweeps):
            temperature = max(1e-6, 1.0 * (1.0 - step / sweeps))
            i = int(rng.integers(0, n))
            # Energy delta for a single flip, without recomputing the form.
            delta = qubo.Q[i, i] * (1 - 2 * w[i])
            delta += (1 - 2 * w[i]) * 2.0 * float(
                np.dot(qubo.Q[i], w) - qubo.Q[i, i] * w[i])
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                w[i] = 1 - w[i]
                energy += delta
        energy = qubo.energy(w)
        if energy < best_e:
            best_e, best_w = energy, w.copy()

    return {"w": best_w, "energy": float(best_e), "solver": "simulated_annealing",
            "is_quantum": False,
            "solver_note": "classical Metropolis heuristic; not quantum execution"}


def solve_qaoa(qubo: QUBO, *, reps: int = 2, maxiter: int = 50,
               seed: int = 42) -> dict[str, Any]:
    """QAOA on a statevector simulator -- genuine quantum algorithm, simulated."""
    from qiskit_machine_learning.optimizers import COBYLA

    n = qubo.n_variables
    # QUBO (w in {0,1}) -> Ising (s in {-1,+1}) via w = (1 - s) / 2.
    Q = qubo.Q
    J = Q / 4.0
    np.fill_diagonal(J, 0.0)
    h = -(np.sum(Q, axis=1) + np.diag(Q)) / 4.0 + np.diag(Q) / 4.0

    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp

    terms: list[tuple[str, float]] = []
    for i in range(n):
        if abs(h[i]) > 1e-12:
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            terms.append(("".join(label), float(h[i])))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                label = ["I"] * n
                label[n - 1 - i] = "Z"
                label[n - 1 - j] = "Z"
                terms.append(("".join(label), float(2.0 * J[i, j])))
    if not terms:
        terms = [("I" * n, 0.0)]
    hamiltonian = SparsePauliOp.from_list(terms)

    beta = ParameterVector("beta", reps)
    gamma = ParameterVector("gamma", reps)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(reps):
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * gamma[layer] * h[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    qc.cx(i, j)
                    qc.rz(2 * gamma[layer] * 2.0 * J[i, j], j)
                    qc.cx(i, j)
        for i in range(n):
            qc.rx(2 * beta[layer], i)

    estimator = StatevectorEstimator()
    params = list(beta) + list(gamma)

    def objective(values):
        job = estimator.run([(qc, hamiltonian, list(values))])
        return float(job.result()[0].data.evs)

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, np.pi, size=len(params))
    result = COBYLA(maxiter=maxiter).minimize(objective, x0)

    # Read out the most probable bitstring from the optimised state.
    from qiskit.quantum_info import Statevector

    bound = qc.assign_parameters(dict(zip(params, result.x)))
    probs = Statevector(bound).probabilities_dict()
    best_bits = max(probs, key=probs.get)
    # Qiskit bitstrings are little-endian: reverse to get qubit order.
    s = np.array([1 if b == "0" else -1 for b in best_bits[::-1]], dtype=float)
    w = (1 - s) / 2.0

    return {"w": w, "energy": float(qubo.energy(w)), "solver": "qaoa",
            "is_quantum": True,
            "solver_note": "gate-model QAOA on a statevector SIMULATOR, not hardware",
            "qaoa_reps": reps, "qaoa_qubits": n,
            "final_hamiltonian_expectation": round(float(result.fun), 6),
            "n_objective_evaluations": int(getattr(result, "nfev", 0) or 0),
            "selected_bitstring_probability": round(float(probs[best_bits]), 6)}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class QBoostClassifier:
    """QUBO-selected ensemble of decision stumps."""

    def __init__(self, *, n_learners: int = 16, lambda_penalty: float = 0.05,
                 solver: str = "simulated_annealing", seed: int | None = None,
                 **solver_kwargs: Any):
        self.n_learners = n_learners
        self.lambda_penalty = lambda_penalty
        self.solver = solver
        self.seed = seed if seed is not None else get_seed()
        self.solver_kwargs = solver_kwargs

        self.learners: list[Any] = []
        self.weights: np.ndarray | None = None
        self.solver_info: dict[str, Any] = {}
        self.qubo: QUBO | None = None
        self.train_seconds: float = 0.0
        self.optimisation_seconds: float = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "QBoostClassifier":
        t0 = time.perf_counter()
        y_pm = 2 * np.asarray(y_train, dtype=int) - 1

        self.learners = fit_weak_learners(X_train, y_train, self.n_learners, self.seed)
        H = predict_matrix(self.learners, X_train)
        self.qubo = build_qubo(H, y_pm.astype(float), self.lambda_penalty)

        t_opt = time.perf_counter()
        if self.solver == "qaoa":
            self.solver_info = solve_qaoa(self.qubo, seed=self.seed,
                                          **self.solver_kwargs)
        else:
            self.solver_info = solve_simulated_annealing(
                self.qubo, seed=self.seed, **self.solver_kwargs)
        self.optimisation_seconds = time.perf_counter() - t_opt

        self.weights = np.asarray(self.solver_info["w"], dtype=float)
        if self.weights.sum() == 0:
            # An empty ensemble predicts nothing. Fall back to the single
            # learner that best agrees with the label, and record it.
            best = int(np.argmax(H.T @ y_pm))
            self.weights = np.zeros(self.n_learners)
            self.weights[best] = 1.0
            self.solver_info["fallback"] = (
                "QUBO selected an empty ensemble; kept the single "
                "best-correlated stump")
            LOGGER.warning("QBoost QUBO selected zero learners; using fallback")

        self.train_seconds = time.perf_counter() - t0
        LOGGER.info("QBoost(%s): %d/%d learners selected, energy=%.6f, %.2fs "
                    "(%.2fs optimisation)", self.solver, int(self.weights.sum()),
                    self.n_learners, self.solver_info["energy"],
                    self.train_seconds, self.optimisation_seconds)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Normalised vote fraction in [-1, 1]. A score, not a probability."""
        H = predict_matrix(self.learners, X)
        n_selected = max(self.weights.sum(), 1.0)
        return (H @ self.weights) / n_selected

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Vote fraction mapped monotonically to [0, 1].

        Ranking is unchanged, so ROC-AUC/PR-AUC are unaffected. This is NOT a
        calibrated probability and is flagged as such in results.
        """
        p = (self.decision_function(X) + 1.0) / 2.0
        return np.column_stack([1.0 - p, p])

    @property
    def ensemble_size(self) -> int:
        return int(self.weights.sum()) if self.weights is not None else 0

    def info(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "is_quantum_solver": bool(self.solver_info.get("is_quantum", False)),
            "solver_note": self.solver_info.get("solver_note"),
            "n_weak_learners": self.n_learners,
            "ensemble_size": self.ensemble_size,
            "lambda_penalty": self.lambda_penalty,
            "qubo_energy": self.solver_info.get("energy"),
            "optimisation_seconds": round(self.optimisation_seconds, 3),
            "training_seconds": round(self.train_seconds, 3),
            "fallback": self.solver_info.get("fallback"),
            **{k: v for k, v in self.solver_info.items()
               if k in ("qaoa_reps", "qaoa_qubits", "n_objective_evaluations",
                        "final_hamiltonian_expectation",
                        "selected_bitstring_probability")},
        }


def qboost_config() -> dict[str, Any]:
    return load_config("qml_experimental")["qboost"]
