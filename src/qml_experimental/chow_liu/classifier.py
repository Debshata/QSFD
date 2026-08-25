"""Class-conditional Chow-Liu fraud classifier (generative).

Two trees are fitted, one per class:

    P(X | Y = legitimate)   from legitimate training transactions
    P(X | Y = fraud)        from fraudulent training transactions

and a transaction is scored by the log-likelihood ratio

    LLR(X) = log P(X | fraud) - log P(X | legitimate)
             + log( P(fraud) / P(legitimate) )

Everything is computed in log space -- a 8-feature product of conditional
probabilities underflows float64 quickly, and the ratio of two underflowed
numbers is meaningless.

Dirichlet/Laplace smoothing is mandatory here, not optional: with 250 training
rows and 8 bins, plenty of (child, parent) cells are empty, and an unsmoothed
zero sends the log-likelihood to -inf for any transaction touching that cell.

**The LLR is not a probability.** Calibration is fitted on validation and kept
only when it improves the Brier score.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.qml_experimental.chow_liu.discretization import Discretizer, fit_discretizer
from src.qml_experimental.chow_liu.tree import ChowLiuTree, build_chow_liu_tree
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ClassConditionalCPTs:
    """Log conditional probability tables for one class's tree."""

    root: int
    parents: dict[int, int | None]
    log_root: np.ndarray                       # (card_root,)
    log_cond: dict[int, np.ndarray] = field(default_factory=dict)   # child -> (card_parent, card_child)

    def log_likelihood(self, X_disc: np.ndarray) -> np.ndarray:
        """log P(x) for each row, via the tree factorisation."""
        n = len(X_disc)
        out = self.log_root[X_disc[:, self.root]].astype(float)
        for child, table in self.log_cond.items():
            parent = self.parents[child]
            out = out + table[X_disc[:, parent], X_disc[:, child]]
        return out


def _fit_cpts(X_disc: np.ndarray, tree: ChowLiuTree, cardinalities: list[int],
              smoothing: float) -> ClassConditionalCPTs:
    """Smoothed log CPTs for one class."""
    root = tree.root
    card_root = cardinalities[root]

    counts = np.bincount(X_disc[:, root], minlength=card_root).astype(float)
    counts += smoothing
    log_root = np.log(counts / counts.sum())

    log_cond: dict[int, np.ndarray] = {}
    for child, parent in tree.parents.items():
        if parent is None:
            continue
        cp, cc = cardinalities[parent], cardinalities[child]
        joint = np.zeros((cp, cc), dtype=float)
        np.add.at(joint, (X_disc[:, parent], X_disc[:, child]), 1.0)
        joint += smoothing
        # Normalise each parent row: P(child | parent).
        log_cond[child] = np.log(joint / joint.sum(axis=1, keepdims=True))
    return ClassConditionalCPTs(root=root, parents=tree.parents,
                                log_root=log_root, log_cond=log_cond)


class ChowLiuClassifier:
    """Generative fraud classifier built from two class-conditional CLTs."""

    def __init__(self, *, n_bins: int = 8, strategy: str = "quantile",
                 smoothing: float = 1.0, estimator: str = "plug_in",
                 n_neighbors: int = 3, seed: int = 0):
        self.n_bins = n_bins
        self.strategy = strategy
        self.smoothing = smoothing
        self.estimator = estimator
        self.n_neighbors = n_neighbors
        self.seed = seed

        self.discretizer: Discretizer | None = None
        self.tree_fraud: ChowLiuTree | None = None
        self.tree_legit: ChowLiuTree | None = None
        self.cpts_fraud: ClassConditionalCPTs | None = None
        self.cpts_legit: ClassConditionalCPTs | None = None
        self.log_prior_ratio: float = 0.0
        self.feature_names: list[str] = []
        self.calibrator = None
        self.calibration_info: dict[str, Any] = {"applied": False}
        self.fit_seconds: float = 0.0

    # -- fitting ----------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: list[str] | None = None) -> "ChowLiuClassifier":
        t0 = time.perf_counter()
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train).astype(int)
        n_features = X_train.shape[1]
        self.feature_names = list(feature_names or
                                  [f"f{i}" for i in range(n_features)])

        # Discretiser is fitted on ALL training rows (both classes) so the two
        # class-conditional tables share a bin grid and their likelihoods are
        # comparable. It never sees validation or test data.
        self.discretizer = fit_discretizer(X_train, strategy=self.strategy,
                                           n_bins=self.n_bins)
        cards = self.discretizer.cardinalities
        X_disc = self.discretizer.transform(X_train)

        fraud_mask = y_train == 1
        legit_mask = ~fraud_mask
        if fraud_mask.sum() < 2 or legit_mask.sum() < 2:
            raise ValueError("Both classes need at least 2 training rows")

        self.tree_fraud = build_chow_liu_tree(
            X_disc[fraud_mask], self.feature_names, cardinalities=cards,
            estimator=self.estimator,
            X_continuous=X_train[fraud_mask], n_neighbors=self.n_neighbors,
            seed=self.seed)
        self.tree_legit = build_chow_liu_tree(
            X_disc[legit_mask], self.feature_names, cardinalities=cards,
            estimator=self.estimator,
            X_continuous=X_train[legit_mask], n_neighbors=self.n_neighbors,
            seed=self.seed)

        self.cpts_fraud = _fit_cpts(X_disc[fraud_mask], self.tree_fraud, cards,
                                    self.smoothing)
        self.cpts_legit = _fit_cpts(X_disc[legit_mask], self.tree_legit, cards,
                                    self.smoothing)

        p_fraud = float(fraud_mask.mean())
        self.log_prior_ratio = float(np.log(p_fraud / (1.0 - p_fraud)))
        self.fit_seconds = time.perf_counter() - t0

        LOGGER.info("ChowLiuClassifier fitted: %d features, prior ratio %.4f, "
                    "%.2fs", n_features, self.log_prior_ratio, self.fit_seconds)
        return self

    # -- scoring ----------------------------------------------------------

    def log_likelihood_ratio(self, X: np.ndarray) -> np.ndarray:
        """Raw LLR. Higher means more fraud-like. NOT a probability."""
        if self.discretizer is None:
            raise RuntimeError("ChowLiuClassifier is not fitted")
        X_disc = self.discretizer.transform(np.asarray(X, dtype=float))
        return (self.cpts_fraud.log_likelihood(X_disc)
                - self.cpts_legit.log_likelihood(X_disc)
                + self.log_prior_ratio)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.log_likelihood_ratio(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability estimates.

        With calibration applied this is the calibrated probability. Without
        it, the logistic transform of the LLR -- which is the *model-implied*
        posterior under the tree assumption, and is reported as such rather
        than as a validated probability.
        """
        llr = self.log_likelihood_ratio(X)
        if self.calibrator is not None:
            p = self.calibrator.predict_proba(llr.reshape(-1, 1))[:, 1]
        else:
            # Numerically stable logistic.
            p = np.where(llr >= 0, 1.0 / (1.0 + np.exp(-np.clip(llr, -700, 700))),
                         np.exp(np.clip(llr, -700, 700))
                         / (1.0 + np.exp(np.clip(llr, -700, 700))))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        return (self.log_likelihood_ratio(X) >= threshold).astype(int)

    # -- calibration ------------------------------------------------------

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray,
                  method: str = "sigmoid") -> dict[str, Any]:
        """Fit a calibrator on VALIDATION data; keep it only if it helps."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss

        y_val = np.asarray(y_val).astype(int)
        before = float(brier_score_loss(y_val, self.predict_proba(X_val)[:, 1]))

        info: dict[str, Any] = {"attempted": True, "method": method,
                                "brier_before": round(before, 6), "applied": False}
        try:
            llr = self.log_likelihood_ratio(X_val).reshape(-1, 1)
            if method == "sigmoid":
                calibrator = LogisticRegression(max_iter=1000)
            elif method == "isotonic":
                from sklearn.isotonic import IsotonicRegression

                class _IsoWrap:
                    def __init__(self):
                        self.iso = IsotonicRegression(out_of_bounds="clip")

                    def fit(self, X, y):
                        self.iso.fit(X.ravel(), y)
                        return self

                    def predict_proba(self, X):
                        p = np.clip(self.iso.predict(X.ravel()), 1e-6, 1 - 1e-6)
                        return np.column_stack([1 - p, p])

                calibrator = _IsoWrap()
            else:
                raise ValueError(f"Unknown calibration method {method!r}")

            calibrator.fit(llr, y_val)
            candidate = calibrator.predict_proba(llr)[:, 1]
            after = float(brier_score_loss(y_val, candidate))
            info["brier_after"] = round(after, 6)

            if after < before:
                self.calibrator = calibrator
                info["applied"] = True
                LOGGER.info("CLT calibration (%s) applied: Brier %.6f -> %.6f",
                            method, before, after)
            else:
                LOGGER.info("CLT calibration (%s) rejected: Brier %.6f -> %.6f",
                            method, before, after)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("CLT calibration failed: %r", exc)
            info["error"] = repr(exc)

        self.calibration_info = info
        return info

    # -- introspection ----------------------------------------------------

    def structure(self) -> dict[str, Any]:
        return {
            "fraud_tree": self.tree_fraud.to_dict() if self.tree_fraud else None,
            "legitimate_tree": self.tree_legit.to_dict() if self.tree_legit else None,
            "discretization": self.discretizer.to_dict() if self.discretizer else None,
            "smoothing": self.smoothing,
            "mi_estimator": self.estimator,
            "log_prior_ratio": round(self.log_prior_ratio, 6),
            "calibration": self.calibration_info,
        }
