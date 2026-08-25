"""Pairwise mutual information I(Xi ; Xj).

This is genuine mutual information, not a correlation stand-in. Correlation
only sees monotone linear structure; MI captures any statistical dependence,
which is the whole reason Chow-Liu uses it.

Two estimators:

* ``plug_in``  -- discrete, from the joint contingency table (default).
* ``ksg``      -- Kraskov-Stoegbauer-Grassberger k-NN estimator for continuous
                  variables, via scikit-learn's ``mutual_info_regression``.
"""
from __future__ import annotations

import numpy as np

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def joint_histogram(xi: np.ndarray, xj: np.ndarray,
                    n_i: int, n_j: int) -> np.ndarray:
    """Contingency table of two integer-coded variables."""
    table = np.zeros((n_i, n_j), dtype=np.float64)
    np.add.at(table, (xi, xj), 1.0)
    return table


def plug_in_mutual_information(xi: np.ndarray, xj: np.ndarray,
                               n_i: int | None = None,
                               n_j: int | None = None) -> float:
    """Plug-in (maximum-likelihood) MI in nats for two discrete variables.

        I(X;Y) = sum_xy p(x,y) log( p(x,y) / (p(x) p(y)) )

    Cells with zero joint probability contribute zero, which is the correct
    limit of p log p as p -> 0.
    """
    xi = np.asarray(xi, dtype=np.int64)
    xj = np.asarray(xj, dtype=np.int64)
    n_i = int(n_i or xi.max() + 1)
    n_j = int(n_j or xj.max() + 1)

    joint = joint_histogram(xi, xj, n_i, n_j)
    total = joint.sum()
    if total <= 0:
        return 0.0
    pxy = joint / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pxy / (px @ py)
        terms = np.where(pxy > 0, pxy * np.log(ratio), 0.0)
    mi = float(np.sum(terms))
    # Sampling noise can push the estimate marginally negative; MI is >= 0.
    return max(mi, 0.0)


def ksg_mutual_information(xi: np.ndarray, xj: np.ndarray,
                           n_neighbors: int = 3, seed: int = 0) -> float:
    """Continuous k-NN MI estimate (nats), for the `ksg` estimator option."""
    from sklearn.feature_selection import mutual_info_regression

    value = mutual_info_regression(
        np.asarray(xi, dtype=float).reshape(-1, 1),
        np.asarray(xj, dtype=float).ravel(),
        n_neighbors=n_neighbors, random_state=seed,
    )[0]
    return max(float(value), 0.0)


def mutual_information_matrix(X_discrete: np.ndarray,
                              cardinalities: list[int] | None = None,
                              *, estimator: str = "plug_in",
                              X_continuous: np.ndarray | None = None,
                              n_neighbors: int = 3,
                              seed: int = 0) -> np.ndarray:
    """Symmetric MI matrix over all feature pairs.

    The diagonal is set to zero: a Chow-Liu tree never uses self-edges, and
    leaving H(Xi) on the diagonal would dominate the spanning-tree search.
    """
    X_discrete = np.asarray(X_discrete, dtype=np.int64)
    n_features = X_discrete.shape[1]
    if cardinalities is None:
        cardinalities = [int(X_discrete[:, j].max() + 1) for j in range(n_features)]

    mi = np.zeros((n_features, n_features), dtype=float)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if estimator == "plug_in":
                value = plug_in_mutual_information(
                    X_discrete[:, i], X_discrete[:, j],
                    cardinalities[i], cardinalities[j])
            elif estimator == "ksg":
                if X_continuous is None:
                    raise ValueError("ksg estimator requires X_continuous")
                value = ksg_mutual_information(
                    X_continuous[:, i], X_continuous[:, j],
                    n_neighbors=n_neighbors, seed=seed)
            else:
                raise ValueError(f"Unknown MI estimator {estimator!r}")
            mi[i, j] = mi[j, i] = value

    LOGGER.info("MI matrix (%s): %d features, mean off-diagonal I=%.4f nats, "
                "max=%.4f", estimator, n_features,
                mi[np.triu_indices(n_features, 1)].mean() if n_features > 1 else 0.0,
                mi.max())
    return mi


def conditional_mutual_information(xi: np.ndarray, xj: np.ndarray,
                                   y: np.ndarray, n_i: int, n_j: int) -> float:
    """I(Xi ; Xj | Y), used to inspect class-dependent structure."""
    total = 0.0
    n = len(y)
    for cls in np.unique(y):
        mask = y == cls
        weight = mask.sum() / n
        if mask.sum() < 2:
            continue
        total += weight * plug_in_mutual_information(xi[mask], xj[mask], n_i, n_j)
    return float(total)
