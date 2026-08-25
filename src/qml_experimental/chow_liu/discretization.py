"""Discretisation for Chow-Liu dependency estimation.

Fraud features are continuous; the plug-in mutual-information estimator needs
discrete variables. The discretiser is **fitted on training data only** and
applied unchanged to validation and test -- discretising the test set
independently would let test-set quantiles leak into the model.

Quantile binning is the default: fraud features (amount-like, PCA components)
are heavy-tailed, and uniform-width bins put almost all mass in one bin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class Discretizer:
    """Fitted bin edges plus the metadata needed to reproduce them."""

    strategy: str
    n_bins: int
    edges: list[np.ndarray]
    n_features: int

    @property
    def cardinalities(self) -> list[int]:
        """Actual bin count per feature (may be < n_bins after dedup)."""
        return [len(e) + 1 for e in self.edges]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map continuous values onto integer bin indices."""
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Discretizer fitted on {self.n_features} features, got {X.shape[1]}")
        out = np.empty(X.shape, dtype=np.int64)
        for j in range(self.n_features):
            # searchsorted keeps out-of-range values in the edge bins rather
            # than producing an invalid index.
            out[:, j] = np.searchsorted(self.edges[j], X[:, j], side="right")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_bins": self.n_bins,
            "n_features": self.n_features,
            "cardinalities": self.cardinalities,
            "edges": [e.round(6).tolist() for e in self.edges],
        }


def fit_discretizer(X_train: np.ndarray, *, strategy: str = "quantile",
                    n_bins: int = 8) -> Discretizer:
    """Fit bin edges on TRAINING data only."""
    X = np.asarray(X_train, dtype=float)
    n_features = X.shape[1]
    edges: list[np.ndarray] = []

    for j in range(n_features):
        col = X[:, j]
        if strategy == "quantile":
            q = np.linspace(0, 100, n_bins + 1)[1:-1]
            cuts = np.percentile(col, q)
        elif strategy == "uniform":
            lo, hi = float(col.min()), float(col.max())
            cuts = np.linspace(lo, hi, n_bins + 1)[1:-1]
        elif strategy == "kmeans":
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=n_bins, n_init=10, random_state=0)
            km.fit(col.reshape(-1, 1))
            centres = np.sort(km.cluster_centers_.ravel())
            cuts = (centres[:-1] + centres[1:]) / 2.0
        else:
            raise ValueError(f"Unknown discretisation strategy {strategy!r}")

        # Duplicate cuts create empty bins, which distort the MI estimate.
        cuts = np.unique(cuts)
        edges.append(cuts)

    disc = Discretizer(strategy=strategy, n_bins=n_bins, edges=edges,
                       n_features=n_features)
    LOGGER.info("Discretizer fitted on %d training rows: %s, %d bins -> "
                "cardinalities %s", len(X), strategy, n_bins, disc.cardinalities)
    return disc
