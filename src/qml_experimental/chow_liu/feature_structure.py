"""CLT in *structural* mode: learn a dependency graph, not a classifier.

The generative classifier in ``classifier.py`` fits one tree per class. For
circuit design we want a single dependency graph over the feature set, learned
from training data only, which then dictates which qubit pairs get entangled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.qml_experimental.chow_liu.discretization import Discretizer, fit_discretizer
from src.qml_experimental.chow_liu.tree import ChowLiuTree, build_chow_liu_tree
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class FeatureStructure:
    """A dependency tree over features, plus how it was estimated."""

    tree: ChowLiuTree
    discretizer: Discretizer
    feature_names: list[str]
    estimator: str
    interpretable: bool

    @property
    def edges(self) -> list[tuple[int, int]]:
        return list(self.tree.edges)

    @property
    def n_edges(self) -> int:
        return self.tree.n_edges

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree": self.tree.to_dict(),
            "discretization": self.discretizer.to_dict(),
            "mi_estimator": self.estimator,
            "interpretable_features": self.interpretable,
        }


def learn_feature_structure(X_train: np.ndarray, feature_names: list[str],
                            *, n_bins: int = 8, strategy: str = "quantile",
                            estimator: str = "plug_in", n_neighbors: int = 3,
                            interpretable: bool = True,
                            seed: int = 0) -> FeatureStructure:
    """Learn the CLT dependency graph from TRAINING features only.

    Note this is unsupervised with respect to the label: it describes how
    transaction features co-vary, which is what the circuit topology should
    reflect. Using labels here would bake target information into the circuit
    architecture itself.
    """
    X_train = np.asarray(X_train, dtype=float)
    discretizer = fit_discretizer(X_train, strategy=strategy, n_bins=n_bins)
    X_disc = discretizer.transform(X_train)

    tree = build_chow_liu_tree(
        X_disc, feature_names, cardinalities=discretizer.cardinalities,
        estimator=estimator, X_continuous=X_train, n_neighbors=n_neighbors,
        seed=seed)

    if not interpretable:
        LOGGER.warning(
            "Feature structure learned over PCA components: edges describe "
            "component co-variation, NOT transaction-variable dependencies"
        )
    return FeatureStructure(tree=tree, discretizer=discretizer,
                            feature_names=list(feature_names),
                            estimator=estimator, interpretable=interpretable)
