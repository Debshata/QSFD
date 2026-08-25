"""The Chow-Liu tree itself: maximum-weight spanning tree over MI edges.

Chow & Liu (1968) showed that the tree-structured distribution closest in
KL divergence to the true joint is the maximum-weight spanning tree of the
complete graph whose edge weights are pairwise mutual informations.

A tree over ``n`` nodes has exactly ``n - 1`` edges. That invariant is checked,
not assumed, and it is the reason CLT gives a *sparse* entanglement pattern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ChowLiuTree:
    """A fitted maximum-weight spanning tree over feature nodes."""

    n_nodes: int
    edges: list[tuple[int, int]]
    weights: list[float]
    feature_names: list[str]
    root: int = 0
    parents: dict[int, int | None] = field(default_factory=dict)
    mi_matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.edges) != self.n_nodes - 1:
            raise ValueError(
                f"A Chow-Liu tree over {self.n_nodes} nodes must have exactly "
                f"{self.n_nodes - 1} edges, got {len(self.edges)}"
            )
        if not self.parents:
            self.parents = self._orient(self.root)

    def _orient(self, root: int) -> dict[int, int | None]:
        """Direct the undirected tree away from ``root`` (BFS).

        The factorisation P(X) = P(root) * prod_i P(Xi | parent(Xi)) needs a
        parent for every non-root node.
        """
        adjacency: dict[int, list[int]] = {i: [] for i in range(self.n_nodes)}
        for a, b in self.edges:
            adjacency[a].append(b)
            adjacency[b].append(a)

        parents: dict[int, int | None] = {root: None}
        queue = [root]
        while queue:
            node = queue.pop(0)
            for nb in adjacency[node]:
                if nb not in parents:
                    parents[nb] = node
                    queue.append(nb)
        if len(parents) != self.n_nodes:
            raise ValueError("Chow-Liu graph is disconnected; not a spanning tree")
        return parents

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def named_edges(self) -> list[tuple[str, str, float]]:
        return [(self.feature_names[a], self.feature_names[b], round(w, 6))
                for (a, b), w in zip(self.edges, self.weights)]

    def degree(self) -> dict[str, int]:
        deg = {name: 0 for name in self.feature_names}
        for a, b in self.edges:
            deg[self.feature_names[a]] += 1
            deg[self.feature_names[b]] += 1
        return deg

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "root": self.root,
            "feature_names": list(self.feature_names),
            "edges": [[int(a), int(b)] for a, b in self.edges],
            "edge_weights_nats": [round(float(w), 6) for w in self.weights],
            "named_edges": [{"a": a, "b": b, "mutual_information": w}
                            for a, b, w in self.named_edges()],
            "parents": {str(k): (None if v is None else int(v))
                        for k, v in self.parents.items()},
            "total_edge_weight": round(float(sum(self.weights)), 6),
            "degree": self.degree(),
        }


def maximum_weight_spanning_tree(weights: np.ndarray) -> tuple[list[tuple[int, int]],
                                                               list[float]]:
    """Maximum-weight spanning tree via Kruskal on negated weights.

    Ties are broken by node index so the tree is deterministic for a given MI
    matrix -- important because the circuit topology depends on it.
    """
    n = weights.shape[0]
    if n < 2:
        return [], []

    candidates = sorted(
        ((float(weights[i, j]), i, j) for i in range(n) for j in range(i + 1, n)),
        key=lambda t: (-t[0], t[1], t[2]),
    )

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]     # path compression
            x = parent[x]
        return x

    edges: list[tuple[int, int]] = []
    chosen: list[float] = []
    for weight, i, j in candidates:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            edges.append((i, j))
            chosen.append(weight)
            if len(edges) == n - 1:
                break

    if len(edges) != n - 1:
        raise ValueError(f"Could not build a spanning tree: got {len(edges)} "
                         f"edges for {n} nodes")
    return edges, chosen


def build_chow_liu_tree(X_discrete: np.ndarray, feature_names: list[str],
                        *, cardinalities: list[int] | None = None,
                        estimator: str = "plug_in",
                        X_continuous: np.ndarray | None = None,
                        n_neighbors: int = 3, root: int = 0,
                        seed: int = 0) -> ChowLiuTree:
    """Estimate pairwise MI on TRAINING data and extract the CLT."""
    from src.qml_experimental.chow_liu.mutual_information import (
        mutual_information_matrix,
    )

    mi = mutual_information_matrix(
        X_discrete, cardinalities, estimator=estimator,
        X_continuous=X_continuous, n_neighbors=n_neighbors, seed=seed)

    edges, weights = maximum_weight_spanning_tree(mi)
    tree = ChowLiuTree(n_nodes=mi.shape[0], edges=edges, weights=weights,
                       feature_names=list(feature_names), root=root, mi_matrix=mi)

    LOGGER.info("Chow-Liu tree: %d nodes, %d edges, total weight %.4f nats",
                tree.n_nodes, tree.n_edges, sum(weights))
    for a, b, w in tree.named_edges():
        LOGGER.info("   %s -- %s   I=%.4f nats", a, b, w)
    return tree
