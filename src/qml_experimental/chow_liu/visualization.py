"""Chow-Liu tree and circuit-comparison figures."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.qml_experimental.chow_liu.tree import ChowLiuTree  # noqa: E402
from src.utils.config import get_path  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger(__name__)


def figures_dir(sub: str = "") -> Path:
    path = Path(get_path("reports")) / "qml_experimental" / "figures"
    if sub:
        path = path / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tree_layout(tree: ChowLiuTree) -> dict[int, tuple[float, float]]:
    """Layered layout by BFS depth from the root -- readable for n <= 10."""
    adjacency: dict[int, list[int]] = {i: [] for i in range(tree.n_nodes)}
    for a, b in tree.edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    levels: dict[int, int] = {tree.root: 0}
    order = [tree.root]
    queue = [tree.root]
    while queue:
        node = queue.pop(0)
        for nb in sorted(adjacency[node]):
            if nb not in levels:
                levels[nb] = levels[node] + 1
                order.append(nb)
                queue.append(nb)

    by_level: dict[int, list[int]] = {}
    for node in order:
        by_level.setdefault(levels[node], []).append(node)

    pos: dict[int, tuple[float, float]] = {}
    for level, nodes in by_level.items():
        for i, node in enumerate(nodes):
            x = (i - (len(nodes) - 1) / 2.0) * 2.0
            pos[node] = (x, -float(level))
    return pos


def plot_chow_liu_tree(tree: ChowLiuTree, title: str, filename: str,
                       subdir: str = "chow_liu") -> str:
    """Draw the dependency tree with mutual-information edge weights."""
    pos = _tree_layout(tree)
    fig, ax = plt.subplots(figsize=(max(7, tree.n_nodes * 1.5), 6))

    weights = np.array(tree.weights, dtype=float)
    wmin, wmax = float(weights.min()), float(weights.max())
    span = (wmax - wmin) or 1.0

    for (a, b), w in zip(tree.edges, tree.weights):
        xa, ya = pos[a]
        xb, yb = pos[b]
        # Thicker/darker edge = stronger measured dependency.
        strength = (w - wmin) / span
        ax.plot([xa, xb], [ya, yb], "-", lw=1.5 + 3.5 * strength,
                color=plt.cm.viridis(0.15 + 0.7 * strength), zorder=1)
        ax.text((xa + xb) / 2, (ya + yb) / 2, f"{w:.3f}", fontsize=8,
                ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    for node, (x, y) in pos.items():
        ax.scatter([x], [y], s=1400, c="#e8eef7", edgecolors="#33475b",
                   linewidths=1.5, zorder=2)
        ax.text(x, y, tree.feature_names[node], fontsize=9, ha="center",
                va="center", zorder=4, weight="bold")

    ax.set_title(f"{title}\n{tree.n_nodes} nodes, {tree.n_edges} edges, "
                 f"total MI {sum(tree.weights):.3f} nats", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    path = figures_dir(subdir) / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Wrote %s", path)
    return str(path)


def plot_mi_matrix(mi: np.ndarray, feature_names: list[str], title: str,
                   filename: str, subdir: str = "chow_liu") -> str:
    fig, ax = plt.subplots(figsize=(1.1 * len(feature_names) + 3,
                                    1.1 * len(feature_names) + 2))
    im = ax.imshow(mi, cmap="viridis")
    ax.set_xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")
    ax.set_yticks(range(len(feature_names)), feature_names)
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            if i != j:
                ax.text(j, i, f"{mi[i, j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if mi[i, j] < mi.max() * 0.6 else "black")
    ax.set_title(f"{title}\npairwise mutual information (nats)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="I(Xi ; Xj) [nats]")
    fig.tight_layout()
    path = figures_dir(subdir) / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Wrote %s", path)
    return str(path)


def plot_structure_comparison(rows: list[dict[str, Any]], filename: str
                              ) -> str | None:
    """Standard vs CLT: depth, 2Q gates, transpiled depth, SWAPs."""
    if not rows:
        return None
    import pandas as pd

    df = pd.DataFrame(rows)
    panels = [
        ("logical_depth", "Logical depth"),
        ("logical_two_qubit_gates", "Logical 2-qubit gates"),
        ("transpiled_depth", "Transpiled depth"),
        ("transpiled_two_qubit_gates", "Transpiled 2-qubit gates"),
    ]
    available = [(c, t) for c, t in panels
                 if c in df.columns and df[c].notna().any()]
    if not available:
        return None

    fig, axes = plt.subplots(1, len(available), figsize=(4.6 * len(available), 4.6))
    axes = np.atleast_1d(axes)
    structures = sorted(df["structure"].dropna().unique())
    qubit_values = sorted(df["qubits"].dropna().unique())
    width = 0.8 / max(len(structures), 1)

    for ax, (col, title) in zip(axes, available):
        for k, structure in enumerate(structures):
            sub = df[df["structure"] == structure]
            values = [sub[sub["qubits"] == q][col].mean() for q in qubit_values]
            offsets = np.arange(len(qubit_values)) + k * width - 0.4 + width / 2
            ax.bar(offsets, values, width, label=structure)
        ax.set_xticks(range(len(qubit_values)), [f"{int(q)}q" for q in qubit_values])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("count / depth")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Circuit complexity: generic vs Chow-Liu entanglement", fontsize=12)
    fig.tight_layout()
    path = figures_dir() / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Wrote %s", path)
    return str(path)
