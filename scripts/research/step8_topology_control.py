"""Step 8: matched-sparsity CLT control.

Compares Chow-Liu tree entanglement (n-1 edges, MI-informed) against:
  - "ring": circular topology (n edges -- same qubit count, off by one edge
    from the tree's sparsity, included because it's the natural sparse
    non-chain topology already implemented in this codebase)
  - "random_sparse": a random spanning tree (exactly n-1 edges, same count
    as the CLT tree) with random (non-MI-informed) edges, seeded per qubit
    count for reproducibility

at 4/6/8 qubits, using QNN (cheapest of the two variational models) on the
'pca' pipeline, single seed (init_seed=42) -- NOT multi-seed like the
existing 3-5-seed structural_benefit.csv numbers; this is a faster, single-
seed screening control and is reported as such.

Reuses build_circuit/circuit_metrics/transpile_metrics so depth/gate-count
comparisons are apples-to-apples with the existing CLT-vs-generic report.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "layer1" / "quantum_contribution"
OUT.mkdir(parents=True, exist_ok=True)


def random_spanning_tree(n: int, seed: int) -> list[tuple[int, int]]:
    """A uniformly random labelled spanning tree via random Prufer sequence."""
    rng = np.random.default_rng(seed)
    if n < 2:
        return []
    if n == 2:
        return [(0, 1)]
    prufer = rng.integers(0, n, size=n - 2).tolist()
    degree = [1] * n
    for node in prufer:
        degree[node] += 1
    import heapq
    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)
    edges = []
    for node in prufer:
        leaf = heapq.heappop(leaves)
        edges.append((min(leaf, node), max(leaf, node)))
        degree[leaf] -= 1
        degree[node] -= 1
        if degree[node] == 1:
            heapq.heappush(leaves, node)
    u, v = sorted(heapq.nsmallest(2, leaves))
    edges.append((u, v))
    return edges


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.qml_experimental.data import build_subset, row_fingerprint
    from src.utils.config import get_seed
    from src.qml_experimental.backends import BackendSpec
    from src.qml_experimental.chow_liu.feature_structure import learn_feature_structure
    from src.qml_experimental.chow_liu.quantum_mapping import baseline_entangling_pairs, build_qubit_mapping
    from src.qml_experimental.circuits import build_circuit, circuit_metrics
    from src.qml_experimental.transpilation import primary_coupling, transpile_metrics
    from src.qml_experimental.models import train_qnn
    from sklearn.metrics import average_precision_score

    fp = row_fingerprint()
    assert fp == "2da40ffb169cd76b", f"fingerprint mismatch: {fp}"
    seed = get_seed()
    backend = BackendSpec(name="ideal")

    rows = []
    for n_qubits in (4, 6, 8):
        subset = build_subset("pca", n_qubits)
        Xtr, ytr = subset.X_train, subset.y_train
        Xva, yva = subset.X_val, subset.y_val

        structure = learn_feature_structure(Xtr, subset.feature_names)
        mapping = build_qubit_mapping(structure, "cz")
        clt_pairs = mapping.entangling_pairs
        n_edges = len(clt_pairs)

        topologies = {
            "chow_liu": clt_pairs,
            "ring": baseline_entangling_pairs(n_qubits, "circular"),
            "random_sparse": random_spanning_tree(n_qubits, seed=1000 + n_qubits),
        }

        for topo_name, pairs in topologies.items():
            built = build_circuit(n_qubits, pairs, feature_map_reps=1, ansatz_reps=2,
                                   entangling_gate="cz")
            cm = circuit_metrics(built["circuit"])
            coupling = primary_coupling()
            tm = transpile_metrics(built["circuit"], coupling)

            t0 = time.perf_counter()
            out = train_qnn(subset, pairs, backend=backend, structure=topo_name,
                             init_seed=seed)
            runtime = time.perf_counter() - t0
            pr_auc = average_precision_score(yva, out["val_score"])

            row = dict(
                qubits=n_qubits, topology=topo_name, n_edges=len(pairs),
                clt_n_edges=n_edges,
                logical_depth=cm.get("logical_depth"),
                logical_two_qubit_gates=cm.get("logical_two_qubit_gates"),
                transpiled_depth=tm.get("transpiled_depth"),
                transpiled_two_qubit_gates=tm.get("transpiled_two_qubit_gates"),
                swap_count=tm.get("swap_count"),
                val_pr_auc=round(float(pr_auc), 4),
                train_seconds=round(runtime, 2),
            )
            rows.append(row)
            print(row)

    with open(OUT / "clt_topology_control.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Step 8 complete -> research/quantum_contribution/clt_topology_control.csv")


if __name__ == "__main__":
    main()
