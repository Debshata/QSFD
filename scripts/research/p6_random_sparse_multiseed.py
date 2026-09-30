"""P6: Random-sparse multi-seed topology study.

Directly answers the open question left by clt_topology_control.csv (single
random graph per qubit count): does Chow-Liu topology lie above, inside, or
below the distribution of random sparse graphs with the SAME edge count?

For 4 and 6 qubits (8 qubits skipped -- ~60s/run makes 10 seeds prohibitively
slow for this direct-execution session; recorded as a scope limitation, not
silently dropped), generates 10 random connected spanning trees (n-1 edges,
matching the Chow-Liu tree's edge count exactly) via distinct Prufer-sequence
seeds, trains a QNN on each with the SAME init_seed=42 (so only the topology
varies), and reports the distribution of validation PR-AUC against the
already-measured chow_liu and ring values from clt_topology_control.csv.

Uses TRAIN/VALIDATION only (never touches the locked test split).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "layer1" / "quantum_contribution"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


def random_spanning_tree(n: int, seed: int):
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
    from src.qml_experimental.data import build_subset, row_fingerprint
    from src.utils.config import get_seed
    from src.qml_experimental.backends import BackendSpec
    from src.qml_experimental.chow_liu.feature_structure import learn_feature_structure
    from src.qml_experimental.chow_liu.quantum_mapping import build_qubit_mapping
    from src.qml_experimental.models import train_qnn
    from sklearn.metrics import average_precision_score

    fp = row_fingerprint()
    assert fp == "2da40ffb169cd76b", f"fingerprint mismatch: {fp}"
    init_seed = get_seed()
    backend = BackendSpec(name="ideal")

    graph_seeds = list(range(5000, 5010))  # 10 distinct random-graph seeds
    rows = []

    for n_qubits in (4, 6):
        subset = build_subset("pca", n_qubits)
        yva = subset.y_val
        pr_aucs = []
        for gseed in graph_seeds:
            pairs = random_spanning_tree(n_qubits, seed=gseed)
            t0 = time.perf_counter()
            out = train_qnn(subset, pairs, backend=backend,
                             structure=f"random_sparse_g{gseed}", init_seed=init_seed)
            runtime = time.perf_counter() - t0
            pr_auc = float(average_precision_score(yva, out["val_score"]))
            pr_aucs.append(pr_auc)
            row = dict(qubits=n_qubits, graph_seed=gseed, n_edges=len(pairs),
                       val_pr_auc=round(pr_auc, 4), train_seconds=round(runtime, 2))
            rows.append(row)
            print(row, flush=True)

        arr = np.array(pr_aucs)
        print(f"qubits={n_qubits}: mean={arr.mean():.4f} std={arr.std():.4f} "
              f"min={arr.min():.4f} max={arr.max():.4f}", flush=True)

    with open(OUT / "random_sparse_multiseed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary comparing against the existing single-seed chow_liu/ring values
    clt_path = OUT / "clt_topology_control.csv"
    clt_rows = list(csv.DictReader(open(clt_path)))
    summary_lines = ["qubits,chow_liu_pr_auc,random_sparse_mean,random_sparse_std,"
                      "random_sparse_min,random_sparse_max,n_random_seeds,"
                      "clt_percentile_in_random_distribution"]
    for n_qubits in (4, 6):
        clt_val = float([r for r in clt_rows if r["qubits"] == str(n_qubits)
                          and r["topology"] == "chow_liu"][0]["val_pr_auc"])
        sub = [r["val_pr_auc"] for r in rows if r["qubits"] == n_qubits]
        arr = np.array(sub)
        percentile = float((arr < clt_val).mean() * 100)
        summary_lines.append(
            f"{n_qubits},{clt_val:.4f},{arr.mean():.4f},{arr.std():.4f},"
            f"{arr.min():.4f},{arr.max():.4f},{len(arr)},{percentile:.1f}"
        )
    with open(OUT / "random_sparse_multiseed_summary.csv", "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))
    print("Wrote random_sparse_multiseed.csv and random_sparse_multiseed_summary.csv")


if __name__ == "__main__":
    main()
