"""P11: Fraud-discriminative Chow-Liu (DCLT).

Ordinary CLT (learn_feature_structure) is unsupervised w.r.t. the label: it
finds features that co-vary in general, not features whose *relationship*
differs between fraud and legitimate transactions. This experiment builds a
class-conditional variant using ONLY the locked TRAIN rows:

    MI_fraud(i,j)      -- pairwise MI computed on fraud rows only
    MI_legit(i,j)      -- pairwise MI computed on legitimate rows only
    D(i,j) = |MI_fraud(i,j) - MI_legit(i,j)|   (discriminative edge score)

then extracts the maximum-weight spanning tree of D (same n-1 edge count,
same discretizer, same downstream circuit-building code as ordinary CLT) and
trains a QNN on it, compared against the existing ordinary CLT result and the
random-sparse distribution (P6).

This is a candidate contribution, not a claimed one -- see novelty_matrix.md.
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


def main():
    from src.qml_experimental.data import build_subset, row_fingerprint
    from src.utils.config import get_seed
    from src.qml_experimental.backends import BackendSpec
    from src.qml_experimental.chow_liu.discretization import fit_discretizer
    from src.qml_experimental.chow_liu.mutual_information import mutual_information_matrix
    from src.qml_experimental.chow_liu.tree import ChowLiuTree, maximum_weight_spanning_tree
    from src.qml_experimental.chow_liu.feature_structure import learn_feature_structure
    from src.qml_experimental.chow_liu.quantum_mapping import build_qubit_mapping
    from src.qml_experimental.models import train_qnn
    from sklearn.metrics import average_precision_score

    fp = row_fingerprint()
    assert fp == "2da40ffb169cd76b", f"fingerprint mismatch: {fp}"
    init_seed = get_seed()
    backend = BackendSpec(name="ideal")

    rows_out = []
    for n_qubits in (4, 6):
        subset = build_subset("pca", n_qubits)
        Xtr, ytr = subset.X_train, subset.y_train
        Xva, yva = subset.X_val, subset.y_val
        names = subset.feature_names

        discretizer = fit_discretizer(Xtr, strategy="quantile", n_bins=8)
        Xdisc = discretizer.transform(Xtr)

        fraud_mask = ytr == 1
        legit_mask = ytr == 0
        mi_fraud = mutual_information_matrix(
            Xdisc[fraud_mask], discretizer.cardinalities, estimator="plug_in")
        mi_legit = mutual_information_matrix(
            Xdisc[legit_mask], discretizer.cardinalities, estimator="plug_in")
        D = np.abs(mi_fraud - mi_legit)

        edges, weights = maximum_weight_spanning_tree(D)
        dtree = ChowLiuTree(n_nodes=n_qubits, edges=edges, weights=weights,
                             feature_names=names)
        print(f"[{n_qubits}q] DCLT edges: {dtree.named_edges()}", flush=True)

        # Ordinary (pooled, unsupervised) CLT for direct side-by-side.
        ordinary_structure = learn_feature_structure(Xtr, names, seed=0)
        print(f"[{n_qubits}q] Ordinary CLT edges: "
              f"{ordinary_structure.tree.named_edges()}", flush=True)

        for label, pairs in (("dclt", dtree.edges),
                              ("clt_ordinary_rerun", ordinary_structure.edges)):
            t0 = time.perf_counter()
            out = train_qnn(subset, pairs, backend=backend, structure=label,
                             init_seed=init_seed)
            runtime = time.perf_counter() - t0
            pr_auc = float(average_precision_score(yva, out["val_score"]))
            row = dict(qubits=n_qubits, topology=label, n_edges=len(pairs),
                       val_pr_auc=round(pr_auc, 4), train_seconds=round(runtime, 2))
            rows_out.append(row)
            print(row, flush=True)

    with open(OUT / "discriminative_clt_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("Wrote discriminative_clt_results.csv")


if __name__ == "__main__":
    main()
