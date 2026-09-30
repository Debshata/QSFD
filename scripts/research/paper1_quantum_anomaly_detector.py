"""PAPER_05-motivated experiment: quantum-kernel one-class SVM anomaly detector.

Source: Kyriienko & Magnusson, "Unsupervised quantum machine learning for
fraud detection" (arXiv:2208.01203v1). That paper runs OC-SVM with an
IQP-style quantum kernel (fidelity |<phi(x)|phi(x')>|^2) trained on NOMINAL
(legitimate) transactions only, on the exact same Kaggle mlg-ulb/
creditcardfraud dataset this repository uses, and compares it against a
classical OC-SVM with an RBF kernel.

Every one of our project's quantum models so far (QNN/VQC/QSVM/CLT variants)
is a SUPERVISED classifier -- trained with both classes and directly
comparable to classical supervised models, which is exactly why the
complementarity analysis found positively-correlated errors (Section 1 of
research/quantum_contribution/). Kyriienko & Magnusson's paper motivates a
genuinely different formulation: an UNSUPERVISED anomaly detector trained
only on the "normal" class, structurally closer to our own classical
Isolation Forest than to XGBoost/SVM. This is the most literature-grounded,
directly relevant idea across the seven supplied papers because it targets
the exact dataset this project's production model uses.

Implementation: reuses build_feature_encoding + FidelityQuantumKernel
(already used for the supervised QSVM) as a precomputed kernel for sklearn's
OneClassSVM, fit ONLY on locked-TRAIN legitimate rows (label==0), scored on
locked VALIDATION (both classes). Compared against:
  - classical OneClassSVM(kernel="rbf") on the identical legitimate-only
    train rows (the paper's own classical baseline)
  - the existing Isolation Forest (already a classical anomaly detector in
    this project, per layer1_context.md Section 6)

Uses TRAIN/VALIDATION only; the locked TEST split is not touched.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "layer1" / "paper_driven_quantum"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


def main():
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import average_precision_score

    from src.qml_experimental.data import build_subset, row_fingerprint
    from src.utils.config import get_seed
    from src.qml_experimental.circuits import build_feature_encoding
    from src.qml_experimental.chow_liu.feature_structure import learn_feature_structure
    from src.qml_experimental.chow_liu.quantum_mapping import build_qubit_mapping, baseline_entangling_pairs
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    fp = row_fingerprint()
    assert fp == "2da40ffb169cd76b", f"fingerprint mismatch: {fp}"
    seed = get_seed()

    rows_out = []
    for n_qubits in (4, 6):
        subset = build_subset("selected", n_qubits)
        Xtr, ytr = subset.X_train, subset.y_train
        Xva, yva = subset.X_val, subset.y_val
        names = subset.feature_names

        legit_mask = ytr == 0
        X_nominal = Xtr[legit_mask]
        print(f"[{n_qubits}q] {legit_mask.sum()} legitimate train rows "
              f"(of {len(ytr)}) used to fit the anomaly detectors", flush=True)

        # Chow-Liu topology (reuse our existing structure-learning code) for
        # the quantum feature map, so the encoding is consistent with the
        # rest of this project's quantum experiments rather than an
        # arbitrary generic map.
        structure = learn_feature_structure(Xtr, names, seed=0)
        clt_pairs = build_qubit_mapping(structure, "cz").entangling_pairs

        for topo_name, pairs in (("clt", clt_pairs),
                                  ("linear", baseline_entangling_pairs(n_qubits, "linear"))):
            feature_map, _ = build_feature_encoding(n_qubits, pairs, reps=1,
                                                      entangling_gate="cz")
            kernel = FidelityQuantumKernel(feature_map=feature_map)

            t0 = time.perf_counter()
            K_train = kernel.evaluate(x_vec=X_nominal)
            oc_svm_q = OneClassSVM(kernel="precomputed", nu=0.1)
            oc_svm_q.fit(K_train)
            train_seconds = time.perf_counter() - t0

            K_val = kernel.evaluate(x_vec=Xva, y_vec=X_nominal)
            # decision_function: higher = more normal. Anomaly score = -decision.
            q_anomaly_score = -oc_svm_q.decision_function(K_val)
            q_ap = float(average_precision_score(yva, q_anomaly_score))

            row = dict(qubits=n_qubits, topology=topo_name, method="quantum_ocsvm",
                       val_pr_auc=round(q_ap, 4), n_nominal_train=int(legit_mask.sum()),
                       train_seconds=round(train_seconds, 2))
            rows_out.append(row)
            print(row, flush=True)

        # Classical controls on the IDENTICAL legitimate-only rows/features.
        oc_svm_rbf = OneClassSVM(kernel="rbf", nu=0.1,
                                  gamma=1.0 / (n_qubits * X_nominal.var()))
        oc_svm_rbf.fit(X_nominal)
        rbf_anomaly_score = -oc_svm_rbf.decision_function(Xva)
        rbf_ap = float(average_precision_score(yva, rbf_anomaly_score))
        row = dict(qubits=n_qubits, topology="n/a", method="classical_ocsvm_rbf",
                   val_pr_auc=round(rbf_ap, 4), n_nominal_train=int(legit_mask.sum()),
                   train_seconds=0.0)
        rows_out.append(row)
        print(row, flush=True)

        iso = IsolationForest(random_state=seed, contamination=0.1)
        iso.fit(X_nominal)
        iso_anomaly_score = -iso.score_samples(Xva)
        iso_ap = float(average_precision_score(yva, iso_anomaly_score))
        row = dict(qubits=n_qubits, topology="n/a", method="isolation_forest",
                   val_pr_auc=round(iso_ap, 4), n_nominal_train=int(legit_mask.sum()),
                   train_seconds=0.0)
        rows_out.append(row)
        print(row, flush=True)

    with open(OUT / "quantum_anomaly_detector_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("Wrote research/paper_driven_quantum/quantum_anomaly_detector_results.csv")


if __name__ == "__main__":
    main()
