"""Bootstrap significance check for the 6-qubit quantum-OCSVM(linear) result.

paper1_quantum_anomaly_detector.py found quantum_ocsvm(linear, 6q) val PR-AUC
0.9390 > classical_ocsvm_rbf 0.9329 > isolation_forest 0.9305 -- the first
result in this entire research programme where a quantum method's point
estimate beats every classical alternative. Before reporting this as
anything beyond a raw point estimate, this script reruns just the 6-qubit
case, persists the raw validation scores, and runs a paired bootstrap
(same resampled validation rows) against both classical baselines.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "paper_driven_quantum"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


def main():
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import average_precision_score

    from src.qml_experimental.data import build_subset, row_fingerprint
    from src.utils.config import get_seed
    from src.qml_experimental.circuits import build_feature_encoding
    from src.qml_experimental.chow_liu.quantum_mapping import baseline_entangling_pairs
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    fp = row_fingerprint()
    assert fp == "2da40ffb169cd76b", f"fingerprint mismatch: {fp}"
    seed = get_seed()

    n_qubits = 6
    subset = build_subset("selected", n_qubits)
    Xtr, ytr = subset.X_train, subset.y_train
    Xva, yva = subset.X_val, subset.y_val

    legit_mask = ytr == 0
    X_nominal = Xtr[legit_mask]

    pairs = baseline_entangling_pairs(n_qubits, "linear")
    feature_map, _ = build_feature_encoding(n_qubits, pairs, reps=1, entangling_gate="cz")
    kernel = FidelityQuantumKernel(feature_map=feature_map)

    t0 = time.perf_counter()
    K_train = kernel.evaluate(x_vec=X_nominal)
    oc_svm_q = OneClassSVM(kernel="precomputed", nu=0.1)
    oc_svm_q.fit(K_train)
    K_val = kernel.evaluate(x_vec=Xva, y_vec=X_nominal)
    q_score = -oc_svm_q.decision_function(K_val)
    print(f"quantum_ocsvm(linear, 6q) rebuilt in {time.perf_counter()-t0:.1f}s, "
          f"val PR-AUC={average_precision_score(yva, q_score):.4f}", flush=True)

    oc_svm_rbf = OneClassSVM(kernel="rbf", nu=0.1, gamma=1.0 / (n_qubits * X_nominal.var()))
    oc_svm_rbf.fit(X_nominal)
    rbf_score = -oc_svm_rbf.decision_function(Xva)
    print(f"classical_ocsvm_rbf val PR-AUC={average_precision_score(yva, rbf_score):.4f}", flush=True)

    iso = IsolationForest(random_state=seed, contamination=0.1)
    iso.fit(X_nominal)
    iso_score = -iso.score_samples(Xva)
    print(f"isolation_forest val PR-AUC={average_precision_score(yva, iso_score):.4f}", flush=True)

    np.savez(OUT / "anomaly_6q_scores.npz", y_val=yva, quantum=q_score,
             classical_rbf=rbf_score, isolation_forest=iso_score)

    rng = np.random.default_rng(seed)
    n_boot = 2000
    idx_all = np.arange(len(yva))

    def bootstrap_delta(score_a, score_b, label):
        deltas = []
        for _ in range(n_boot):
            bidx = rng.choice(idx_all, size=len(idx_all), replace=True)
            if yva[bidx].sum() == 0 or yva[bidx].sum() == len(bidx):
                continue
            pr_a = average_precision_score(yva[bidx], score_a[bidx])
            pr_b = average_precision_score(yva[bidx], score_b[bidx])
            deltas.append(pr_a - pr_b)
        deltas = np.array(deltas)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        point = average_precision_score(yva, score_a) - average_precision_score(yva, score_b)
        print(f"{label}: point={point:+.4f}  bootstrap 95% CI=[{lo:+.4f}, {hi:+.4f}] "
              f"({'crosses zero' if lo < 0 < hi else 'does NOT cross zero'})", flush=True)
        return point, lo, hi

    bootstrap_delta(q_score, rbf_score, "quantum(linear,6q) - classical_ocsvm_rbf")
    bootstrap_delta(q_score, iso_score, "quantum(linear,6q) - isolation_forest")

    print("Done.")


if __name__ == "__main__":
    main()
