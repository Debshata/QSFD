"""Quantum Signal Confirmation, step 2 (Track A): anomaly detector at scale.

Reproduces the promising 6-qubit quantum-kernel OC-SVM result from
research/paper_driven_quantum/ on the larger, independent QML-M development
context (600 train / 250 arch_val, from qsc_01_build_confirmation_context.py)
rather than the original 250/100 locked QML-S subset, to check whether the
signal survives more legitimate-training rows and a materially larger
validation set (250 vs 100 rows -- 2.5x the fraud count for a more reliable
PR-AUC estimate).

Uses ARCH_VAL only (never CONFIRM_VAL, per the confirmation protocol -- this
is architecture-level screening, not a frozen-finalist evaluation).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "layer1" / "quantum_signal_confirmation"
sys.path.insert(0, str(ROOT))


def main():
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import average_precision_score

    from src.utils.config import get_seed
    from src.qml_experimental.circuits import build_feature_encoding
    from src.qml_experimental.chow_liu.quantum_mapping import baseline_entangling_pairs
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    seed = get_seed()
    protocol = json.loads((OUT / "confirmation_protocol.json").read_text())
    data = np.load(OUT / "qml_m_context.npz", allow_pickle=True)
    Xtr, ytr = data["X_train"], data["y_train"]
    Xav, yav = data["X_arch_val"], data["y_arch_val"]
    n_qubits = Xtr.shape[1]
    print(f"QML-M context loaded: train={len(ytr)} ({ytr.sum()} fraud), "
          f"arch_val={len(yav)} ({yav.sum()} fraud), qubits={n_qubits}", flush=True)

    legit_mask = ytr == 0
    X_nominal = Xtr[legit_mask]
    print(f"Fitting anomaly detectors on {legit_mask.sum()} legitimate TRAIN rows "
          f"(vs 175 in the original QML-S screen -- a {legit_mask.sum()/175:.1f}x increase)",
          flush=True)

    pairs = baseline_entangling_pairs(n_qubits, "linear")  # the topology that won at 6q before
    feature_map, _ = build_feature_encoding(n_qubits, pairs, reps=1, entangling_gate="cz")
    kernel = FidelityQuantumKernel(feature_map=feature_map)

    print("STARTED quantum kernel fit", time.strftime("%H:%M:%S"), flush=True)
    t0 = time.perf_counter()
    K_train = kernel.evaluate(x_vec=X_nominal)
    oc_svm_q = OneClassSVM(kernel="precomputed", nu=0.1)
    oc_svm_q.fit(K_train)
    K_val = kernel.evaluate(x_vec=Xav, y_vec=X_nominal)
    q_score = -oc_svm_q.decision_function(K_val)
    q_runtime = time.perf_counter() - t0
    q_ap = float(average_precision_score(yav, q_score))
    print(f"FINISHED quantum kernel: PR-AUC={q_ap:.4f} runtime={q_runtime:.1f}s "
          f"({time.strftime('%H:%M:%S')})", flush=True)

    oc_svm_rbf = OneClassSVM(kernel="rbf", nu=0.1, gamma=1.0 / (n_qubits * X_nominal.var()))
    oc_svm_rbf.fit(X_nominal)
    rbf_score = -oc_svm_rbf.decision_function(Xav)
    rbf_ap = float(average_precision_score(yav, rbf_score))
    print(f"classical_ocsvm_rbf: PR-AUC={rbf_ap:.4f}", flush=True)

    iso = IsolationForest(random_state=seed, contamination=0.1)
    iso.fit(X_nominal)
    iso_score = -iso.score_samples(Xav)
    iso_ap = float(average_precision_score(yav, iso_score))
    print(f"isolation_forest: PR-AUC={iso_ap:.4f}", flush=True)

    # Error-diversity / score-correlation check (Section 15 of the task).
    corr_q_rbf = float(np.corrcoef(q_score, rbf_score)[0, 1])
    corr_q_iso = float(np.corrcoef(q_score, iso_score)[0, 1])
    print(f"score correlation: quantum vs rbf={corr_q_rbf:.3f}, "
          f"quantum vs isolation_forest={corr_q_iso:.3f}", flush=True)

    # Paired bootstrap on this larger arch_val set.
    rng = np.random.default_rng(seed)
    n_boot = 2000
    idx_all = np.arange(len(yav))

    def bootstrap(score_a, score_b, label):
        deltas = []
        for _ in range(n_boot):
            bidx = rng.choice(idx_all, size=len(idx_all), replace=True)
            if yav[bidx].sum() == 0 or yav[bidx].sum() == len(bidx):
                continue
            pr_a = average_precision_score(yav[bidx], score_a[bidx])
            pr_b = average_precision_score(yav[bidx], score_b[bidx])
            deltas.append(pr_a - pr_b)
        deltas = np.array(deltas)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        point = average_precision_score(yav, score_a) - average_precision_score(yav, score_b)
        print(f"{label}: point={point:+.4f} bootstrap 95% CI=[{lo:+.4f}, {hi:+.4f}] "
              f"({'crosses zero' if lo < 0 < hi else 'does NOT cross zero'})", flush=True)
        return point, float(lo), float(hi)

    d_rbf = bootstrap(q_score, rbf_score, "quantum(linear) - classical_ocsvm_rbf @ QML-M")
    d_iso = bootstrap(q_score, iso_score, "quantum(linear) - isolation_forest @ QML-M")

    results = {
        "context": "QML-M (arch_val, n=250, 75 fraud)",
        "n_legitimate_train": int(legit_mask.sum()),
        "qubits": n_qubits,
        "topology": "linear",
        "quantum_ocsvm_pr_auc": q_ap,
        "classical_ocsvm_rbf_pr_auc": rbf_ap,
        "isolation_forest_pr_auc": iso_ap,
        "quantum_runtime_seconds": round(q_runtime, 1),
        "score_correlation_quantum_vs_rbf": corr_q_rbf,
        "score_correlation_quantum_vs_isolation_forest": corr_q_iso,
        "bootstrap_delta_vs_rbf": {"point": d_rbf[0], "ci_low": d_rbf[1], "ci_high": d_rbf[2]},
        "bootstrap_delta_vs_isolation_forest": {"point": d_iso[0], "ci_low": d_iso[1], "ci_high": d_iso[2]},
        "original_qml_s_result_for_comparison": {
            "quantum_ocsvm_linear_6q": 0.9390, "classical_ocsvm_rbf": 0.9329,
            "isolation_forest": 0.9305, "n_legitimate_train": 175, "n_val": 100,
        },
    }
    with open(OUT / "anomaly_confirmation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(OUT / "anomaly_confirmation_scores.npz", y_val=yav, quantum=q_score,
              classical_rbf=rbf_score, isolation_forest=iso_score)
    print("Wrote anomaly_confirmation_results.json and anomaly_confirmation_scores.npz")


if __name__ == "__main__":
    main()
