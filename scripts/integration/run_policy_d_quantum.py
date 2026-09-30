"""Policy D (quantum-assisted routing), evaluated on its own separate sample.

Per docs/protocols/LAYER2_INTEGRATION_PROTOCOL.md: the frozen Local-Sum QNN
(exactly as validated in Notebook 1, no retuning) is re-scored once on the
locked QML-M CONFIRM_VAL subset (250 rows), alongside the strongest
controlled classical baseline (SVM-RBF). classical_quantum_disagreement is
computed and used for one pre-specified escalation rule. Never merged
numerically with the Policy A/B/C sample (different size, prevalence, and
feature representation).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "integration"
sys.path.insert(0, str(ROOT))

N_QUBITS = 4
SEED = 0


def main():
    from sklearn.svm import SVC
    from sklearn.metrics import average_precision_score

    from src.fraud_engine.schemas import FraudAssessment
    from src.security.integration.schemas import SecurityContext
    from src.security.policy.security_selector import ContinuousPolicy, PolicyThresholds
    from src.qml_experimental.circuits import build_circuit
    from src.qml_experimental.chow_liu.quantum_mapping import baseline_entangling_pairs
    from src.qml_experimental.models import initial_point
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
    from qiskit_machine_learning import optimizers as opt
    from src.qml_experimental.backends import build_estimator, BackendSpec

    data = np.load(ROOT / "results" / "layer1" / "quantum_signal_confirmation" / "qml_m_context.npz",
                    allow_pickle=True)
    Xtr_full, ytr = data["X_train"], data["y_train"]
    Xcv_full, ycv = data["X_confirm_val"], data["y_confirm_val"]
    Xtr, Xcv = Xtr_full[:, :N_QUBITS], Xcv_full[:, :N_QUBITS]
    print(f"Policy D sample: CONFIRM_VAL n={len(ycv)} ({int(ycv.sum())} fraud) -- SEPARATE from Policy A/B/C sample",
          flush=True)

    # Strongest controlled classical baseline (SVM-RBF, per LAYER1_FINAL_RESULTS.md).
    svm = SVC(kernel="rbf", probability=True, random_state=SEED)
    svm.fit(Xtr, ytr)
    p_classical = svm.predict_proba(Xcv)[:, 1]
    print(f"SVM-RBF (controlled) PR-AUC: {average_precision_score(ycv, p_classical):.4f}", flush=True)

    # Frozen Local-Sum QNN (identical architecture/seed as Notebook 1's validated seed=0).
    pairs = baseline_entangling_pairs(N_QUBITS, "linear")
    built = build_circuit(N_QUBITS, pairs, feature_map_reps=1, ansatz_reps=2, entangling_gate="cz")
    obs = SparsePauliOp.from_list(
        [(("I" * i) + "Z" + ("I" * (N_QUBITS - 1 - i)), 1.0 / N_QUBITS) for i in range(N_QUBITS)])
    estimator, pm = build_estimator(BackendSpec(name="ideal"))
    qnn = EstimatorQNN(circuit=built["circuit"], observables=obs, input_params=built["input_params"],
                        weight_params=built["weight_params"], estimator=estimator, pass_manager=pm)
    init = initial_point(qnn.num_weights, SEED)
    model = NeuralNetworkClassifier(neural_network=qnn, optimizer=opt.COBYLA(maxiter=60),
                                     loss="squared_error", initial_point=init)
    y_pm = (2 * ytr.astype(int) - 1).astype(float)
    t0 = time.perf_counter()
    model.fit(Xtr, y_pm)
    print(f"Local-Sum QNN retrained (frozen architecture, seed={SEED}) in {time.perf_counter()-t0:.1f}s", flush=True)
    raw = np.asarray(qnn.forward(Xcv, model.weights), dtype=float).ravel()
    p_quantum = (raw + 1.0) / 2.0
    print(f"Local-Sum QNN (frozen) PR-AUC: {average_precision_score(ycv, p_quantum):.4f}", flush=True)

    disagreement = np.abs(p_classical - p_quantum)
    print(f"Mean disagreement: {disagreement.mean():.4f}, std: {disagreement.std():.4f}", flush=True)

    thresholds = PolicyThresholds(continuous_pqc_threshold=0.30, continuous_hybrid_threshold=0.70,
                                   continuous_qkd_threshold=0.90, disagreement_escalation_threshold=0.30)
    policy_b = ContinuousPolicy(thresholds=thresholds)
    policy_d = ContinuousPolicy(thresholds=thresholds, use_quantum_signal=True)

    rows = []
    for i in range(len(ycv)):
        fa = FraudAssessment(transaction_id=f"cv-{i}", fraud_probability=float(p_classical[i]),
                              fraud_prediction=int(p_classical[i] >= 0.5), risk_score=float(p_classical[i] * 100),
                              risk_level=("LOW" if p_classical[i] <= 0.30 else "MEDIUM" if p_classical[i] <= 0.70
                                          else "HIGH"),
                              model_name="svm_rbf_controlled", model_version="v1", threshold=0.5)
        ctx_b = SecurityContext.from_fraud_assessment(fa)
        ctx_d = SecurityContext.from_fraud_assessment(
            fa, quantum_signal=float(p_quantum[i]), classical_quantum_disagreement=float(disagreement[i]))
        dec_b = policy_b.decide(ctx_b)
        dec_d = policy_d.decide(ctx_d)
        rows.append(dict(row=i, fraud_label=int(ycv[i]), p_classical=round(float(p_classical[i]), 4),
                          p_quantum=round(float(p_quantum[i]), 4), disagreement=round(float(disagreement[i]), 4),
                          profile_B=dec_b.profile, profile_D=dec_d.profile,
                          escalated_only_by_D=(dec_d.profile != dec_b.profile)))

    with open(RESULTS / "quantum_auxiliary_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    fraud_mask = ycv.astype(bool)
    profiles_b = np.array([r["profile_B"] for r in rows])
    profiles_d = np.array([r["profile_D"] for r in rows])
    esc_b = profiles_b != "BASELINE"
    esc_d = profiles_d != "BASELINE"
    fwc_b = float(esc_b[fraud_mask].mean())
    fwc_d = float(esc_d[fraud_mask].mean())
    ber_b = float(esc_b[~fraud_mask].mean())
    ber_d = float(esc_d[~fraud_mask].mean())
    n_changed = int((profiles_b != profiles_d).sum())
    n_changed_fraud = int(((profiles_b != profiles_d) & fraud_mask).sum())

    print(f"\\nPolicy B (no quantum) on CONFIRM_VAL: FWC={fwc_b:.4f} BER={ber_b:.4f}", flush=True)
    print(f"Policy D (quantum-assisted) on CONFIRM_VAL: FWC={fwc_d:.4f} BER={ber_d:.4f}", flush=True)
    print(f"Rows where D changed routing vs B: {n_changed} (of which {n_changed_fraud} were real fraud)", flush=True)

    result = dict(fwc_B=round(fwc_b, 4), fwc_D=round(fwc_d, 4), ber_B=round(ber_b, 4), ber_D=round(ber_d, 4),
                  delta_fwc=round(fwc_d - fwc_b, 4), delta_ber=round(ber_d - ber_b, 4),
                  n_routing_changed=n_changed, n_routing_changed_that_were_fraud=n_changed_fraud,
                  svm_rbf_pr_auc=round(float(average_precision_score(ycv, p_classical)), 4),
                  local_sum_qnn_pr_auc=round(float(average_precision_score(ycv, p_quantum)), 4),
                  mean_disagreement=round(float(disagreement.mean()), 4))
    import json
    with open(RESULTS / "quantum_auxiliary_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\\nWrote quantum_auxiliary_ablation.csv and quantum_auxiliary_result.json")


if __name__ == "__main__":
    main()
