"""Quantum Signal Confirmation, step 4: FROZEN finalist evaluation on CONFIRM_VAL.

A1 (local-sum observable) is the only candidate that showed a reproducible
signal on ARCH_VAL (mean PR-AUC 0.8460 vs A0's 0.7721, lower seed variance
too, replicating the earlier QML-S screen). Per the confirmation protocol,
it is now frozen -- same circuit, same 5 seeds, same training code as
qsc_03_variational_ablation.py -- and evaluated exactly ONCE on the
CONFIRM_VAL split (250 rows, disjoint from TRAIN and ARCH_VAL, never used
for any architecture decision so far). A0 (the existing baseline) is
evaluated identically as the required comparator. No architecture changes
are made based on this result -- it either confirms or it doesn't.
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

SEEDS = [42, 7, 13, 99, 2024]
N_QUBITS = 4


def main():
    from src.qml_experimental.circuits import build_circuit
    from src.qml_experimental.chow_liu.quantum_mapping import baseline_entangling_pairs
    from src.qml_experimental.models import initial_point
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
    from qiskit_machine_learning import optimizers as opt
    from sklearn.metrics import average_precision_score
    from src.qml_experimental.backends import build_estimator, BackendSpec
    from scipy.stats import mannwhitneyu

    data = np.load(OUT / "qml_m_context.npz", allow_pickle=True)
    Xtr_full, ytr = data["X_train"], data["y_train"]
    Xcv_full, ycv = data["X_confirm_val"], data["y_confirm_val"]
    Xtr, Xcv = Xtr_full[:, :N_QUBITS], Xcv_full[:, :N_QUBITS]
    print(f"CONFIRM_VAL (frozen, first use): n={len(ycv)}, fraud={int(ycv.sum())}", flush=True)

    pairs = baseline_entangling_pairs(N_QUBITS, "linear")
    built = build_circuit(N_QUBITS, pairs, feature_map_reps=1, ansatz_reps=2,
                           entangling_gate="cz")

    def run(observable_name):
        prs = []
        for seed in SEEDS:
            n_qubits = N_QUBITS
            if observable_name == "global":
                obs = SparsePauliOp("Z" * n_qubits)
            else:
                obs = SparsePauliOp.from_list(
                    [(("I" * i) + "Z" + ("I" * (n_qubits - 1 - i)), 1.0 / n_qubits)
                     for i in range(n_qubits)])
            estimator, pm = build_estimator(BackendSpec(name="ideal"))
            qnn = EstimatorQNN(circuit=built["circuit"], observables=obs,
                                input_params=built["input_params"],
                                weight_params=built["weight_params"],
                                estimator=estimator, pass_manager=pm)
            init = initial_point(qnn.num_weights, seed)
            model = NeuralNetworkClassifier(
                neural_network=qnn, optimizer=opt.COBYLA(maxiter=60),
                loss="squared_error", initial_point=init)
            y_pm = (2 * ytr.astype(int) - 1).astype(float)
            t0 = time.perf_counter()
            model.fit(Xtr, y_pm)
            runtime = time.perf_counter() - t0
            raw = np.asarray(qnn.forward(Xcv, model.weights), dtype=float).ravel()
            score = (raw + 1.0) / 2.0
            pr_auc = float(average_precision_score(ycv, score))
            prs.append(pr_auc)
            print(f"[{observable_name}] seed={seed} CONFIRM_VAL PR-AUC={pr_auc:.4f} "
                  f"runtime={runtime:.1f}s", flush=True)
        return np.array(prs)

    a0_confirm = run("global")
    a1_confirm = run("local_sum")

    u_stat, p_value = mannwhitneyu(a1_confirm, a0_confirm, alternative="greater")

    result = {
        "split": "CONFIRM_VAL (frozen, n=250, 75 fraud, first and only use)",
        "A0_baseline_global_observable": {
            "seeds": SEEDS, "pr_auc_values": a0_confirm.tolist(),
            "mean": round(float(a0_confirm.mean()), 4), "std": round(float(a0_confirm.std()), 4),
        },
        "A1_local_sum_observable": {
            "seeds": SEEDS, "pr_auc_values": a1_confirm.tolist(),
            "mean": round(float(a1_confirm.mean()), 4), "std": round(float(a1_confirm.std()), 4),
        },
        "delta_mean": round(float(a1_confirm.mean() - a0_confirm.mean()), 4),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p_one_sided": float(p_value),
        "arch_val_result_for_comparison": {
            "A0_mean": 0.7721, "A0_std": 0.0302, "A1_mean": 0.8460, "A1_std": 0.0280,
        },
        "confirmed": bool(p_value < 0.05 and a1_confirm.mean() > a0_confirm.mean()),
    }
    print(json.dumps(result, indent=2), flush=True)
    with open(OUT / "confirmation_eval_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Wrote confirmation_eval_result.json")


if __name__ == "__main__":
    main()
