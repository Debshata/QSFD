"""Local-Sum Decisive validation, step 1: 20-seed PAIRED confirmation + gradient diagnostics.

Architecture frozen exactly as it produced the previous QML-M/CONFIRM_VAL
result (4 qubits, linear topology, feature_map_reps=1, ansatz_reps=2, COBYLA
maxiter=60, uniform(-pi,pi) init). The ONLY variable is the observable:
GLOBAL (Z^n, current project default) vs LOCAL_SUM (mean of single-qubit
Z_i). Trained on QML-M TRAIN (600 rows), evaluated on the FROZEN CONFIRM_VAL
split (250 rows) -- per the task's own instruction that a pre-specified
20-seed confirmation may reuse CONFIRM_VAL once the architecture is frozen
(it already produced one 5-seed result on this split in the prior round;
this widens that to 20 seeds for a properly powered paired test, not a new
architecture search).

20 predetermined seeds, identical for both arms (paired design): each seed
trains BOTH observables from the same initial_point() draw, so
Delta_s = PR_AUC_local_sum_s - PR_AUC_global_s isolates the observable's
effect from initialization luck.

Gradient diagnostics: EstimatorQNN.backward() is a full forward+backward
pass independent of the (gradient-free) COBYLA optimizer actually used to
train -- this project's classifier uses COBYLA precisely because qiskit's
own default has historically been unstable for these circuit sizes, but the
QNN object itself supports computing a genuine parameter-shift/analytic
gradient at any weight vector on demand. This script captures the full
per-iteration weight trajectory during training (via the classifier
callback) and computes gradient norm / mean |grad| / grad variance at 4
checkpoints along that trajectory: initialization, ~33%, ~67%, final.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
QSC = ROOT / "research" / "quantum_signal_confirmation"
OUT = ROOT / "results" / "layer1" / "local_sum_decisive"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

SEEDS = list(range(20))  # 20 predetermined seeds, fixed before any result is seen
N_QUBITS = 4


def make_observable(name, n_qubits):
    from qiskit.quantum_info import SparsePauliOp
    if name == "global":
        return SparsePauliOp("Z" * n_qubits)
    return SparsePauliOp.from_list(
        [(("I" * i) + "Z" + ("I" * (n_qubits - 1 - i)), 1.0 / n_qubits)
         for i in range(n_qubits)])


def main():
    from src.qml_experimental.circuits import build_circuit, circuit_metrics
    from src.qml_experimental.chow_liu.quantum_mapping import baseline_entangling_pairs
    from src.qml_experimental.models import initial_point
    from src.qml_experimental.transpilation import primary_coupling, transpile_metrics
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
    from qiskit_machine_learning import optimizers as opt
    from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score, fbeta_score
    from src.qml_experimental.backends import build_estimator, BackendSpec

    data = np.load(QSC / "qml_m_context.npz", allow_pickle=True)
    Xtr_full, ytr = data["X_train"], data["y_train"]
    Xcv_full, ycv = data["X_confirm_val"], data["y_confirm_val"]
    Xtr, Xcv = Xtr_full[:, :N_QUBITS], Xcv_full[:, :N_QUBITS]
    print(f"TRAIN n={len(ytr)} ({ytr.sum()} fraud), CONFIRM_VAL n={len(ycv)} ({ycv.sum()} fraud), "
          f"qubits={N_QUBITS}", flush=True)

    pairs = baseline_entangling_pairs(N_QUBITS, "linear")
    built = build_circuit(N_QUBITS, pairs, feature_map_reps=1, ansatz_reps=2, entangling_gate="cz")
    circuit = built["circuit"]
    cm = circuit_metrics(circuit)
    tm = transpile_metrics(circuit, primary_coupling())
    print(f"Circuit resources (identical for both arms): {cm}, transpiled: {tm}", flush=True)

    seed_rows = []
    grad_rows = []
    output_dist_rows = []
    learning_curves = {}  # {(observable, seed): np.array of objective values}

    for seed in SEEDS:
        init = initial_point(built["weight_params"].__len__(), seed)
        per_observable = {}
        for obs_name in ("global", "local_sum"):
            obs = make_observable(obs_name, N_QUBITS)
            estimator, pm = build_estimator(BackendSpec(name="ideal"))
            qnn = EstimatorQNN(circuit=circuit, observables=obs,
                                input_params=built["input_params"],
                                weight_params=built["weight_params"],
                                estimator=estimator, pass_manager=pm)

            weight_trace = []
            value_trace = []

            def cb(w, v, _wt=weight_trace, _vt=value_trace):
                _wt.append(np.array(w, dtype=float).copy())
                _vt.append(float(v))

            model = NeuralNetworkClassifier(
                neural_network=qnn, optimizer=opt.COBYLA(maxiter=60),
                loss="squared_error", initial_point=init.copy(), callback=cb)
            y_pm = (2 * ytr.astype(int) - 1).astype(float)
            t0 = time.perf_counter()
            model.fit(Xtr, y_pm)
            runtime = time.perf_counter() - t0

            raw = np.asarray(qnn.forward(Xcv, model.weights), dtype=float).ravel()
            score = (raw + 1.0) / 2.0
            pred = (score >= 0.5).astype(int)
            pr_auc = float(average_precision_score(ycv, score))
            recall = float(recall_score(ycv, pred, zero_division=0))
            precision = float(precision_score(ycv, pred, zero_division=0))
            f1 = float(f1_score(ycv, pred, zero_division=0))
            f2 = float(fbeta_score(ycv, pred, beta=2, zero_division=0))

            row = dict(seed=seed, observable=obs_name, pr_auc=round(pr_auc, 4),
                       recall=round(recall, 4), precision=round(precision, 4),
                       f1=round(f1, 4), f2=round(f2, 4), runtime=round(runtime, 2),
                       n_iterations=len(value_trace))
            seed_rows.append(row)
            per_observable[obs_name] = pr_auc
            learning_curves[f"{obs_name}_{seed}"] = np.array(value_trace)
            print(f"seed={seed} obs={obs_name} PR-AUC={pr_auc:.4f} F2={f2:.4f} "
                  f"runtime={runtime:.1f}s", flush=True)

            # Gradient diagnostics at 4 checkpoints along the actual training trajectory.
            n_steps = len(weight_trace)
            if n_steps >= 4:
                checkpoints = {"init": init, "early_33pct": weight_trace[n_steps // 3],
                                "mid_67pct": weight_trace[2 * n_steps // 3],
                                "final": weight_trace[-1]}
            else:
                checkpoints = {"init": init, "final": weight_trace[-1] if weight_trace else init}
            for cp_name, w in checkpoints.items():
                _, weight_grad = qnn.backward(Xtr, w)
                # weight_grad shape: (n_samples, n_outputs, n_weights) -> flatten per-sample
                g = np.asarray(weight_grad).reshape(len(Xtr), -1)
                grad_norm = float(np.linalg.norm(g.mean(axis=0)))
                mean_abs_grad = float(np.mean(np.abs(g)))
                grad_variance = float(np.var(g))
                grad_rows.append(dict(seed=seed, observable=obs_name, checkpoint=cp_name,
                                       grad_norm=round(grad_norm, 6),
                                       mean_abs_grad=round(mean_abs_grad, 6),
                                       grad_variance=round(grad_variance, 8)))

            # Output-distribution stats by class, for this seed/observable.
            for cls, mask in (("fraud", ycv == 1), ("legitimate", ycv == 0)):
                vals = score[mask]
                output_dist_rows.append(dict(
                    seed=seed, observable=obs_name, cls=cls,
                    mean=round(float(vals.mean()), 4), std=round(float(vals.std()), 4),
                    median=round(float(np.median(vals)), 4),
                    iqr=round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 4)))

        seed_rows[-1]["_pair_check"] = per_observable  # not written to CSV, sanity only

    with open(OUT / "seed_results.csv", "w", newline="") as f:
        fieldnames = [k for k in seed_rows[0].keys() if k != "_pair_check"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in seed_rows:
            w.writerow({k: v for k, v in r.items() if k != "_pair_check"})

    with open(OUT / "gradient_diagnostics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grad_rows[0].keys()))
        w.writeheader(); w.writerows(grad_rows)

    with open(OUT / "output_distribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(output_dist_rows[0].keys()))
        w.writeheader(); w.writerows(output_dist_rows)

    np.savez(OUT / "learning_curves.npz", **{k: v for k, v in learning_curves.items()})

    with open(OUT / "resource_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "global", "local_sum"])
        w.writerow(["n_weight_params", len(built["weight_params"]), len(built["weight_params"])])
        w.writerow(["logical_depth", cm.get("logical_depth"), cm.get("logical_depth")])
        w.writerow(["logical_two_qubit_gates", cm.get("logical_two_qubit_gates"),
                    cm.get("logical_two_qubit_gates")])
        w.writerow(["transpiled_depth", tm.get("transpiled_depth"), tm.get("transpiled_depth")])
        w.writerow(["transpiled_two_qubit_gates", tm.get("transpiled_two_qubit_gates"),
                    tm.get("transpiled_two_qubit_gates")])
        w.writerow(["note", "Circuit is IDENTICAL for both arms -- only the measured observable differs, "
                             "so resource cost is exactly equal by construction.", ""])

    print("Wrote seed_results.csv, gradient_diagnostics.csv, output_distribution.csv, "
          "learning_curves.npz, resource_comparison.csv", flush=True)


if __name__ == "__main__":
    main()
