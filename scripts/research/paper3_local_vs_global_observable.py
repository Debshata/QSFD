"""PAPER_03-motivated experiment: local vs global observable trainability.

Source: Cerezo, Sone, Volkoff, Cincio, Coles, "Cost function dependent barren
plateaus in shallow parametrized quantum circuits" (Nature Communications
12:1791, 2021). Central result: a GLOBAL observable (acting on all qubits,
e.g. Z^{\\otimes n}) causes exponentially vanishing gradients even in SHALLOW
circuits, whereas a LOCAL observable (acting on O(1) qubits) has at worst a
polynomially vanishing gradient.

This project's existing train_qnn (src/qml_experimental/models.py) builds
EstimatorQNN WITHOUT an explicit `observables` argument, which means it uses
qiskit-machine-learning's default global Z^{\\otimes n} observable -- exactly
the choice this paper associates with harder training. This script does not
modify models.py (keeping the existing, tested production QNN untouched);
instead it duplicates the minimal training loop with an explicit
`observables` argument, holding circuit/topology/optimizer/seed fixed, and
compares:

  - "global": Z on every qubit (the current implicit default)
  - "local_single": Z on qubit 0 only
  - "local_sum": (1/n) * sum_i Z_i  (average single-qubit Z)

across 5 seeds at 4 qubits (CLT topology, reusing the existing chow_liu code),
using TRAIN/VALIDATION only.
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
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
    from sklearn.metrics import average_precision_score

    from src.qml_experimental.data import build_subset, row_fingerprint
    from src.qml_experimental.circuits import build_circuit
    from src.qml_experimental.chow_liu.feature_structure import learn_feature_structure
    from src.qml_experimental.chow_liu.quantum_mapping import build_qubit_mapping
    from src.qml_experimental.models import initial_point, _optimizer, _circuit_cfg
    from src.qml_experimental.backends import build_estimator, BackendSpec

    fp = row_fingerprint()
    assert fp == "2da40ffb169cd76b", f"fingerprint mismatch: {fp}"

    n_qubits = 4
    subset = build_subset("selected", n_qubits)
    Xtr, ytr = subset.X_train, subset.y_train
    Xva, yva = subset.X_val, subset.y_val
    names = subset.feature_names

    structure = learn_feature_structure(Xtr, names, seed=0)
    clt_pairs = build_qubit_mapping(structure, "cz").entangling_pairs

    cfg = _circuit_cfg()
    built = build_circuit(n_qubits, clt_pairs,
                           feature_map_reps=int(cfg["feature_map_reps"]),
                           ansatz_reps=int(cfg["ansatz_reps"]),
                           entangling_gate=cfg["entangling_gate"])
    circuit = built["circuit"]

    observables = {
        "global": SparsePauliOp("Z" * n_qubits),
        "local_single": SparsePauliOp("I" * (n_qubits - 1) + "Z"),
        "local_sum": SparsePauliOp.from_list(
            [(("I" * i) + "Z" + ("I" * (n_qubits - 1 - i)), 1.0 / n_qubits)
             for i in range(n_qubits)]),
    }

    estimator, pass_manager = build_estimator(BackendSpec(name="ideal"))
    seeds = [42, 7, 13, 99, 2024]
    rows_out = []

    for obs_name, obs in observables.items():
        pr_aucs, final_objs, n_evals = [], [], []
        for seed in seeds:
            qnn = EstimatorQNN(circuit=circuit, observables=obs,
                                input_params=built["input_params"],
                                weight_params=built["weight_params"],
                                estimator=estimator, pass_manager=pass_manager)
            trace = []
            model = NeuralNetworkClassifier(
                neural_network=qnn,
                optimizer=_optimizer(cfg["optimizer"], cfg["maxiter"]),
                loss="squared_error",
                initial_point=initial_point(qnn.num_weights, seed),
                callback=lambda _w, v: trace.append(float(v)),
            )
            y_pm = (2 * ytr.astype(int) - 1).astype(float)
            t0 = time.perf_counter()
            model.fit(Xtr, y_pm)
            runtime = time.perf_counter() - t0

            raw = np.asarray(qnn.forward(Xva, model.weights), dtype=float).ravel()
            score = (raw + 1.0) / 2.0
            pr_auc = float(average_precision_score(yva, score))
            pr_aucs.append(pr_auc)
            final_objs.append(trace[-1] if trace else float("nan"))
            n_evals.append(len(trace))
            print(f"[{obs_name}] seed={seed} PR-AUC={pr_auc:.4f} "
                  f"final_obj={trace[-1] if trace else float('nan'):.4f} "
                  f"n_evals={len(trace)} runtime={runtime:.1f}s", flush=True)

        arr = np.array(pr_aucs)
        row = dict(observable=obs_name, qubits=n_qubits, topology="chow_liu",
                   n_seeds=len(seeds), pr_auc_mean=round(float(arr.mean()), 4),
                   pr_auc_std=round(float(arr.std()), 4),
                   pr_auc_min=round(float(arr.min()), 4),
                   pr_auc_max=round(float(arr.max()), 4),
                   mean_final_objective=round(float(np.nanmean(final_objs)), 4),
                   mean_n_evaluations=round(float(np.mean(n_evals)), 1))
        rows_out.append(row)
        print("SUMMARY:", row, flush=True)

    with open(OUT / "local_vs_global_observable_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("Wrote research/paper_driven_quantum/local_vs_global_observable_results.csv")


if __name__ == "__main__":
    main()
