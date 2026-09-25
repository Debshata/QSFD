"""Quantum Signal Confirmation, step 3 (Track B): controlled variational ablation.

Runs A0-A3 on the QML-M development context (TRAIN + ARCH_VAL only, per the
confirmation protocol), 5 predetermined seeds each, 4 qubits, linear
topology (kept fixed and simple per the task's explicit instruction not to
reopen a topology search this round):

  A0  existing QNN (global Z^n observable, uniform(-pi,pi) init) -- the
      unmodified baseline, exactly as src/qml_experimental/models.py builds
      it today.
  A1  A0 + local-sum observable (mean of single-qubit Z_i) -- the
      Cerezo-motivated change already screened once on QML-S; confirmed
      here on the larger, independent QML-M context.
  A2  A0 + GENUINE data re-uploading (Perez-Salinas et al.): the encoding
      block and a trainable ansatz block are interleaved L=2 times --
      encode(x) -> ansatz_1(theta) -> encode(x) -> ansatz_2(theta) ->
      measure -- built here explicitly rather than reusing the existing
      feature_map_reps knob, which (verified by reading circuits.py) only
      repeats the ENCODING block back-to-back with no trainable layer
      between repetitions and is therefore NOT genuine re-uploading.
  A3  A0 + near-zero-angle initialization (an adapted version of Grant et
      al.'s identity-block strategy -- see paper_method_registry.md for why
      the literal identity-block method isn't a direct fit for this
      project's ansatz).

A4-A7 (combinations) are explicitly NOT run this round unless A1-A3 each
show a real, defensible individual signal on this larger context --
combining unconfirmed components would repeat the exact premature-
architecture-stacking mistake this project's protocol is designed to avoid.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "quantum_signal_confirmation"
sys.path.insert(0, str(ROOT))

SEEDS = [42, 7, 13, 99, 2024]
N_QUBITS = 4


def build_reuploading_circuit(n_qubits: int, pairs, n_layers: int = 2):
    """encode(x) -> ansatz_l(theta_l) -> encode(x) -> ... -> measure."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from src.qml_experimental.circuits import build_feature_encoding, build_variational_ansatz

    x_all = ParameterVector("x", n_qubits)
    theta_all = []
    qc = QuantumCircuit(n_qubits)
    for layer in range(n_layers):
        fm, x_layer = build_feature_encoding(n_qubits, pairs, reps=1, entangling_gate="cz")
        fm = fm.assign_parameters({x_layer[i]: x_all[i] for i in range(n_qubits)})
        qc.compose(fm, inplace=True)

        ansatz, theta_layer = build_variational_ansatz(n_qubits, pairs, reps=1,
                                                         entangling_gate="cz")
        theta_layer_named = ParameterVector(f"theta_l{layer}", len(theta_layer))
        ansatz = ansatz.assign_parameters(
            {theta_layer[i]: theta_layer_named[i] for i in range(len(theta_layer))})
        qc.compose(ansatz, inplace=True)
        theta_all.extend(list(theta_layer_named))
    return qc, list(x_all), theta_all


def train_and_score(circuit, input_params, weight_params, Xtr, ytr, Xva, yva,
                     *, observable_name, init_seed, maxiter=60):
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
    from qiskit_machine_learning import optimizers as opt
    from sklearn.metrics import average_precision_score
    from src.qml_experimental.models import initial_point
    from src.qml_experimental.backends import build_estimator, BackendSpec

    n_qubits = circuit.num_qubits
    if observable_name == "global":
        obs = SparsePauliOp("Z" * n_qubits)
    elif observable_name == "local_sum":
        obs = SparsePauliOp.from_list(
            [(("I" * i) + "Z" + ("I" * (n_qubits - 1 - i)), 1.0 / n_qubits)
             for i in range(n_qubits)])
    else:
        raise ValueError(observable_name)

    estimator, pass_manager = build_estimator(BackendSpec(name="ideal"))
    qnn = EstimatorQNN(circuit=circuit, observables=obs, input_params=input_params,
                        weight_params=weight_params, estimator=estimator,
                        pass_manager=pass_manager)

    init = initial_point(qnn.num_weights, init_seed)
    trace = []
    model = NeuralNetworkClassifier(
        neural_network=qnn, optimizer=opt.COBYLA(maxiter=maxiter),
        loss="squared_error", initial_point=init,
        callback=lambda _w, v: trace.append(float(v)))

    y_pm = (2 * ytr.astype(int) - 1).astype(float)
    t0 = time.perf_counter()
    model.fit(Xtr, y_pm)
    runtime = time.perf_counter() - t0

    raw = np.asarray(qnn.forward(Xva, model.weights), dtype=float).ravel()
    score = (raw + 1.0) / 2.0
    pr_auc = float(average_precision_score(yva, score))
    return pr_auc, runtime, trace


def near_zero_init(n_params: int, seed: int) -> np.ndarray:
    """Adapted Grant-et-al.-motivated initialization: small angles, not full range."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.1, 0.1, size=n_params)


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

    data = np.load(OUT / "qml_m_context.npz", allow_pickle=True)
    Xtr_full, ytr = data["X_train"], data["y_train"]
    Xav_full, yav = data["X_arch_val"], data["y_arch_val"]
    # Use the first N_QUBITS of the 6 MI-selected features for this cheaper
    # 4-qubit ablation (consistent feature ORDER, not a re-selection).
    Xtr, Xav = Xtr_full[:, :N_QUBITS], Xav_full[:, :N_QUBITS]
    print(f"QML-M ablation context: train={len(ytr)} arch_val={len(yav)} qubits={N_QUBITS}",
          flush=True)

    pairs = baseline_entangling_pairs(N_QUBITS, "linear")
    baseline_built = build_circuit(N_QUBITS, pairs, feature_map_reps=1, ansatz_reps=2,
                                    entangling_gate="cz")
    reup_circuit, reup_x, reup_theta = build_reuploading_circuit(N_QUBITS, pairs, n_layers=2)

    rows = []

    def run_arm(arm_name, circuit, input_params, weight_params, observable_name, init_mode):
        prs, runtimes = [], []
        for seed in SEEDS:
            n_qubits = circuit.num_qubits
            if observable_name == "global":
                obs = SparsePauliOp("Z" * n_qubits)
            else:
                obs = SparsePauliOp.from_list(
                    [(("I" * i) + "Z" + ("I" * (n_qubits - 1 - i)), 1.0 / n_qubits)
                     for i in range(n_qubits)])
            estimator, pm = build_estimator(BackendSpec(name="ideal"))
            qnn = EstimatorQNN(circuit=circuit, observables=obs, input_params=input_params,
                                weight_params=weight_params, estimator=estimator, pass_manager=pm)
            init = (initial_point(qnn.num_weights, seed) if init_mode == "standard"
                    else near_zero_init(qnn.num_weights, seed))
            trace = []
            model = NeuralNetworkClassifier(
                neural_network=qnn, optimizer=opt.COBYLA(maxiter=60), loss="squared_error",
                initial_point=init, callback=lambda _w, v: trace.append(float(v)))
            y_pm = (2 * ytr.astype(int) - 1).astype(float)
            t0 = time.perf_counter()
            model.fit(Xtr, y_pm)
            runtime = time.perf_counter() - t0
            raw = np.asarray(qnn.forward(Xav, model.weights), dtype=float).ravel()
            score = (raw + 1.0) / 2.0
            pr_auc = float(average_precision_score(yav, score))
            prs.append(pr_auc); runtimes.append(runtime)
            print(f"[{arm_name}] seed={seed} PR-AUC={pr_auc:.4f} runtime={runtime:.1f}s",
                  flush=True)
        arr = np.array(prs)
        row = dict(arm=arm_name, n_seeds=len(SEEDS), pr_auc_mean=round(float(arr.mean()), 4),
                   pr_auc_std=round(float(arr.std()), 4), pr_auc_min=round(float(arr.min()), 4),
                   pr_auc_max=round(float(arr.max()), 4),
                   mean_runtime=round(float(np.mean(runtimes)), 1))
        rows.append(row)
        print("SUMMARY:", row, flush=True)
        return arr

    a0 = run_arm("A0_baseline_global_obs", baseline_built["circuit"], baseline_built["input_params"],
                 baseline_built["weight_params"], "global", "standard")
    a1 = run_arm("A1_local_sum_observable", baseline_built["circuit"], baseline_built["input_params"],
                 baseline_built["weight_params"], "local_sum", "standard")
    a2 = run_arm("A2_data_reuploading", reup_circuit, reup_x, reup_theta, "global", "standard")
    a3 = run_arm("A3_near_zero_init", baseline_built["circuit"], baseline_built["input_params"],
                 baseline_built["weight_params"], "global", "near_zero")

    with open(OUT / "confirmation_leaderboard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Ablation deltas vs A0.
    abl_rows = []
    for name, arr in (("A1_local_sum_observable", a1), ("A2_data_reuploading", a2),
                       ("A3_near_zero_init", a3)):
        delta_mean = float(arr.mean() - a0.mean())
        delta_std = float(arr.std() - a0.std())
        abl_rows.append(dict(baseline="A0_baseline_global_obs", component_added=name,
                              pr_auc_delta_mean=round(delta_mean, 4),
                              seed_std_delta=round(delta_std, 4),
                              a0_mean=round(float(a0.mean()), 4), a0_std=round(float(a0.std()), 4),
                              variant_mean=round(float(arr.mean()), 4),
                              variant_std=round(float(arr.std()), 4)))
    with open(OUT / "ablation_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(abl_rows[0].keys()))
        w.writeheader()
        w.writerows(abl_rows)

    print("Wrote confirmation_leaderboard.csv and ablation_results.csv")


if __name__ == "__main__":
    main()
