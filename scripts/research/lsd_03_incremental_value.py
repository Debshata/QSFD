"""Local-Sum Decisive validation, step 3: incremental value to XGBoost.

Local-Sum QNN passed its 20-seed paired promotion gate (statistical_tests.json:
mean Delta PR-AUC +0.0536, bootstrap 95% CI [+0.0149,+0.0913] excludes zero).
This step asks the question that matters for the actual fraud system: does it
add anything to XGBoost, and is that different from just adding another
nonlinear feature?

Builds, on QML-M (TRAIN for fitting, CONFIRM_VAL for evaluation, the same
context used throughout this round):
  C0: XGBoost alone
  C1: XGBoost + Global-QNN score (OOF-generated)
  C2: XGBoost + Local-Sum-QNN score (OOF-generated)
  C3: XGBoost + classical nonlinear capacity control (a small MLP, same
      1-dimensional output as either QNN, trained on the identical 4
      reduced features) -- the mandatory control per the task brief,
      protecting against "another nonlinear feature helped" being
      mistaken for "quantum helped"

All meta-features are 3-fold out-of-fold on TRAIN; the logistic meta-learner
is fit on those OOF features, then applied to base-model scores from models
refit on the FULL TRAIN set, scored on CONFIRM_VAL (which has now been used
twice in this round: once for the frozen 20-seed observable comparison,
once here for the follow-on incremental-value question -- both pre-specified
by the task brief as legitimate reuses of a frozen split, not new
architecture search).

Uses a single representative seed (0) for the QNN components, consistent
with a "frozen architecture" decision -- this step is not re-running the
20-seed comparison, it's using the already-validated architecture once.
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
OUT = ROOT / "research" / "local_sum_decisive"
sys.path.insert(0, str(ROOT))

N_QUBITS = 4
SEED = 0


def make_observable(name, n_qubits):
    from qiskit.quantum_info import SparsePauliOp
    if name == "global":
        return SparsePauliOp("Z" * n_qubits)
    return SparsePauliOp.from_list(
        [(("I" * i) + "Z" + ("I" * (n_qubits - 1 - i)), 1.0 / n_qubits)
         for i in range(n_qubits)])


def train_qnn_score(Xtr, ytr, Xeval, observable_name, circuit, input_params, weight_params, seed):
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
    from qiskit_machine_learning import optimizers as opt
    from src.qml_experimental.models import initial_point
    from src.qml_experimental.backends import build_estimator, BackendSpec

    obs = make_observable(observable_name, circuit.num_qubits)
    estimator, pm = build_estimator(BackendSpec(name="ideal"))
    qnn = EstimatorQNN(circuit=circuit, observables=obs, input_params=input_params,
                        weight_params=weight_params, estimator=estimator, pass_manager=pm)
    init = initial_point(qnn.num_weights, seed)
    model = NeuralNetworkClassifier(neural_network=qnn, optimizer=opt.COBYLA(maxiter=60),
                                     loss="squared_error", initial_point=init)
    y_pm = (2 * ytr.astype(int) - 1).astype(float)
    model.fit(Xtr, y_pm)
    raw = np.asarray(qnn.forward(Xeval, model.weights), dtype=float).ravel()
    return (raw + 1.0) / 2.0


def main():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score, fbeta_score
    import xgboost as xgb
    from src.qml_experimental.circuits import build_circuit
    from src.qml_experimental.chow_liu.quantum_mapping import baseline_entangling_pairs

    data = np.load(QSC / "qml_m_context.npz", allow_pickle=True)
    Xtr_full, ytr = data["X_train"], data["y_train"]
    Xcv_full, ycv = data["X_confirm_val"], data["y_confirm_val"]
    Xtr, Xcv = Xtr_full[:, :N_QUBITS], Xcv_full[:, :N_QUBITS]
    n = len(ytr)
    print(f"TRAIN n={n} ({ytr.sum()} fraud), CONFIRM_VAL n={len(ycv)} ({ycv.sum()} fraud)", flush=True)

    pairs = baseline_entangling_pairs(N_QUBITS, "linear")
    built = build_circuit(N_QUBITS, pairs, feature_map_reps=1, ansatz_reps=2, entangling_gate="cz")

    def fit_xgb(X, y):
        m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                               eval_metric="logloss", random_state=SEED)
        m.fit(X, y)
        return m

    def fit_mlp(X, y):
        scaler = StandardScaler().fit(X)
        m = MLPClassifier(hidden_layer_sizes=(16,), activation="relu", alpha=1e-3,
                           max_iter=2000, random_state=SEED, early_stopping=True,
                           n_iter_no_change=20)
        m.fit(scaler.transform(X), y)
        return m, scaler

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof_xgb = np.zeros(n)
    oof_global = np.zeros(n)
    oof_local = np.zeros(n)
    oof_mlp = np.zeros(n)

    t_start = time.perf_counter()
    for fold_i, (tr_idx, ho_idx) in enumerate(skf.split(Xtr, ytr)):
        Xf_tr, yf_tr = Xtr[tr_idx], ytr[tr_idx]
        Xf_ho = Xtr[ho_idx]

        m_xgb = fit_xgb(Xf_tr, yf_tr)
        oof_xgb[ho_idx] = m_xgb.predict_proba(Xf_ho)[:, 1]

        oof_global[ho_idx] = train_qnn_score(Xf_tr, yf_tr, Xf_ho, "global", built["circuit"],
                                              built["input_params"], built["weight_params"], SEED)
        oof_local[ho_idx] = train_qnn_score(Xf_tr, yf_tr, Xf_ho, "local_sum", built["circuit"],
                                             built["input_params"], built["weight_params"], SEED)

        m_mlp, scaler = fit_mlp(Xf_tr, yf_tr)
        oof_mlp[ho_idx] = m_mlp.predict_proba(scaler.transform(Xf_ho))[:, 1]

        print(f"fold {fold_i} done ({time.perf_counter()-t_start:.1f}s elapsed)", flush=True)

    # Refit on FULL train, score CONFIRM_VAL.
    xgb_full = fit_xgb(Xtr, ytr)
    val_xgb = xgb_full.predict_proba(Xcv)[:, 1]
    val_global = train_qnn_score(Xtr, ytr, Xcv, "global", built["circuit"],
                                  built["input_params"], built["weight_params"], SEED)
    val_local = train_qnn_score(Xtr, ytr, Xcv, "local_sum", built["circuit"],
                                 built["input_params"], built["weight_params"], SEED)
    mlp_full, scaler_full = fit_mlp(Xtr, ytr)
    val_mlp = mlp_full.predict_proba(scaler_full.transform(Xcv))[:, 1]

    def pr_auc(y, s):
        return float(average_precision_score(y, s))

    print(f"\nSolo CONFIRM_VAL PR-AUC: XGBoost={pr_auc(ycv, val_xgb):.4f} "
          f"Global-QNN={pr_auc(ycv, val_global):.4f} Local-Sum-QNN={pr_auc(ycv, val_local):.4f} "
          f"MLP-control={pr_auc(ycv, val_mlp):.4f}", flush=True)

    def stack(oof_feats: dict, val_feats: dict):
        Xoof = np.column_stack([oof_feats[k] for k in oof_feats])
        Xval = np.column_stack([val_feats[k] for k in val_feats])
        meta = LogisticRegression(max_iter=2000)
        meta.fit(Xoof, ytr)
        val_pred = meta.predict_proba(Xval)[:, 1]
        return pr_auc(ycv, val_pred), val_pred

    c0, pred_c0 = stack({"xgb": oof_xgb}, {"xgb": val_xgb})
    c1, pred_c1 = stack({"xgb": oof_xgb, "q": oof_global}, {"xgb": val_xgb, "q": val_global})
    c2, pred_c2 = stack({"xgb": oof_xgb, "q": oof_local}, {"xgb": val_xgb, "q": val_local})
    c3, pred_c3 = stack({"xgb": oof_xgb, "q": oof_mlp}, {"xgb": val_xgb, "q": val_mlp})

    print(f"\nC0 XGBoost alone:                  PR-AUC={c0:.4f}", flush=True)
    print(f"C1 XGBoost + Global-QNN:           PR-AUC={c1:.4f}  (delta vs C0: {c1-c0:+.4f})", flush=True)
    print(f"C2 XGBoost + Local-Sum-QNN:        PR-AUC={c2:.4f}  (delta vs C0: {c2-c0:+.4f})", flush=True)
    print(f"C3 XGBoost + classical MLP control: PR-AUC={c3:.4f}  (delta vs C0: {c3-c0:+.4f})", flush=True)
    print(f"\nDECISIVE: C2 (Local-Sum) vs C3 (classical control): delta={c2-c3:+.4f}", flush=True)

    # Paired bootstrap on CONFIRM_VAL rows for the two decisive comparisons.
    rng = np.random.default_rng(SEED)
    n_boot = 5000
    idx_all = np.arange(len(ycv))

    def bootstrap(pred_a, pred_b, label):
        deltas = []
        for _ in range(n_boot):
            bidx = rng.choice(idx_all, size=len(idx_all), replace=True)
            if ycv[bidx].sum() == 0 or ycv[bidx].sum() == len(bidx):
                continue
            deltas.append(average_precision_score(ycv[bidx], pred_a[bidx])
                           - average_precision_score(ycv[bidx], pred_b[bidx]))
        deltas = np.array(deltas)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        point = average_precision_score(ycv, pred_a) - average_precision_score(ycv, pred_b)
        excludes_zero = bool(lo > 0 or hi < 0)
        print(f"{label}: point={point:+.4f} 95% CI=[{lo:+.4f}, {hi:+.4f}] "
              f"({'EXCLUDES zero' if excludes_zero else 'crosses zero'})", flush=True)
        return {"point": round(float(point), 4), "ci_low": round(float(lo), 4),
                "ci_high": round(float(hi), 4), "excludes_zero": excludes_zero}

    d_c2_vs_c0 = bootstrap(pred_c2, pred_c0, "C2 (Local-Sum) - C0 (XGBoost alone)")
    d_c2_vs_c3 = bootstrap(pred_c2, pred_c3, "C2 (Local-Sum) - C3 (classical MLP control)")

    result = {
        "solo_pr_auc": {"xgboost": round(pr_auc(ycv, val_xgb), 4),
                         "global_qnn": round(pr_auc(ycv, val_global), 4),
                         "local_sum_qnn": round(pr_auc(ycv, val_local), 4),
                         "classical_mlp_control": round(pr_auc(ycv, val_mlp), 4)},
        "C0_xgboost_alone": round(c0, 4),
        "C1_xgboost_plus_global_qnn": round(c1, 4),
        "C2_xgboost_plus_local_sum_qnn": round(c2, 4),
        "C3_xgboost_plus_classical_control": round(c3, 4),
        "bootstrap_C2_vs_C0": d_c2_vs_c0,
        "bootstrap_C2_vs_C3": d_c2_vs_c3,
        "incremental_quantum_value_verified": bool(
            d_c2_vs_c0["excludes_zero"] and d_c2_vs_c0["point"] > 0
            and d_c2_vs_c3["excludes_zero"] and d_c2_vs_c3["point"] > 0),
    }
    with open(OUT / "incremental_value.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["architecture", "pr_auc", "delta_vs_xgboost_alone"])
        w.writerow(["C0_xgboost_alone", round(c0, 4), 0.0])
        w.writerow(["C1_xgboost_plus_global_qnn", round(c1, 4), round(c1 - c0, 4)])
        w.writerow(["C2_xgboost_plus_local_sum_qnn", round(c2, 4), round(c2 - c0, 4)])
        w.writerow(["C3_xgboost_plus_classical_control", round(c3, 4), round(c3 - c0, 4)])
    with open(OUT / "incremental_value_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nWrote incremental_value.csv and incremental_value_result.json")


if __name__ == "__main__":
    main()
