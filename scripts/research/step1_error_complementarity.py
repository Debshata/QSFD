"""Step 1: error complementarity between classical and quantum models.

Retrains classical (SVM-RBF, RandomForest, XGBoost, LogisticRegression) and
quantum (QNN, VQC, QSVM, CLT-QNN, CLT-VQC) models on the LOCKED controlled
subset (train/val rows only -- test is untouched here), scores VALIDATION
rows, and computes pairwise complementarity statistics.

Run: python scripts/research/step1_error_complementarity.py
"""
from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "quantum_contribution"
OUT.mkdir(parents=True, exist_ok=True)

LOG_COLUMNS = [
    "experiment_id", "timestamp", "architecture", "classical_component",
    "quantum_component", "feature_strategy", "qubits", "circuit_depth",
    "fusion_method", "seed", "validation_PR_AUC", "validation_precision",
    "validation_recall", "validation_F1", "validation_F2", "FPR", "runtime",
    "status", "promoted", "reason",
]


def log_experiment(path: Path, row: dict):
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


def log_negative(path: Path, row: dict):
    cols = ["hypothesis", "architecture", "result", "reason_rejected", "what_was_learned"]
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.qml_experimental.data import build_subset, row_fingerprint
    from src.utils.config import load_config, get_seed

    exp_log = OUT / "experiment_log.csv"
    neg_log = OUT / "negative_results.csv"

    fp = row_fingerprint()
    print("Row fingerprint:", fp)
    assert fp == "2da40ffb169cd76b", f"Fingerprint mismatch! got {fp}"

    n_features = 4  # cheapest qubit count; keeps QSVM O(n^2) fit tractable
    subset = build_subset("selected", n_features)
    print(subset.summary())

    Xtr, ytr = subset.X_train, subset.y_train
    Xva, yva = subset.X_val, subset.y_val

    results = {}  # name -> val_score (probability-like, higher = more fraud-like)

    # ---------------- classical models ----------------
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except Exception:
        has_xgb = False

    seed = get_seed()

    classical_models = {
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=seed, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced"),
    }
    if has_xgb:
        pos = max(1, int(ytr.sum()))
        neg = max(1, int(len(ytr) - ytr.sum()))
        classical_models["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=4, random_state=seed,
            scale_pos_weight=neg / pos, eval_metric="logloss")

    for name, model in classical_models.items():
        t0 = time.perf_counter()
        try:
            model.fit(Xtr, ytr)
            score = model.predict_proba(Xva)[:, 1]
            runtime = time.perf_counter() - t0
            results[name] = score
            status = "success"
            reason = ""
        except Exception as e:
            runtime = time.perf_counter() - t0
            status = "failed"
            reason = str(e)
            traceback.print_exc()
        _log_metrics(exp_log, name, "classical", name, "", "selected",
                     n_features, "", "", seed, results.get(name), yva, runtime, status, reason)

    # ---------------- quantum models ----------------
    from src.qml_experimental.backends import BackendSpec
    from src.qml_experimental.chow_liu.feature_structure import learn_feature_structure
    from src.qml_experimental.chow_liu.quantum_mapping import baseline_entangling_pairs
    from src.qml_experimental.models import train_qnn, train_vqc, train_qsvm

    backend = BackendSpec(name="ideal")
    generic_pairs = baseline_entangling_pairs(n_features, "linear")

    # Chow-Liu tree structure learned on TRAIN rows only.
    structure = learn_feature_structure(Xtr, subset.feature_names)
    clt_pairs = structure.tree.edges if hasattr(structure, "tree") else structure.edges

    quantum_specs = [
        ("qnn", train_qnn, generic_pairs, "generic"),
        ("vqc", train_vqc, generic_pairs, "generic"),
        ("qsvm", train_qsvm, generic_pairs, "generic"),
        ("clt_qnn", train_qnn, clt_pairs, "chow_liu"),
        ("clt_vqc", train_vqc, clt_pairs, "chow_liu"),
    ]

    for name, fn, pairs, structure_tag in quantum_specs:
        t0 = time.perf_counter()
        try:
            kwargs = dict(entangling_pairs=pairs, backend=backend, structure=structure_tag)
            if fn is not train_qsvm:
                kwargs["init_seed"] = seed
            out = fn(subset, **kwargs)
            score = out["val_score"]
            if not out.get("score_is_probability", False):
                # min-max normalise to [0,1] for ranking / calibration purposes
                lo, hi = score.min(), score.max()
                score = (score - lo) / (hi - lo) if hi > lo else np.full_like(score, 0.5)
            runtime = time.perf_counter() - t0
            results[name] = score
            status = "success"
            reason = ""
        except Exception as e:
            runtime = time.perf_counter() - t0
            status = "failed"
            reason = str(e)
            traceback.print_exc()
        _log_metrics(exp_log, name, "", name, name, "selected",
                     n_features, "", "", seed, results.get(name), yva, runtime, status, reason)

    # persist raw scores for downstream steps (fusion, stacking, etc.)
    np.savez(OUT / "step1_val_scores.npz", y_val=yva, **results)
    print("Models with usable validation scores:", sorted(results.keys()))

    if len(results) < 2:
        log_negative(neg_log, {
            "hypothesis": "Classical and quantum models make complementary errors",
            "architecture": "all",
            "result": "insufficient models trained successfully",
            "reason_rejected": "fewer than 2 models produced usable scores",
            "what_was_learned": "environment/training failures blocked complementarity analysis",
        })
        print("Not enough successful models to compute complementarity.")
        return

    # ---------------- complementarity analysis ----------------
    _compute_complementarity(results, yva)


def _log_metrics(path, name, classical_component, cc2, qc, feature_strategy,
                  qubits, depth, fusion, seed, score, y_val, runtime, status, reason):
    row = {
        "experiment_id": f"step1_{name}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "architecture": name,
        "classical_component": classical_component,
        "quantum_component": qc,
        "feature_strategy": feature_strategy,
        "qubits": qubits,
        "circuit_depth": depth,
        "fusion_method": fusion,
        "seed": seed,
        "runtime": round(runtime, 3),
        "status": status,
        "promoted": "no",
        "reason": reason,
    }
    if score is not None:
        m = _binary_metrics(y_val, score)
        row.update({
            "validation_PR_AUC": m["pr_auc"],
            "validation_precision": m["precision"],
            "validation_recall": m["recall"],
            "validation_F1": m["f1"],
            "validation_F2": m["f2"],
            "FPR": m["fpr"],
        })
    log_experiment(path, row)


def _binary_metrics(y_true, score, threshold=0.5):
    from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, fbeta_score
    y_pred = (score >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, score)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return dict(pr_auc=round(float(pr_auc), 4), precision=round(float(precision), 4),
                recall=round(float(recall), 4), f1=round(float(f1), 4),
                f2=round(float(f2), 4), fpr=round(float(fpr), 4))


def _compute_complementarity(results: dict, y_val: np.ndarray):
    classical = ["svm_rbf", "random_forest", "logistic_regression", "xgboost"]
    quantum = ["qnn", "vqc", "qsvm", "clt_qnn", "clt_vqc"]
    classical = [c for c in classical if c in results]
    quantum = [q for q in quantum if q in results]

    preds = {name: (score >= 0.5).astype(int) for name, score in results.items()}
    errors = {name: (preds[name] != y_val).astype(int) for name in results}

    rows = []
    matrix = {}
    for c in classical:
        matrix[c] = {}
        for q in quantum:
            pc, pq = preds[c], preds[q]
            ec, eq = errors[c], errors[q]
            n = len(y_val)
            disagreement = float(np.mean(pc != pq))
            double_fault = float(np.mean((ec == 1) & (eq == 1)))
            pred_corr = float(np.corrcoef(pc, pq)[0, 1]) if np.std(pc) > 0 and np.std(pq) > 0 else float("nan")
            err_corr = float(np.corrcoef(ec, eq)[0, 1]) if np.std(ec) > 0 and np.std(eq) > 0 else float("nan")
            # Q-statistic (Yule's Q) on correct/incorrect agreement
            n11 = np.sum((ec == 0) & (eq == 0))  # both correct
            n00 = np.sum((ec == 1) & (eq == 1))  # both wrong
            n10 = np.sum((ec == 0) & (eq == 1))  # c correct, q wrong
            n01 = np.sum((ec == 1) & (eq == 0))  # c wrong, q correct
            denom = (n11 * n00 + n10 * n01)
            q_stat = float((n11 * n00 - n10 * n01) / denom) if denom > 0 else float("nan")

            fraud_mask = y_val == 1
            both_fraud = int(np.sum((pc[fraud_mask] == 1) & (pq[fraud_mask] == 1)))
            classical_only_fraud = int(np.sum((pc[fraud_mask] == 1) & (pq[fraud_mask] == 0)))
            quantum_only_fraud = int(np.sum((pc[fraud_mask] == 0) & (pq[fraud_mask] == 1)))
            neither_fraud = int(np.sum((pc[fraud_mask] == 0) & (pq[fraud_mask] == 0)))

            legit_mask = y_val == 0
            both_fp = int(np.sum((pc[legit_mask] == 1) & (pq[legit_mask] == 1)))
            classical_only_fp = int(np.sum((pc[legit_mask] == 1) & (pq[legit_mask] == 0)))
            quantum_only_fp = int(np.sum((pc[legit_mask] == 0) & (pq[legit_mask] == 1)))

            row = dict(
                classical_model=c, quantum_model=q, n=n,
                prediction_correlation=round(pred_corr, 4) if pred_corr == pred_corr else "nan",
                error_correlation=round(err_corr, 4) if err_corr == err_corr else "nan",
                disagreement_rate=round(disagreement, 4),
                double_fault_rate=round(double_fault, 4),
                q_statistic=round(q_stat, 4) if q_stat == q_stat else "nan",
                frauds_caught_both=both_fraud,
                frauds_caught_classical_only=classical_only_fraud,
                frauds_caught_quantum_only=quantum_only_fraud,
                frauds_caught_neither=neither_fraud,
                false_positives_shared=both_fp,
                false_positives_classical_only=classical_only_fp,
                false_positives_quantum_only=quantum_only_fp,
            )
            rows.append(row)
            matrix[c][q] = row["error_correlation"]

    with open(OUT / "error_overlap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(OUT / "complementarity_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classical_model"] + quantum)
        for c in classical:
            w.writerow([c] + [matrix[c][q] for q in quantum])

    # narrative summary
    lines = ["# Complementarity Analysis (Step 1)\n",
             f"\nControlled validation subset: n={len(y_val)}, "
             f"frauds={int(y_val.sum())}. Row fingerprint 2da40ffb169cd76b.\n",
             "\nModels successfully trained: classical=" + str(classical) +
             ", quantum=" + str(quantum) + "\n",
             "\n## Pairwise error correlation (lower / more negative = more complementary)\n"]
    for c in classical:
        for q in quantum:
            lines.append(f"- {c} vs {q}: error_corr={matrix[c][q]}\n")

    any_negative = any(isinstance(matrix[c][q], (int, float)) and matrix[c][q] < -0.05
                        for c in classical for q in quantum)
    verdict = ("At least one classical-quantum pair shows negative error "
               "correlation (< -0.05), suggestive of complementary errors worth "
               "testing in fusion (step 2)." if any_negative else
               "No classical-quantum pair shows meaningfully negative error "
               "correlation on this small validation set (n=100). Errors are "
               "either positively correlated or near-independent (noise). "
               "This does NOT support a complementarity-driven fusion advantage "
               "a priori, but fusion (step 2) is still run per protocol to check "
               "empirically.")
    lines.append(f"\n## Verdict\n{verdict}\n")
    (OUT / "complementarity_analysis.md").write_text("".join(lines))
    print(verdict)


if __name__ == "__main__":
    main()
