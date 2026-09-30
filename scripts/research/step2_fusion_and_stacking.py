"""Steps 2-4: validation-selected fusion, classical stacking control, and
quantum-augmented stacking, on the locked controlled subset (n_features=4,
pipeline='selected', matching step1_error_complementarity.py exactly so rows
and feature space are identical).

Trains classical + quantum models fresh so both TRAIN and VALIDATION scores
are available (step1 only persisted validation scores). Logs every experiment
to research/quantum_contribution/experiment_log.csv (append) and writes the
full fusion grid to fusion_results.csv, plus stacking_results.csv.

Small-N caveat: with 250 train / 100 val rows, calibration and meta-learner
fitting are noisy. Platt scaling (logistic regression on the 1-D score) is
used rather than isotonic, since isotonic regression overfits badly at this
sample size. All fusion-weight and stacking-architecture selection uses
VALIDATION PR-AUC only; TEST is never touched here.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "layer1" / "quantum_contribution"
OUT.mkdir(parents=True, exist_ok=True)

LOG_COLUMNS = [
    "experiment_id", "timestamp", "architecture", "classical_component",
    "quantum_component", "feature_strategy", "qubits", "circuit_depth",
    "fusion_method", "seed", "validation_PR_AUC", "validation_precision",
    "validation_recall", "validation_F1", "validation_F2", "FPR", "runtime",
    "status", "promoted", "reason",
]


def log_experiment(row: dict):
    path = OUT / "experiment_log.csv"
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


def binary_metrics(y_true, score, threshold=0.5):
    from sklearn.metrics import average_precision_score, precision_score, recall_score, fbeta_score
    y_pred = (score >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, score)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = fbeta_score(y_true, y_pred, beta=1, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return dict(pr_auc=round(float(pr_auc), 4), precision=round(float(precision), 4),
                recall=round(float(recall), 4), f1=round(float(f1), 4),
                f2=round(float(f2), 4), fpr=round(float(fpr), 4))


def best_threshold(y, score):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 97):
        pred = (score >= t).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def platt_calibrate(train_score, train_y, apply_score):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(train_score.reshape(-1, 1), train_y)
    return lr.predict_proba(apply_score.reshape(-1, 1))[:, 1]


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.qml_experimental.data import build_subset, row_fingerprint
    from src.utils.config import get_seed
    from src.qml_experimental.backends import BackendSpec
    from src.qml_experimental.chow_liu.feature_structure import learn_feature_structure
    from src.qml_experimental.chow_liu.quantum_mapping import baseline_entangling_pairs
    from src.qml_experimental.models import train_qnn, train_vqc

    fp = row_fingerprint()
    assert fp == "2da40ffb169cd76b", f"fingerprint mismatch: {fp}"
    seed = get_seed()

    n_features = 4
    subset = build_subset("selected", n_features)
    Xtr, ytr = subset.X_train, subset.y_train
    Xva, yva = subset.X_val, subset.y_val

    # ---------------- classical models (train + val scores + OOF) ----------------
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from xgboost import XGBClassifier

    classical_defs = {
        "svm_rbf": lambda: SVC(kernel="rbf", probability=True, random_state=seed, class_weight="balanced"),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced"),
        "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced"),
        "xgboost": lambda: XGBClassifier(n_estimators=200, max_depth=4, random_state=seed,
                                          scale_pos_weight=(len(ytr) - ytr.sum()) / max(ytr.sum(), 1),
                                          eval_metric="logloss"),
    }

    classical_train_scores, classical_val_scores, classical_oof = {}, {}, {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for name, ctor in classical_defs.items():
        model = ctor()
        model.fit(Xtr, ytr)
        classical_train_scores[name] = model.predict_proba(Xtr)[:, 1]
        classical_val_scores[name] = model.predict_proba(Xva)[:, 1]
        classical_oof[name] = cross_val_predict(ctor(), Xtr, ytr, cv=skf, method="predict_proba")[:, 1]

    # ---------------- quantum models (train + val scores) ----------------
    backend = BackendSpec(name="ideal")
    generic_pairs = baseline_entangling_pairs(n_features, "linear")
    structure = learn_feature_structure(Xtr, subset.feature_names)
    clt_pairs = structure.tree.edges if hasattr(structure, "tree") else structure.edges

    quantum_specs = [("qnn", train_qnn, generic_pairs, "generic"),
                      ("vqc", train_vqc, generic_pairs, "generic"),
                      ("clt_qnn", train_qnn, clt_pairs, "chow_liu"),
                      ("clt_vqc", train_vqc, clt_pairs, "chow_liu")]

    quantum_train_scores, quantum_val_scores = {}, {}
    for name, fn, pairs, structure_tag in quantum_specs:
        try:
            out = fn(subset, entangling_pairs=pairs, backend=backend,
                      structure=structure_tag, init_seed=seed)
            v = out["val_score"]
            t = out.get("train_score")
            if t is None:
                # recompute train-set score using the fitted estimator directly
                est = out["estimator"]
                nn = est.neural_network
                raw_t = np.asarray(nn.forward(Xtr, est.weights), dtype=float)
                raw_v = np.asarray(nn.forward(Xva, est.weights), dtype=float)
                if raw_t.ndim == 2:
                    t = raw_t[:, 1] if raw_t.shape[1] == 2 else raw_t.ravel()
                    v = raw_v[:, 1] if raw_v.shape[1] == 2 else raw_v.ravel()
                else:
                    t = (raw_t.ravel() + 1.0) / 2.0
                    v = (raw_v.ravel() + 1.0) / 2.0
            quantum_train_scores[name] = np.asarray(t, dtype=float)
            quantum_val_scores[name] = np.asarray(v, dtype=float)
            print(name, "trained OK")
        except Exception as e:
            print(name, "FAILED:", e)

    np.savez(OUT / "step2_scores.npz", y_train=ytr, y_val=yva,
              **{f"ctr_{k}": v for k, v in classical_train_scores.items()},
              **{f"cva_{k}": v for k, v in classical_val_scores.items()},
              **{f"coof_{k}": v for k, v in classical_oof.items()},
              **{f"qtr_{k}": v for k, v in quantum_train_scores.items()},
              **{f"qva_{k}": v for k, v in quantum_val_scores.items()})

    # ================= STEP 2: validation-selected fusion =================
    fusion_rows = []
    alphas = np.round(np.arange(0.50, 0.951, 0.05), 2)
    for cname, cval in classical_val_scores.items():
        ctrain = classical_train_scores[cname]
        for qname, qval_raw in quantum_val_scores.items():
            qtrain_raw = quantum_train_scores[qname]
            q_cal_val = platt_calibrate(qtrain_raw, ytr, qval_raw)
            for alpha in alphas:
                fused = alpha * cval + (1 - alpha) * q_cal_val
                m = binary_metrics(yva, fused, threshold=best_threshold(yva, fused))
                fusion_rows.append(dict(classical=cname, quantum=qname, alpha=alpha,
                                         **m))
    with open(OUT / "fusion_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fusion_rows[0].keys()))
        w.writeheader()
        w.writerows(fusion_rows)

    # classical-only PR-AUC reference per pair, to compute deltas
    classical_solo = {c: binary_metrics(yva, v, threshold=best_threshold(yva, v))["pr_auc"]
                       for c, v in classical_val_scores.items()}

    best_per_pair = {}
    for row in fusion_rows:
        key = (row["classical"], row["quantum"])
        if key not in best_per_pair or row["pr_auc"] > best_per_pair[key]["pr_auc"]:
            best_per_pair[key] = row

    for (cname, qname), row in best_per_pair.items():
        delta = row["pr_auc"] - classical_solo[cname]
        log_experiment(dict(
            experiment_id=f"step2_fusion_{cname}_{qname}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            architecture=f"fusion_{cname}_{qname}", classical_component=cname,
            quantum_component=qname, feature_strategy="selected", qubits=n_features,
            circuit_depth="", fusion_method=f"platt+alpha{row['alpha']}", seed=seed,
            validation_PR_AUC=row["pr_auc"], validation_precision=row["precision"],
            validation_recall=row["recall"], validation_F1=row["f1"],
            validation_F2=row["f2"], FPR=row["fpr"], runtime="", status="success",
            promoted="no",
            reason=f"delta_vs_classical_solo={delta:+.4f}"))
        print(f"fusion {cname}+{qname}: best alpha={row['alpha']} PR-AUC={row['pr_auc']:.4f} "
              f"(classical solo={classical_solo[cname]:.4f}, delta={delta:+.4f})")

    # ================= STEP 3: classical stacking control =================
    from sklearn.linear_model import LogisticRegression as MetaLR

    clist = list(classical_defs.keys())
    X_meta_train = np.column_stack([classical_oof[c] for c in clist])
    X_meta_val = np.column_stack([classical_val_scores[c] for c in clist])
    meta = MetaLR(max_iter=2000)
    meta.fit(X_meta_train, ytr)
    stack_val_score = meta.predict_proba(X_meta_val)[:, 1]
    m_stack_classical = binary_metrics(yva, stack_val_score, threshold=best_threshold(yva, stack_val_score))
    log_experiment(dict(
        experiment_id="step3_classical_stack_control", timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        architecture="classical_stack", classical_component="+".join(clist),
        quantum_component="", feature_strategy="selected", qubits=n_features,
        circuit_depth="", fusion_method="oof_logistic_meta", seed=seed,
        validation_PR_AUC=m_stack_classical["pr_auc"], validation_precision=m_stack_classical["precision"],
        validation_recall=m_stack_classical["recall"], validation_F1=m_stack_classical["f1"],
        validation_F2=m_stack_classical["f2"], FPR=m_stack_classical["fpr"], runtime="",
        status="success", promoted="no", reason="control for step4"))
    print("classical stack control PR-AUC:", m_stack_classical["pr_auc"])

    # ================= STEP 4: quantum-augmented stacking =================
    stack_rows = []
    for qname in quantum_val_scores:
        X_meta_train_q = np.column_stack([classical_oof[c] for c in clist] + [quantum_train_scores[qname]])
        X_meta_val_q = np.column_stack([classical_val_scores[c] for c in clist] + [quantum_val_scores[qname]])
        meta_q = MetaLR(max_iter=2000)
        meta_q.fit(X_meta_train_q, ytr)
        sq = meta_q.predict_proba(X_meta_val_q)[:, 1]
        mq = binary_metrics(yva, sq, threshold=best_threshold(yva, sq))
        delta = mq["pr_auc"] - m_stack_classical["pr_auc"]
        stack_rows.append(dict(quantum=qname, **mq, delta_vs_classical_stack=round(delta, 4)))
        log_experiment(dict(
            experiment_id=f"step4_quantum_stack_{qname}", timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            architecture=f"quantum_stack_{qname}", classical_component="+".join(clist),
            quantum_component=qname, feature_strategy="selected", qubits=n_features,
            circuit_depth="", fusion_method="oof_logistic_meta+quantum_feature", seed=seed,
            validation_PR_AUC=mq["pr_auc"], validation_precision=mq["precision"],
            validation_recall=mq["recall"], validation_F1=mq["f1"], validation_F2=mq["f2"],
            FPR=mq["fpr"], runtime="", status="success",
            promoted="yes" if delta > 0 else "no",
            reason=f"delta_vs_classical_stack_control={delta:+.4f}"))
        print(f"quantum-augmented stack +{qname}: PR-AUC={mq['pr_auc']:.4f} "
              f"(control={m_stack_classical['pr_auc']:.4f}, delta={delta:+.4f})")

    with open(OUT / "stacking_results.csv", "w", newline="") as f:
        fieldnames = ["quantum", "pr_auc", "precision", "recall", "f1", "f2", "fpr",
                      "delta_vs_classical_stack"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(stack_rows)

    print("\nStep 2-4 complete.")


if __name__ == "__main__":
    main()
