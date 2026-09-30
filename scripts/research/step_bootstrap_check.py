"""Quick paired-bootstrap significance check on the validation set for the
most promising candidates from step2 (fusion, quantum-augmented stacking),
before deciding whether multi-seed confirmation / finalist promotion is
warranted. Uses the already-computed scores in step2_scores.npz -- no
retraining. 2000 resamples, same indices for both arms of each pair.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "layer1" / "quantum_contribution"

d = np.load(OUT / "step2_scores.npz")
yva = d["y_val"]
rng = np.random.default_rng(42)
n = len(yva)
N_BOOT = 2000

def pr_auc(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return average_precision_score(y, s)

def bootstrap_delta(score_a, score_b, y):
    deltas = []
    idx_all = np.arange(n)
    for _ in range(N_BOOT):
        idx = rng.choice(idx_all, size=n, replace=True)
        if y[idx].sum() == 0:
            continue
        pa = pr_auc(y[idx], score_a[idx])
        pb = pr_auc(y[idx], score_b[idx])
        deltas.append(pa - pb)
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)

print("=== Fusion: svm_rbf + clt_qnn (alpha=0.55) vs svm_rbf solo ===")
cval = d["cva_svm_rbf"]
qtr = d["qtr_clt_qnn"]; qva = d["qva_clt_qnn"]
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(qtr.reshape(-1,1), d["y_train"])
qcal = lr.predict_proba(qva.reshape(-1,1))[:,1]
fused = 0.55 * cval + 0.45 * qcal
mean, lo, hi = bootstrap_delta(fused, cval, yva)
print(f"delta PR-AUC (fused - classical_solo): mean={mean:+.4f} 95% CI=[{lo:+.4f}, {hi:+.4f}]")

print("\n=== Quantum-augmented stack (+clt_qnn) vs classical stack control ===")
clist = ["svm_rbf", "random_forest", "logistic_regression", "xgboost"]
X_meta_train = np.column_stack([d[f"coof_{c}"] for c in clist])
X_meta_val = np.column_stack([d[f"cva_{c}"] for c in clist])
meta = LogisticRegression(max_iter=2000).fit(X_meta_train, d["y_train"])
stack_control = meta.predict_proba(X_meta_val)[:, 1]

X_meta_train_q = np.column_stack([d[f"coof_{c}"] for c in clist] + [d["qtr_clt_qnn"]])
X_meta_val_q = np.column_stack([d[f"cva_{c}"] for c in clist] + [d["qva_clt_qnn"]])
meta_q = LogisticRegression(max_iter=2000).fit(X_meta_train_q, d["y_train"])
stack_quantum = meta_q.predict_proba(X_meta_val_q)[:, 1]

mean, lo, hi = bootstrap_delta(stack_quantum, stack_control, yva)
print(f"delta PR-AUC (quantum_stack - classical_stack): mean={mean:+.4f} 95% CI=[{lo:+.4f}, {hi:+.4f}]")

import csv
with open(OUT / "bootstrap_screening.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["comparison", "mean_delta_pr_auc", "ci_lo", "ci_hi", "n_val", "n_boot"])
    # recompute both for saving
    mean1, lo1, hi1 = bootstrap_delta(fused, cval, yva)
    w.writerow(["fusion_svm_clt_qnn_vs_svm_solo", round(mean1,4), round(lo1,4), round(hi1,4), n, N_BOOT])
    mean2, lo2, hi2 = bootstrap_delta(stack_quantum, stack_control, yva)
    w.writerow(["quantum_stack_clt_qnn_vs_classical_stack", round(mean2,4), round(lo2,4), round(hi2,4), n, N_BOOT])
print("\nSaved research/quantum_contribution/bootstrap_screening.csv")
