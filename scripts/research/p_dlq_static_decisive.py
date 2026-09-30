"""DL+Quantum branch, static track: the decisive comparison.

Per the DL+quantum research brief, the purpose is NOT to make an ensemble
bigger -- it's to test:

    BEST CLASSICAL + DEEP LEARNING + QUANTUM  >  BEST CLASSICAL + DEEP LEARNING

on IDENTICAL rows. This script builds that comparison on the locked
controlled subset (fingerprint 2da40ffb169cd76b), using:

  - XGBoost as the classical component (already trained/cached per-row scores
    exist in step1_val_scores.npz for validation, but stacking needs
    out-of-fold TRAIN predictions too, which this script generates fresh via
    3-fold CV on the 250 locked TRAIN rows only)
  - sklearn MLPClassifier as the "deep learning" component (torch/tensorflow
    are heavier than justified for a 250-row dataset; a small MLP is the
    correct-scale deep-learning baseline here, and this scope limitation is
    stated explicitly rather than silently using a toy network that adds
    nothing over the MLP)
  - CLT-QNN as the quantum component (the only quantum model with a plausible
    structural motivation from the parallel research; using it here avoids
    re-litigating which quantum model to use)

All stacking meta-features are strictly out-of-fold on TRAIN; the meta-model
is fit on those OOF features + label, then applied to VALIDATION scores from
each base model refit on the FULL locked train set. The locked TEST rows are
not touched by this script.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "layer1" / "dl_quantum"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


def main():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    from src.qml_experimental.data import build_subset, row_fingerprint
    from src.utils.config import get_seed
    from src.qml_experimental.backends import BackendSpec
    from src.qml_experimental.chow_liu.feature_structure import learn_feature_structure
    from src.qml_experimental.chow_liu.quantum_mapping import build_qubit_mapping
    from src.qml_experimental.models import train_qnn

    fp = row_fingerprint()
    assert fp == "2da40ffb169cd76b", f"fingerprint mismatch: {fp}"
    seed = get_seed()
    backend = BackendSpec(name="ideal")

    subset = build_subset("selected", 4)
    Xtr, ytr = subset.X_train, subset.y_train
    Xva, yva = subset.X_val, subset.y_val
    names = subset.feature_names
    n = len(ytr)
    print(f"Locked subset: train={n}, val={len(yva)}, features={names}", flush=True)

    structure = learn_feature_structure(Xtr, names, seed=0)
    mapping = build_qubit_mapping(structure, "cz")
    clt_pairs = mapping.entangling_pairs
    print(f"CLT edges: {structure.tree.named_edges()}", flush=True)

    def fit_xgb(X, y):
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            eval_metric="logloss", random_state=seed, use_label_encoder=False)
        model.fit(X, y)
        return model

    def fit_mlp(X, y, scaler=None):
        if scaler is None:
            scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        model = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                               alpha=1e-3, max_iter=2000, random_state=seed,
                               early_stopping=True, n_iter_no_change=20)
        model.fit(Xs, y)
        return model, scaler

    # --- 3-fold OOF on TRAIN for xgboost, mlp, clt_qnn ---
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof_xgb = np.zeros(n)
    oof_mlp = np.zeros(n)
    oof_qnn = np.zeros(n)

    t_start = time.perf_counter()
    for fold_i, (tr_idx, ho_idx) in enumerate(skf.split(Xtr, ytr)):
        Xf_tr, yf_tr = Xtr[tr_idx], ytr[tr_idx]
        Xf_ho = Xtr[ho_idx]

        m_xgb = fit_xgb(Xf_tr, yf_tr)
        oof_xgb[ho_idx] = m_xgb.predict_proba(Xf_ho)[:, 1]

        m_mlp, scaler = fit_mlp(Xf_tr, yf_tr)
        oof_mlp[ho_idx] = m_mlp.predict_proba(scaler.transform(Xf_ho))[:, 1]

        import copy
        fold_subset = copy.copy(subset)
        fold_subset.X_train, fold_subset.y_train = Xf_tr, yf_tr
        fold_subset.X_val, fold_subset.y_val = Xf_ho, ytr[ho_idx]
        out = train_qnn(fold_subset, clt_pairs, backend=backend,
                         structure="clt_qnn_oof", init_seed=seed)
        oof_qnn[ho_idx] = out["val_score"]
        print(f"fold {fold_i} done ({time.perf_counter()-t_start:.1f}s elapsed)", flush=True)

    # --- Refit base models on FULL locked train, score VALIDATION ---
    xgb_full = fit_xgb(Xtr, ytr)
    val_xgb = xgb_full.predict_proba(Xva)[:, 1]

    mlp_full, scaler_full = fit_mlp(Xtr, ytr)
    val_mlp = mlp_full.predict_proba(scaler_full.transform(Xva))[:, 1]

    qnn_out = train_qnn(subset, clt_pairs, backend=backend, structure="clt_qnn_full",
                         init_seed=seed)
    val_qnn = qnn_out["val_score"]

    def pr_auc(y, s):
        return float(average_precision_score(y, s))

    print("Solo PR-AUC on validation:", flush=True)
    print(f"  XGBoost solo : {pr_auc(yva, val_xgb):.4f}", flush=True)
    print(f"  MLP solo     : {pr_auc(yva, val_mlp):.4f}", flush=True)
    print(f"  CLT-QNN solo : {pr_auc(yva, val_qnn):.4f}", flush=True)

    def stack_and_eval(oof_features: dict, val_features: dict, label: str):
        Xoof = np.column_stack([oof_features[k] for k in oof_features])
        Xvalm = np.column_stack([val_features[k] for k in val_features])
        meta = LogisticRegression(max_iter=2000)
        meta.fit(Xoof, ytr)
        val_pred = meta.predict_proba(Xvalm)[:, 1]
        score = pr_auc(yva, val_pred)
        print(f"  {label:35s}: PR-AUC={score:.4f}  (features: {list(oof_features.keys())})",
              flush=True)
        return score, val_pred

    print("\nEnsemble PR-AUC on validation (meta-learner fit on 3-fold OOF train features):",
          flush=True)
    results = {}
    results["A_xgboost_solo"] = pr_auc(yva, val_xgb)
    results["B_mlp_solo"] = pr_auc(yva, val_mlp)
    results["C_xgb_mlp"], _ = stack_and_eval(
        {"xgb": oof_xgb, "mlp": oof_mlp}, {"xgb": val_xgb, "mlp": val_mlp}, "C: XGBoost+MLP")
    results["F_xgb_qnn"], _ = stack_and_eval(
        {"xgb": oof_xgb, "qnn": oof_qnn}, {"xgb": val_xgb, "qnn": val_qnn}, "F: XGBoost+QNN(CLT)")
    results["G_xgb_mlp_qnn"], _ = stack_and_eval(
        {"xgb": oof_xgb, "mlp": oof_mlp, "qnn": oof_qnn},
        {"xgb": val_xgb, "mlp": val_mlp, "qnn": val_qnn},
        "G/K: XGBoost+MLP+QNN(CLT) [decisive]")

    decisive_delta = results["G_xgb_mlp_qnn"] - results["C_xgb_mlp"]
    print(f"\nDECISIVE COMPARISON: (XGBoost+MLP+CLT-QNN) - (XGBoost+MLP) = "
          f"{decisive_delta:+.4f} PR-AUC", flush=True)

    # Paired bootstrap on the same validation rows for the decisive comparison.
    rng = np.random.default_rng(seed)
    _, val_pred_C = stack_and_eval(
        {"xgb": oof_xgb, "mlp": oof_mlp}, {"xgb": val_xgb, "mlp": val_mlp}, "(recompute C)")
    _, val_pred_G = stack_and_eval(
        {"xgb": oof_xgb, "mlp": oof_mlp, "qnn": oof_qnn},
        {"xgb": val_xgb, "mlp": val_mlp, "qnn": val_qnn}, "(recompute G)")
    n_boot = 2000
    deltas = []
    idx_all = np.arange(len(yva))
    for _ in range(n_boot):
        bidx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if yva[bidx].sum() == 0 or yva[bidx].sum() == len(bidx):
            continue
        pr_c = average_precision_score(yva[bidx], val_pred_C[bidx])
        pr_g = average_precision_score(yva[bidx], val_pred_G[bidx])
        deltas.append(pr_g - pr_c)
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    print(f"Bootstrap 95% CI on decisive delta PR-AUC ({len(deltas)} resamples): "
          f"[{ci_lo:+.4f}, {ci_hi:+.4f}]  (point estimate {decisive_delta:+.4f})", flush=True)

    with open(OUT / "static_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["architecture", "val_pr_auc"])
        for k, v in results.items():
            w.writerow([k, round(v, 4)])

    with open(OUT / "DL_QUANTUM_RESULTS.md", "w") as f:
        f.write("# DL + Quantum Ensemble Results (static track)\n\n")
        f.write("Locked controlled subset, row fingerprint `2da40ffb169cd76b`. "
                "Meta-learner fit on 3-fold out-of-fold TRAIN predictions only; "
                "reported PR-AUC is on the locked VALIDATION rows (locked TEST untouched).\n\n")
        f.write("Deep-learning component is an sklearn MLPClassifier(32,16) -- "
                "explicitly scoped down from LSTM/GRU/torch given the ~250-row "
                "training set here (see temporal-track note below for why torch "
                "was reserved for a genuinely sequential dataset instead).\n\n")
        f.write("| Architecture | Validation PR-AUC |\n|---|---:|\n")
        for k, v in results.items():
            f.write(f"| {k} | {v:.4f} |\n")
        f.write(f"\n**Decisive comparison** (Classical+DL+Quantum vs Classical+DL): "
                f"delta = {decisive_delta:+.4f} PR-AUC, "
                f"bootstrap 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] "
                f"({'crosses zero -- not distinguishable from noise' if ci_lo < 0 < ci_hi else 'does not cross zero'}).\n")

    print("\nWrote research/dl_quantum/static_results.csv and DL_QUANTUM_RESULTS.md")


if __name__ == "__main__":
    main()
