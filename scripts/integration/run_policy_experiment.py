"""Layer 1 -> Layer 2 integration experiment.

Implements exactly the protocol pre-specified in
docs/protocols/LAYER2_INTEGRATION_PROTOCOL.md, committed BEFORE this script
was run. Do not change policy formulations or thresholds after reading the
printed results.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "integration"
FIGURES = ROOT / "artifacts" / "figures" / "integration"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


def build_production_test_fold():
    """Rebuilds the exact production test fold using saved (not refit) artifacts."""
    from src.data.loaders import load_dataset
    from src.data.preprocessing import make_splits, transform
    from src.data.registry import available_datasets
    from src.utils.config import load_config

    ds_cfg = load_config("datasets")
    name = ds_cfg["primary_dataset"]
    rec = available_datasets()[name]
    loaded = load_dataset(name, rec["entry"], rec["file"])
    splits = make_splits(loaded.X, loaded.y, loaded.spec)

    preprocessor = joblib.load(ROOT / "models" / "classical" / "xgboost" / "preprocessor.joblib")
    Xte = transform(preprocessor, splits.X_test)
    yte = splits.y_test.to_numpy()
    return Xte, yte


def main():
    from src.fraud_engine.schemas import FraudAssessment
    from src.security.integration.schemas import SecurityContext
    from src.security.policy.security_selector import CategoricalPolicy, ContinuousPolicy, PolicyThresholds
    from src.security.integration.secure_transaction import generate_key_material, protect_transaction

    print("STARTED build_production_test_fold", time.strftime("%H:%M:%S"), flush=True)
    Xte, yte = build_production_test_fold()
    print(f"Production test fold: n={len(yte)}, fraud={int(yte.sum())}", flush=True)

    xgb_model = joblib.load(ROOT / "models" / "classical" / "xgboost" / "model.joblib")
    calibrator_path = ROOT / "models" / "classical" / "xgboost" / "calibrator.joblib"
    scorer = joblib.load(calibrator_path) if calibrator_path.exists() else xgb_model
    fraud_probability = scorer.predict_proba(Xte)[:, 1]

    xgb_metadata = json.load(open(ROOT / "models" / "classical" / "xgboost" / "metadata.json"))
    threshold = xgb_metadata.get("threshold", 0.5)
    import yaml
    risk_cfg = yaml.safe_load(open(ROOT / "config" / "risk.yaml"))["risk"]
    low_upper, med_upper = risk_cfg["low_upper_bound"], risk_cfg["medium_upper_bound"]

    def risk_level(p):
        if p <= low_upper:
            return "LOW"
        elif p <= med_upper:
            return "MEDIUM"
        return "HIGH"

    iso_model = joblib.load(ROOT / "models" / "classical" / "isolation_forest" / "model.joblib")
    iso_preprocessor = joblib.load(ROOT / "models" / "classical" / "isolation_forest" / "preprocessor.joblib")
    from src.data.preprocessing import transform as _transform
    from src.data.loaders import load_dataset as _load
    from src.data.preprocessing import make_splits as _splits
    from src.data.registry import available_datasets as _avail
    from src.utils.config import load_config as _cfg
    ds_cfg = _cfg("datasets")
    rec = _avail()[ds_cfg["primary_dataset"]]
    loaded = _load(ds_cfg["primary_dataset"], rec["entry"], rec["file"])
    splits_raw = _splits(loaded.X, loaded.y, loaded.spec)
    Xte_iso = _transform(iso_preprocessor, splits_raw.X_test)
    iso_raw = -iso_model.score_samples(Xte_iso)  # higher = more anomalous
    anomaly_score = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min())
    print("Isolation Forest anomaly scores computed", flush=True)

    n = len(yte)
    contexts = []
    for i in range(n):
        fa = FraudAssessment(
            transaction_id=f"test-{i}", fraud_probability=float(fraud_probability[i]),
            fraud_prediction=int(fraud_probability[i] >= threshold), risk_score=float(fraud_probability[i] * 100),
            risk_level=risk_level(fraud_probability[i]), model_name="xgboost", model_version="v1",
            threshold=threshold,
        )
        ctx = SecurityContext.from_fraud_assessment(fa, anomaly_score=float(anomaly_score[i]))
        contexts.append(ctx)
    print(f"Built {len(contexts)} SecurityContexts", flush=True)

    thresholds = PolicyThresholds(continuous_pqc_threshold=low_upper, continuous_hybrid_threshold=med_upper,
                                   continuous_qkd_threshold=0.90)
    policy_a = CategoricalPolicy()
    policy_b = ContinuousPolicy(thresholds=thresholds)
    policy_c = ContinuousPolicy(thresholds=thresholds, use_anomaly_signal=True)

    def evaluate_policy(policy, label):
        decisions = [policy.decide(ctx) for ctx in contexts]
        profiles = np.array([d.profile for d in decisions])
        escalated = profiles != "BASELINE"
        fraud_mask = yte.astype(bool)
        ser = float(escalated.mean())
        fwc = float(escalated[fraud_mask].mean()) if fraud_mask.sum() > 0 else float("nan")
        ber = float(escalated[~fraud_mask].mean())
        profile_counts = pd.Series(profiles).value_counts().to_dict()
        row = dict(policy=label, n=n, n_fraud=int(fraud_mask.sum()),
                   security_escalation_rate=round(ser, 4),
                   fraud_weighted_coverage=round(fwc, 4), benign_escalation_rate=round(ber, 4),
                   **{f"pct_{k}": round(100 * v / n, 2) for k, v in profile_counts.items()})
        print(row, flush=True)
        return row, decisions

    rows = []
    row_a, dec_a = evaluate_policy(policy_a, "A_categorical")
    row_b, dec_b = evaluate_policy(policy_b, "B_continuous")
    row_c, dec_c = evaluate_policy(policy_c, "C_anomaly_assisted")
    rows.extend([row_a, row_b, row_c])

    all_cols = set()
    for r in rows:
        all_cols.update(r.keys())
    fieldnames = ["policy", "n", "n_fraud", "security_escalation_rate", "fraud_weighted_coverage",
                  "benign_escalation_rate"] + sorted(c for c in all_cols if c.startswith("pct_"))
    with open(RESULTS / "policy_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, 0) for k in fieldnames})

    # --- Cryptographic latency on the pre-specified 200-row sub-sample ---
    print("\\nSTARTED cryptographic latency sub-sample", time.strftime("%H:%M:%S"), flush=True)
    rng = np.random.default_rng(42)
    fraud_idx = np.where(yte == 1)[0]
    legit_idx = rng.choice(np.where(yte == 0)[0], size=200 - len(fraud_idx), replace=False)
    sample_idx = np.concatenate([fraud_idx, legit_idx])
    keys = generate_key_material()

    latency_rows = []
    for i in sample_idx:
        decision = policy_b.decide(contexts[i])
        # Skip the ~15s SLH-DSA cost for the bulk sample -- measure only 5
        # HYBRID_MAXIMUM/QKD_EXPERIMENTAL cases (already benchmarked
        # separately and exhaustively in results/layer2/) to keep this
        # sub-sample's total runtime bounded, per the pre-specified compute budget.
        if decision.profile in ("HYBRID_MAXIMUM", "QKD_EXPERIMENTAL") and \
                sum(1 for r in latency_rows if r["profile"] in ("HYBRID_MAXIMUM", "QKD_EXPERIMENTAL")) >= 5:
            continue
        t0 = time.perf_counter()
        pt = protect_transaction(decision, b'{"amount":100.0}', keys, bb84_seed=1)
        latency_rows.append(dict(row=int(i), profile=decision.profile, fraud=int(yte[i]),
                                  total_security_ms=pt.processing_ms["total_ms"]))
    with open(RESULTS / "latency_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(latency_rows[0].keys()))
        w.writeheader(); w.writerows(latency_rows)
    lat_df = pd.DataFrame(latency_rows)
    print(lat_df.groupby("profile")["total_security_ms"].agg(["mean", "median", "count"]), flush=True)

    with open(RESULTS / "routing_decisions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "fraud_label", "fraud_probability", "risk_level", "anomaly_score",
                    "profile_A", "profile_B", "profile_C"])
        for i in range(n):
            w.writerow([i, int(yte[i]), round(float(fraud_probability[i]), 4), contexts[i].risk_level,
                        round(float(anomaly_score[i]), 4), dec_a[i].profile, dec_b[i].profile, dec_c[i].profile])

    print("\\nWrote policy_ablation.csv, latency_summary.csv, routing_decisions.csv")
    return dict(rows=rows, fraud_probability=fraud_probability, yte=yte, anomaly_score=anomaly_score)


if __name__ == "__main__":
    main()
