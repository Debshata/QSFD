"""BB84 trial suite + Layer 1 -> Layer 2 end-to-end latency benchmark."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "layer2" / "results"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


def main():
    from src.security.qkd.bb84 import run_bb84
    from src.fraud_engine.schemas import FraudAssessment
    from src.security.integration.schemas import SecurityContext
    from src.security.policy.security_selector import CategoricalPolicy
    from src.security.integration.secure_transaction import generate_key_material, protect_transaction

    # --- BB84 trials ---
    conditions = [
        ("clean", {}), ("noise_1pct", dict(channel_noise=0.01)),
        ("noise_5pct", dict(channel_noise=0.05)), ("noise_10pct", dict(channel_noise=0.10)),
        ("eve_intercept_resend", dict(eve_intercept_resend=True)),
        ("eve_and_noise_2pct", dict(eve_intercept_resend=True, channel_noise=0.02)),
    ]
    raw_length = 2000
    n_trials = 30
    trial_rows = []
    for label, kwargs in conditions:
        for seed in range(n_trials):
            r = run_bb84(raw_length, seed=seed + 1000, **kwargs)
            trial_rows.append(dict(condition=label, seed=seed + 1000, raw_length=raw_length,
                                    sifted_length=r.sifted_length, qber=r.qber,
                                    key_accepted=r.key_accepted, final_key_length=r.final_key_length,
                                    eve_present=r.eve_present))
    with open(OUT / "bb84_trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
        w.writeheader(); w.writerows(trial_rows)

    summary_rows = []
    for label, _ in conditions:
        sub = [r for r in trial_rows if r["condition"] == label]
        qbers = np.array([r["qber"] for r in sub])
        sifted = np.array([r["sifted_length"] for r in sub])
        accepted = np.array([r["key_accepted"] for r in sub])
        summary_rows.append(dict(
            condition=label, n_trials=len(sub),
            mean_qber=round(float(qbers.mean()), 4), std_qber=round(float(qbers.std()), 4),
            mean_sifted_length=round(float(sifted.mean()), 1),
            sifted_ratio=round(float(sifted.mean() / raw_length), 4),
            detection_rate=round(float(1 - accepted.mean()), 4) if "eve" in label else None,
            key_acceptance_rate=round(float(accepted.mean()), 4),
        ))
        print(summary_rows[-1], flush=True)
    with open(OUT / "bb84_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print("Wrote bb84_trials.csv and bb84_summary.csv", flush=True)

    # --- End-to-end latency across security profiles ---
    keys = generate_key_material()
    policy = CategoricalPolicy()
    test_cases = [
        ("very_low_risk", dict(fraud_probability=0.02, fraud_prediction=0, risk_score=2.0, risk_level="LOW")),
        ("borderline_low_medium", dict(fraud_probability=0.29, fraud_prediction=0, risk_score=29.0, risk_level="LOW")),
        ("medium_risk", dict(fraud_probability=0.5, fraud_prediction=0, risk_score=50.0, risk_level="MEDIUM")),
        ("borderline_medium_high", dict(fraud_probability=0.69, fraud_prediction=1, risk_score=69.0, risk_level="MEDIUM")),
        ("high_risk", dict(fraud_probability=0.9, fraud_prediction=1, risk_score=90.0, risk_level="HIGH")),
    ]
    payload = b'{"transaction_id":"tx","amount":500.0,"merchant":"m"}'
    e2e_rows = []
    for label, overrides in test_cases:
        fa = FraudAssessment(transaction_id=f"tx-{label}", model_name="xgboost", model_version="v1",
                              threshold=0.581, **overrides)
        t0 = time.perf_counter()
        ctx = SecurityContext.from_fraud_assessment(fa)
        ctx_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        decision = policy.decide(ctx)
        policy_ms = (time.perf_counter() - t0) * 1000
        # Skip repeated SLH-DSA-heavy profiles for the timed average; measure once.
        pt = protect_transaction(decision, payload, keys, bb84_seed=1)
        e2e_rows.append(dict(
            case=label, fraud_probability=fa.fraud_probability, risk_level=fa.risk_level,
            selected_profile=decision.profile, context_build_ms=round(ctx_ms, 4),
            policy_decision_ms=round(policy_ms, 4),
            key_establishment_ms=round(pt.processing_ms["key_establishment_ms"], 4),
            signature_ms=round(pt.processing_ms["signature_ms"], 4),
            symmetric_encryption_ms=round(pt.processing_ms["symmetric_encryption_ms"], 4),
            total_security_ms=round(pt.processing_ms["total_ms"], 4),
            ciphertext_size_bytes=pt.ciphertext_size_bytes,
        ))
        print(e2e_rows[-1], flush=True)
    with open(OUT / "end_to_end_benchmarks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(e2e_rows[0].keys()))
        w.writeheader(); w.writerows(e2e_rows)
    print("Wrote end_to_end_benchmarks.csv", flush=True)


if __name__ == "__main__":
    main()
