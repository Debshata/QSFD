"""Tests for the Layer 1 -> Layer 2 integration contract and end-to-end pipeline."""
import pytest

from src.fraud_engine.schemas import FraudAssessment
from src.security.integration.schemas import SecurityContext
from src.security.policy.security_selector import (
    CategoricalPolicy, ContinuousPolicy, PolicyThresholds,
)
from src.security.integration.secure_transaction import generate_key_material, protect_transaction

PROFILE_ORDER = ["BASELINE", "PQC_ENHANCED", "HYBRID_MAXIMUM", "QKD_EXPERIMENTAL"]


def _fa(**overrides):
    base = dict(transaction_id="tx-1", fraud_probability=0.1, fraud_prediction=0,
                risk_score=10.0, risk_level="LOW", model_name="xgboost", model_version="v1",
                threshold=0.581)
    base.update(overrides)
    return FraudAssessment(**base)


def test_security_context_from_fraud_assessment_copies_real_fields():
    fa = _fa(fraud_probability=0.42, risk_level="MEDIUM", risk_score=42.0)
    ctx = SecurityContext.from_fraud_assessment(fa)
    assert ctx.fraud_probability == 0.42
    assert ctx.risk_level == "MEDIUM"
    assert ctx.model_name == "xgboost"
    assert ctx.quantum_signal is None  # never fabricated when not supplied


def test_security_context_rejects_invalid_probability():
    fa = _fa()
    with pytest.raises(ValueError):
        SecurityContext.from_fraud_assessment(fa, anomaly_score=1.5)  # out of [0,1]


def test_categorical_policy_never_weakens_low_risk_below_baseline():
    fa = _fa(risk_level="LOW")
    ctx = SecurityContext.from_fraud_assessment(fa)
    decision = CategoricalPolicy().decide(ctx)
    assert decision.profile == "BASELINE"
    # BASELINE still uses AES-256-GCM + a real key-establishment algorithm --
    # never "no encryption" or a weak cipher.
    assert decision.algorithms["symmetric"] == "AES-256-GCM"


def test_categorical_policy_high_risk_gets_stronger_profile_than_low():
    ctx_low = SecurityContext.from_fraud_assessment(_fa(risk_level="LOW"))
    ctx_high = SecurityContext.from_fraud_assessment(_fa(risk_level="HIGH", fraud_probability=0.9,
                                                           risk_score=90.0, fraud_prediction=1))
    policy = CategoricalPolicy()
    low_profile = policy.decide(ctx_low).profile
    high_profile = policy.decide(ctx_high).profile
    assert PROFILE_ORDER.index(high_profile) > PROFILE_ORDER.index(low_profile)


def test_continuous_policy_distinguishes_within_same_risk_level():
    """0.05 and 0.29 are both LOW under the categorical thresholds but should
    not be forced to an identical profile by the continuous policy alone."""
    thresholds = PolicyThresholds(continuous_pqc_threshold=0.20)
    policy = ContinuousPolicy(thresholds=thresholds)
    ctx_a = SecurityContext.from_fraud_assessment(_fa(fraud_probability=0.05))
    ctx_b = SecurityContext.from_fraud_assessment(_fa(fraud_probability=0.29))
    assert policy.decide(ctx_a).profile == "BASELINE"
    assert policy.decide(ctx_b).profile == "PQC_ENHANCED"


def test_continuous_policy_optional_anomaly_signal_can_escalate():
    thresholds = PolicyThresholds()
    policy_without = ContinuousPolicy(thresholds=thresholds, use_anomaly_signal=False)
    policy_with = ContinuousPolicy(thresholds=thresholds, use_anomaly_signal=True)
    fa = _fa(fraud_probability=0.05)  # would otherwise be BASELINE
    ctx = SecurityContext.from_fraud_assessment(fa, anomaly_score=0.95)
    assert policy_without.decide(ctx).profile == "BASELINE"
    assert policy_with.decide(ctx).profile != "BASELINE"


def test_continuous_policy_works_without_optional_quantum_signal():
    """The policy must run identically whether or not quantum signals exist --
    the production (classical-only) path must never require them."""
    policy = ContinuousPolicy(use_quantum_signal=True)
    fa = _fa(fraud_probability=0.5)
    ctx_no_quantum = SecurityContext.from_fraud_assessment(fa)
    decision = policy.decide(ctx_no_quantum)
    assert decision.profile == "PQC_ENHANCED"
    assert "classical_quantum_disagreement" not in decision.layer1_inputs_used


def test_elevated_sensitivity_never_lowers_profile():
    fa = _fa(risk_level="LOW")
    ctx_standard = SecurityContext.from_fraud_assessment(fa, transaction_sensitivity="standard")
    ctx_elevated = SecurityContext.from_fraud_assessment(fa, transaction_sensitivity="elevated")
    policy = CategoricalPolicy()
    standard_profile = policy.decide(ctx_standard).profile
    elevated_profile = policy.decide(ctx_elevated).profile
    assert PROFILE_ORDER.index(elevated_profile) >= PROFILE_ORDER.index(standard_profile)


def test_end_to_end_protected_transaction_baseline_profile():
    fa = _fa(risk_level="LOW")
    ctx = SecurityContext.from_fraud_assessment(fa)
    decision = CategoricalPolicy().decide(ctx)
    keys = generate_key_material()
    pt = protect_transaction(decision, b'{"amount": 10.0}', keys)
    assert pt.symmetric_algorithm == "AES-256-GCM"
    assert pt.ciphertext_size_bytes > 0
    assert pt.processing_ms["total_ms"] > 0
    assert pt.qkd_summary is None  # BASELINE never touches QKD


def test_end_to_end_protected_transaction_pqc_profile():
    fa = _fa(risk_level="MEDIUM", fraud_probability=0.5, risk_score=50.0)
    ctx = SecurityContext.from_fraud_assessment(fa)
    decision = CategoricalPolicy().decide(ctx)
    assert decision.profile == "PQC_ENHANCED"
    keys = generate_key_material()
    pt = protect_transaction(decision, b'{"amount": 500.0}', keys)
    assert pt.kem_algorithm == "ML-KEM-768"
    assert pt.signature_algorithm == "ML-DSA-65"


def test_protected_transaction_never_exposes_a_key():
    fa = _fa(risk_level="LOW")
    ctx = SecurityContext.from_fraud_assessment(fa)
    decision = CategoricalPolicy().decide(ctx)
    keys = generate_key_material()
    pt = protect_transaction(decision, b"payload", keys)
    d = pt.to_dict()
    serialized = str(d)
    # No raw key material anywhere in the audit-safe dict.
    assert "private_key" not in serialized
    assert "secret_key" not in serialized
    assert "decapsulation_key" not in serialized
