"""Risk-adaptive security policy engine.

Core design rule (per the integration brief): LOW fraud risk never implies
weak cryptography. Every transaction gets the BASELINE profile's full
AES-256-GCM + standards-based key establishment at minimum; risk only
controls ADDITIONAL protection layered on top (stronger PQC parameter
choices, hybrid classical+PQC key establishment, an additional PQC
signature, or -- for the experimental tier only -- a QKD-derived key
component).

Two policy families are implemented, per the task's required ablation:

  CategoricalPolicy  -- routes on FraudAssessment.risk_level (LOW/MEDIUM/HIGH)
                        only, the simplest possible policy.
  ContinuousPolicy   -- routes on the continuous fraud_probability, so e.g.
                        0.29 and 0.01 (both LOW) are NOT treated identically.

Fraud risk and cryptographic quantum-threat posture are deliberately kept as
SEPARATE inputs (see SecurityContext.transaction_sensitivity and
quantum_safe_mode) -- fraud probability informs operational escalation
(more verification, more audit, stronger key establishment), it does not by
itself determine "this transaction is quantum-vulnerable."
"""
from __future__ import annotations

from dataclasses import dataclass

from src.security.integration.schemas import SecurityContext, SecurityDecision, SecurityProfileName

POLICY_VERSION = "adaptive_security_v1"


ALGORITHMS_BY_PROFILE: dict[SecurityProfileName, dict[str, str]] = {
    "BASELINE": {
        "key_establishment": "ECDH-P256",
        "symmetric": "AES-256-GCM",
        "signature": "ECDSA-P256",
    },
    "PQC_ENHANCED": {
        "key_establishment": "ML-KEM-768",
        "symmetric": "AES-256-GCM",
        "signature": "ML-DSA-65",
    },
    "HYBRID_MAXIMUM": {
        "key_establishment": "ECDH-P256 + ML-KEM-768 (hybrid KDF)",
        "symmetric": "AES-256-GCM",
        "signature": "ML-DSA-65 + SLH-DSA-SHA2-128s (dual signature)",
    },
    "QKD_EXPERIMENTAL": {
        "key_establishment": "ECDH-P256 + ML-KEM-768 (hybrid KDF) + BB84-simulated key material",
        "symmetric": "AES-256-GCM",
        "signature": "ML-DSA-65 + SLH-DSA-SHA2-128s (dual signature)",
    },
}


@dataclass
class PolicyThresholds:
    """Configurable, versioned routing thresholds -- never tuned on locked test data."""

    # Categorical policy: risk_level -> profile.
    categorical_map: dict[str, SecurityProfileName] = None  # type: ignore[assignment]

    # Continuous policy: fraud_probability breakpoints (monotonic, in [0,1]).
    continuous_pqc_threshold: float = 0.30
    continuous_hybrid_threshold: float = 0.70
    continuous_qkd_threshold: float = 0.90

    # Cross-cutting escalation independent of fraud probability.
    escalate_on_elevated_sensitivity: bool = True
    disagreement_escalation_threshold: float = 0.30  # only used if classical_quantum_disagreement given

    def __post_init__(self):
        if self.categorical_map is None:
            self.categorical_map = {"LOW": "BASELINE", "MEDIUM": "PQC_ENHANCED", "HIGH": "HYBRID_MAXIMUM"}


def _algorithms_for(profile: SecurityProfileName) -> dict[str, str]:
    return dict(ALGORITHMS_BY_PROFILE[profile])


def _escalate_for_sensitivity(profile: SecurityProfileName, ctx: SecurityContext,
                               thresholds: PolicyThresholds) -> SecurityProfileName:
    """Elevated transaction_sensitivity never lowers the profile, only raises it."""
    order: list[SecurityProfileName] = ["BASELINE", "PQC_ENHANCED", "HYBRID_MAXIMUM", "QKD_EXPERIMENTAL"]
    if thresholds.escalate_on_elevated_sensitivity and ctx.transaction_sensitivity == "elevated":
        idx = order.index(profile)
        profile = order[min(idx + 1, len(order) - 1)]
    return profile


class CategoricalPolicy:
    """SYSTEM A: routes purely on FraudAssessment.risk_level."""

    version = f"{POLICY_VERSION}-categorical"

    def __init__(self, thresholds: PolicyThresholds | None = None):
        self.thresholds = thresholds or PolicyThresholds()

    def decide(self, ctx: SecurityContext) -> SecurityDecision:
        ctx.validate()
        profile = self.thresholds.categorical_map[ctx.risk_level]
        profile = _escalate_for_sensitivity(profile, ctx, self.thresholds)
        reason = (f"categorical routing: risk_level={ctx.risk_level} -> {profile}"
                  + (" (escalated: elevated transaction_sensitivity)"
                     if ctx.transaction_sensitivity == "elevated" else ""))
        return SecurityDecision(
            transaction_id=ctx.transaction_id, profile=profile, reason=reason,
            policy_version=self.version,
            layer1_inputs_used={"risk_level": ctx.risk_level,
                                "transaction_sensitivity": ctx.transaction_sensitivity},
            algorithms=_algorithms_for(profile),
        )


class ContinuousPolicy:
    """SYSTEM B: routes on the continuous fraud_probability.

    Optionally consumes anomaly_score / quantum_signal / disagreement when
    present on the SecurityContext (SYSTEM C / D of the required ablation) --
    all three are None-safe: the policy runs identically to a pure
    probability-based route when they are absent.
    """

    version = f"{POLICY_VERSION}-continuous"

    def __init__(self, thresholds: PolicyThresholds | None = None, *,
                 use_anomaly_signal: bool = False, use_quantum_signal: bool = False):
        self.thresholds = thresholds or PolicyThresholds()
        self.use_anomaly_signal = use_anomaly_signal
        self.use_quantum_signal = use_quantum_signal

    def decide(self, ctx: SecurityContext) -> SecurityDecision:
        ctx.validate()
        t = self.thresholds
        p = ctx.fraud_probability

        if p >= t.continuous_qkd_threshold:
            profile: SecurityProfileName = "QKD_EXPERIMENTAL"
        elif p >= t.continuous_hybrid_threshold:
            profile = "HYBRID_MAXIMUM"
        elif p >= t.continuous_pqc_threshold:
            profile = "PQC_ENHANCED"
        else:
            profile = "BASELINE"

        inputs_used = {"fraud_probability": p}
        escalated_reason = ""

        if self.use_anomaly_signal and ctx.anomaly_score is not None:
            inputs_used["anomaly_score"] = ctx.anomaly_score
            if ctx.anomaly_score >= 0.8 and profile == "BASELINE":
                profile = "PQC_ENHANCED"
                escalated_reason += " (escalated: high anomaly_score)"

        if self.use_quantum_signal and ctx.classical_quantum_disagreement is not None:
            inputs_used["classical_quantum_disagreement"] = ctx.classical_quantum_disagreement
            if (ctx.classical_quantum_disagreement >= t.disagreement_escalation_threshold
                    and profile in ("BASELINE", "PQC_ENHANCED")):
                order: list[SecurityProfileName] = ["BASELINE", "PQC_ENHANCED", "HYBRID_MAXIMUM",
                                                     "QKD_EXPERIMENTAL"]
                profile = order[min(order.index(profile) + 1, len(order) - 1)]
                escalated_reason += " (escalated: high classical/quantum disagreement)"

        profile = _escalate_for_sensitivity(profile, ctx, t)
        reason = f"continuous routing: fraud_probability={p:.4f} -> {profile}{escalated_reason}"
        return SecurityDecision(
            transaction_id=ctx.transaction_id, profile=profile, reason=reason,
            policy_version=self.version, layer1_inputs_used=inputs_used,
            algorithms=_algorithms_for(profile),
        )


__all__ = ["PolicyThresholds", "CategoricalPolicy", "ContinuousPolicy", "ALGORITHMS_BY_PROFILE",
           "POLICY_VERSION"]
