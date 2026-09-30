"""Layer 1 -> Layer 2 integration contract.

    FraudAssessment (Layer 1, src/fraud_engine/schemas.py -- UNCHANGED)
            |
            v
    SecurityContext   -- constructed here, from real FraudAssessment fields only
            |
            v
    SecurityPolicyEngine (src/security/policy/security_selector.py)
            |
            v
    SecurityDecision  -- the chosen profile + the reasoning, fully auditable
            |
            v
    ProtectedTransaction -- safe metadata only, never a secret key

Design rule enforced throughout this module (per the integration brief):
Layer 2 code must never branch on which Layer 1 model produced a score
(no "if model_name == 'xgboost'"). It consumes only the fields defined on
FraudAssessment and SecurityContext, so a future Layer 1 model swap requires
no Layer 2 change.

Fraud risk and cryptographic quantum-threat posture are two different risk
dimensions (see research/integration/LAYER1_LAYER2_INTEGRATION.md). This
module keeps them as separate constructor inputs to SecurityContext rather
than deriving one from the other.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

SecurityProfileName = Literal["BASELINE", "PQC_ENHANCED", "HYBRID_MAXIMUM", "QKD_EXPERIMENTAL"]


@dataclass
class SecurityContext:
    """Everything the security policy engine is allowed to see.

    Every field here is either copied directly from a real FraudAssessment
    (src/fraud_engine/schemas.py) or is an explicitly-labelled, optional
    research signal. Nothing is fabricated: quantum_signal/quantum_disagreement
    are None whenever the caller did not supply them, and the policy engine
    must work correctly with them absent (see policy/security_selector.py).
    """

    transaction_id: str
    fraud_probability: float           # FraudAssessment.fraud_probability, continuous, NOT just the label
    fraud_prediction: int              # FraudAssessment.fraud_prediction
    risk_score: float                  # FraudAssessment.risk_score
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]  # FraudAssessment.risk_level
    model_name: str                    # FraudAssessment.model_name -- audit only, never branched on
    model_version: str                 # FraudAssessment.model_version
    threshold: float                   # FraudAssessment.threshold
    top_contributing_features: list[dict[str, Any]] = field(default_factory=list)

    # Optional research signals. All None unless the caller explicitly
    # supplies them (e.g. from an experimental anomaly detector or the
    # Local-Sum QNN auxiliary score). Never required for the policy engine
    # to run -- this is what keeps the quantum path a clean, ablatable
    # addition rather than a hard dependency.
    anomaly_score: float | None = None                # e.g. Isolation Forest, higher = more anomalous
    quantum_signal: float | None = None                # e.g. Local-Sum QNN auxiliary score, [0,1]
    classical_quantum_disagreement: float | None = None  # |p_classical - p_quantum|, [0,1]

    # Separate risk dimension: NOT derived from fraud_probability. Set by
    # the caller / system configuration, defaults to a neutral "unknown"
    # value that the default policy treats as ordinary sensitivity. This is
    # a controlled, explicitly-synthetic policy attribute when no real
    # transaction-sensitivity field exists in the source dataset -- see
    # research/integration/LAYER1_LAYER2_INTEGRATION.md for why it is kept
    # separate from fraud risk rather than inferred from it.
    transaction_sensitivity: Literal["standard", "elevated"] = "standard"
    quantum_safe_mode: bool = False   # system-wide switch: has this deployment opted into PQC-by-default?

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def validate(self) -> None:
        if not 0.0 <= self.fraud_probability <= 1.0:
            raise ValueError(f"fraud_probability out of range: {self.fraud_probability}")
        if self.fraud_prediction not in (0, 1):
            raise ValueError(f"fraud_prediction must be 0 or 1: {self.fraud_prediction}")
        if self.risk_level not in ("LOW", "MEDIUM", "HIGH"):
            raise ValueError(f"invalid risk_level: {self.risk_level}")
        if not 0.0 <= self.risk_score <= 100.0:
            raise ValueError(f"risk_score out of range: {self.risk_score}")
        for name, val in (("anomaly_score", self.anomaly_score),
                          ("quantum_signal", self.quantum_signal),
                          ("classical_quantum_disagreement", self.classical_quantum_disagreement)):
            if val is not None and not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} out of range [0,1]: {val}")
        if self.transaction_sensitivity not in ("standard", "elevated"):
            raise ValueError(f"invalid transaction_sensitivity: {self.transaction_sensitivity}")

    @classmethod
    def from_fraud_assessment(cls, assessment: Any, *, anomaly_score: float | None = None,
                               quantum_signal: float | None = None,
                               classical_quantum_disagreement: float | None = None,
                               transaction_sensitivity: str = "standard",
                               quantum_safe_mode: bool = False) -> "SecurityContext":
        """Build a SecurityContext from a real FraudAssessment instance.

        Reads only attributes that exist on FraudAssessment
        (src/fraud_engine/schemas.py) -- no field is invented here. Optional
        research signals are keyword-only and default to None/"standard" so
        the production (classical-only) call site never has to know about
        experimental quantum inputs.
        """
        ctx = cls(
            transaction_id=assessment.transaction_id,
            fraud_probability=assessment.fraud_probability,
            fraud_prediction=assessment.fraud_prediction,
            risk_score=assessment.risk_score,
            risk_level=assessment.risk_level,
            model_name=assessment.model_name,
            model_version=assessment.model_version,
            threshold=assessment.threshold,
            top_contributing_features=list(assessment.top_contributing_features),
            anomaly_score=anomaly_score,
            quantum_signal=quantum_signal,
            classical_quantum_disagreement=classical_quantum_disagreement,
            transaction_sensitivity=transaction_sensitivity,
            quantum_safe_mode=quantum_safe_mode,
        )
        ctx.validate()
        return ctx

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SecurityDecision:
    """The output of the policy engine -- fully auditable, never a secret."""

    transaction_id: str
    profile: SecurityProfileName
    reason: str
    policy_version: str
    layer1_inputs_used: dict[str, Any]
    algorithms: dict[str, str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProtectedTransaction:
    """Safe audit metadata about a cryptographically protected transaction.

    NEVER holds a secret key, shared secret, or private key -- only sizes,
    algorithm identifiers, and timing, which is what an audit log needs.
    """

    transaction_id: str
    security_decision: SecurityDecision
    kem_algorithm: str | None
    signature_algorithm: str | None
    symmetric_algorithm: str
    ciphertext_size_bytes: int
    signature_size_bytes: int | None
    processing_ms: dict[str, float]
    qkd_summary: dict[str, Any] | None = None  # sifted-key rate, QBER, detection flag -- no key bits

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d
