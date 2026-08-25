"""The integration contract between the fraud layer and the security layer.

Everything downstream (PQC selection, QKD negotiation, dashboard) consumes
:class:`FraudAssessment`. Its shape is stable; fields are added, never
repurposed. Plain dataclasses are used rather than pydantic so the contract
carries no extra runtime dependency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass
class FeatureContribution:
    feature: str
    contribution: float
    direction: str = ""
    value: float | None = None


@dataclass
class ModelOutput:
    """One model's opinion about one transaction.

    ``fraud_probability`` is populated only when the model genuinely produces a
    calibrated-scale probability. A model that emits an uncalibrated decision
    value fills ``score`` instead and leaves ``fraud_probability`` as None --
    we do not manufacture a probability that the model cannot support.
    """

    model: str
    prediction: int
    fraud_probability: float | None = None
    score: float | None = None
    score_is_probability: bool = True
    threshold: float = 0.5
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class FraudAssessment:
    """Stable result object handed to the security-selection layer."""

    transaction_id: str
    fraud_probability: float
    fraud_prediction: int
    risk_score: float
    risk_level: RiskLevel
    model_name: str
    model_version: str
    threshold: float
    top_contributing_features: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0

    # Optional detail: populated by compare_models / the QML branch.
    classical: dict[str, Any] | None = None
    qml: dict[str, Any] | None = None
    latency_breakdown_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def validate(self) -> None:
        """Assert the contract. Used by the smoke test and the test suite."""
        if not 0.0 <= self.fraud_probability <= 1.0:
            raise ValueError(f"fraud_probability out of range: {self.fraud_probability}")
        if self.fraud_prediction not in (0, 1):
            raise ValueError(f"fraud_prediction must be 0 or 1: {self.fraud_prediction}")
        if self.risk_level not in ("LOW", "MEDIUM", "HIGH"):
            raise ValueError(f"invalid risk_level: {self.risk_level}")
        if not 0.0 <= self.risk_score <= 100.0:
            raise ValueError(f"risk_score out of range: {self.risk_score}")
        if not isinstance(self.top_contributing_features, list):
            raise TypeError("top_contributing_features must be a list")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")


# Field names guaranteed present in a serialised assessment. The security
# layer may rely on exactly these.
CONTRACT_FIELDS = (
    "transaction_id",
    "fraud_probability",
    "fraud_prediction",
    "risk_score",
    "risk_level",
    "model_name",
    "model_version",
    "threshold",
    "top_contributing_features",
    "latency_ms",
)
