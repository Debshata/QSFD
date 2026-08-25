"""Risk scoring.

Fraud probability and risk level are deliberately separate:

* **fraud_probability** is the model's output -- a statistical estimate.
* **fraud_prediction** is that probability compared against the model's
  selected decision threshold.
* **risk_score** is the probability rescaled to 0-100 for the downstream
  security-selection layer.
* **risk_level** is a *business* mapping (LOW / MEDIUM / HIGH) whose bands live
  in ``config/risk.yaml``, not in code.

The threshold floor exists because the two concepts can disagree: a model
whose threshold is 0.12 can flag a transaction that still falls inside the LOW
probability band. Silently reporting LOW for a flagged transaction would hide
a positive detection from the security layer, so a flagged transaction is
floored at MEDIUM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.config import load_config

LOW, MEDIUM, HIGH = "LOW", "MEDIUM", "HIGH"
RISK_LEVELS = (LOW, MEDIUM, HIGH)


@dataclass(frozen=True)
class RiskConfig:
    low_upper_bound: float
    medium_upper_bound: float
    score_scale: float
    enforce_threshold_floor: bool

    @classmethod
    def from_config(cls, override: dict[str, Any] | None = None) -> "RiskConfig":
        cfg = dict(load_config("risk")["risk"])
        if override:
            cfg.update(override)
        rc = cls(
            low_upper_bound=float(cfg["low_upper_bound"]),
            medium_upper_bound=float(cfg["medium_upper_bound"]),
            score_scale=float(cfg.get("score_scale", 100)),
            enforce_threshold_floor=bool(cfg.get("enforce_threshold_floor", True)),
        )
        rc.validate()
        return rc

    def validate(self) -> None:
        if not 0.0 <= self.low_upper_bound < self.medium_upper_bound <= 1.0:
            raise ValueError(
                "Risk bands must satisfy 0 <= low_upper_bound < "
                f"medium_upper_bound <= 1, got {self.low_upper_bound} / "
                f"{self.medium_upper_bound}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "low_upper_bound": self.low_upper_bound,
            "medium_upper_bound": self.medium_upper_bound,
            "score_scale": self.score_scale,
            "enforce_threshold_floor": self.enforce_threshold_floor,
        }


class RiskEngine:
    """Maps a fraud probability + threshold onto a risk score and level."""

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig.from_config()

    def score(self, fraud_probability: float) -> float:
        p = float(fraud_probability)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"fraud_probability must be in [0, 1], got {p}")
        return round(p * self.config.score_scale, 2)

    def level(self, fraud_probability: float, *, threshold: float | None = None,
              prediction: int | None = None) -> str:
        p = float(fraud_probability)
        if p <= self.config.low_upper_bound:
            level = LOW
        elif p <= self.config.medium_upper_bound:
            level = MEDIUM
        else:
            level = HIGH

        if self.config.enforce_threshold_floor and level == LOW:
            flagged = prediction == 1 if prediction is not None else (
                threshold is not None and p >= threshold)
            if flagged:
                level = MEDIUM
        return level

    def assess(self, fraud_probability: float, threshold: float) -> dict[str, Any]:
        p = float(fraud_probability)
        prediction = int(p >= float(threshold))
        return {
            "fraud_probability": round(p, 6),
            "fraud_prediction": prediction,
            "risk_score": self.score(p),
            "risk_level": self.level(p, threshold=threshold, prediction=prediction),
            "threshold": round(float(threshold), 6),
        }

    def describe(self) -> dict[str, Any]:
        c = self.config
        return {
            **c.as_dict(),
            "mapping": {
                LOW: f"p <= {c.low_upper_bound}",
                MEDIUM: f"{c.low_upper_bound} < p <= {c.medium_upper_bound}"
                        + (" (or p >= model threshold)" if c.enforce_threshold_floor else ""),
                HIGH: f"p > {c.medium_upper_bound}",
            },
        }
