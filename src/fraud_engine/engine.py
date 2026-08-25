"""The unified fraud detection engine.

This is the integration surface for the rest of the system. It loads a
persisted model from disk (never a model that happens to be in memory), scores
a transaction, maps the probability to a risk band and attaches the top
contributing features.

    engine = FraudDetectionEngine.load()
    result = engine.predict({"V1": ..., "Amount": 149.62})
    result.risk_level      # -> "LOW" | "MEDIUM" | "HIGH"

The security-selection layer consumes :class:`FraudAssessment` and nothing
else; see ``src/fraud_engine/schemas.py``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd

from src.fraud_engine.explainability import Explainer
from src.fraud_engine.risk import RiskEngine
from src.fraud_engine.schemas import FraudAssessment, ModelOutput
from src.utils.config import get_path
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the requested artifacts are not on disk."""


class FraudDetectionEngine:
    """Loads persisted artifacts and turns transactions into risk decisions."""

    def __init__(self, model, preprocessor, metadata: dict[str, Any],
                 calibrator=None, risk_engine: RiskEngine | None = None):
        self.model = model
        self.preprocessor = preprocessor
        self.calibrator = calibrator
        self.metadata = metadata
        self.risk = risk_engine or RiskEngine()

        self.model_name: str = metadata["model_name"]
        self.model_version: str = metadata.get("model_version", "unknown")
        self.threshold: float = float(metadata["threshold"])
        self.raw_features: list[str] = list(metadata["raw_feature_columns"])
        self.feature_names: list[str] = list(metadata["transformed_feature_names"])
        self.score_is_probability: bool = bool(metadata.get("score_is_probability", True))

        self.explainer = Explainer(model, self.model_name, self.feature_names)
        self._qml: dict[str, Any] | None = None

    # -- construction -----------------------------------------------------

    @classmethod
    def load(cls, model_name: str | None = None,
             models_dir: str | Path | None = None) -> "FraudDetectionEngine":
        """Load the production model (or a named one) from disk."""
        root = Path(models_dir) if models_dir else Path(get_path("models"))
        registry_path = root / "metadata" / "classical_registry.json"

        if model_name is None:
            if not registry_path.exists():
                raise ModelNotFoundError(
                    f"No classical registry at {registry_path}. "
                    "Run: python -m src.classical_ml.train"
                )
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            model_name = registry.get("production_model")
            if not model_name:
                raise ModelNotFoundError("Registry names no production model")

        model_dir = root / "classical" / model_name
        if not (model_dir / "model.joblib").exists():
            raise ModelNotFoundError(f"No saved model at {model_dir}")

        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        model = joblib.load(model_dir / "model.joblib")
        preprocessor = joblib.load(model_dir / "preprocessor.joblib")
        calibrator = None
        if (model_dir / "calibrator.joblib").exists():
            calibrator = joblib.load(model_dir / "calibrator.joblib")

        LOGGER.info("Loaded engine: %s (version %s, threshold %.4f%s)",
                    model_name, metadata.get("model_version"),
                    float(metadata["threshold"]),
                    ", calibrated" if calibrator is not None else "")
        return cls(model, preprocessor, metadata, calibrator)

    def attach_qml(self, model_name: str | None = None,
                   models_dir: str | Path | None = None) -> bool:
        """Optionally attach a persisted QML model for ``compare_models``.

        Returns False (and logs) rather than raising if the QML branch was not
        trained or its estimator could not be unpickled -- the classical engine
        must stay usable either way.
        """
        root = Path(models_dir) if models_dir else Path(get_path("models"))
        registry_path = root / "metadata" / "qml_registry.json"
        if not registry_path.exists():
            LOGGER.info("No QML registry; compare_models will report classical only")
            return False
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        saved = registry.get("saved_models") or {}
        model_name = model_name or next(iter(saved), None)
        if not model_name or model_name not in saved:
            LOGGER.info("No usable QML model in registry")
            return False
        model_dir = root / "qml" / model_name
        try:
            meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
            reduction = joblib.load(model_dir / "feature_reduction.joblib")
            estimator = joblib.load(model_dir / "model.joblib")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not load QML model %s: %r", model_name, exc)
            return False
        self._qml = {"name": model_name, "estimator": estimator,
                     "reduction": reduction, "metadata": meta}
        LOGGER.info("Attached QML model %s", model_name)
        return True

    # -- feature handling -------------------------------------------------

    def _to_frame(self, transactions: Any) -> pd.DataFrame:
        """Coerce dict / list-of-dicts / DataFrame into the training schema."""
        if isinstance(transactions, pd.DataFrame):
            df = transactions.copy()
        elif isinstance(transactions, dict):
            df = pd.DataFrame([transactions])
        elif isinstance(transactions, Sequence):
            if not transactions:
                raise ValueError("Empty transaction batch")
            if not all(isinstance(t, dict) for t in transactions):
                raise TypeError("Batch must be a sequence of dicts or a DataFrame")
            df = pd.DataFrame(list(transactions))
        else:
            raise TypeError(
                f"Unsupported transaction type {type(transactions).__name__}; "
                "expected dict, sequence of dicts, or DataFrame"
            )

        missing = [c for c in self.raw_features if c not in df.columns]
        if missing:
            raise ValueError(
                f"Transaction is missing {len(missing)} required feature(s): "
                f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
            )
        # Column order must match training; extras are ignored, not an error,
        # so upstream systems can pass richer payloads (e.g. a transaction id).
        return df[self.raw_features]

    def _probabilities(self, X: np.ndarray) -> np.ndarray:
        scorer = self.calibrator if self.calibrator is not None else self.model
        if hasattr(scorer, "predict_proba"):
            return np.asarray(scorer.predict_proba(X))[:, 1]
        from src.classical_ml.models import score_samples

        scores, _ = score_samples(scorer, X, self.model_name)
        return scores

    # -- public API -------------------------------------------------------

    def predict(self, transaction: dict | pd.Series,
                transaction_id: str | None = None,
                explain: bool = True) -> FraudAssessment:
        """Score a single transaction."""
        if isinstance(transaction, pd.Series):
            transaction = transaction.to_dict()
        t0 = time.perf_counter()

        df = self._to_frame(transaction)
        t_pre = time.perf_counter()
        X = np.asarray(self.preprocessor.transform(df), dtype=float)
        pre_ms = 1000 * (time.perf_counter() - t_pre)

        t_inf = time.perf_counter()
        probability = float(self._probabilities(X)[0])
        inf_ms = 1000 * (time.perf_counter() - t_inf)

        assessment = self.risk.assess(probability, self.threshold)
        contributions = self.explainer.explain(X[0]) if explain else []

        warnings: list[str] = []
        if not self.score_is_probability:
            warnings.append(
                f"{self.model_name} emits an anomaly score, not a calibrated "
                "probability; risk_score is a rank, not a likelihood"
            )
        if explain and not contributions:
            warnings.append(
                f"no per-feature attribution available for {self.model_name} "
                f"(method={self.explainer.method})"
            )

        total_ms = 1000 * (time.perf_counter() - t0)
        result = FraudAssessment(
            transaction_id=str(transaction_id or transaction.get("transaction_id",
                                                                 "unknown")),
            fraud_probability=assessment["fraud_probability"],
            fraud_prediction=assessment["fraud_prediction"],
            risk_score=assessment["risk_score"],
            risk_level=assessment["risk_level"],
            model_name=self.model_name,
            model_version=self.model_version,
            threshold=self.threshold,
            top_contributing_features=contributions,
            latency_ms=round(total_ms, 3),
            latency_breakdown_ms={"preprocessing": round(pre_ms, 3),
                                  "inference": round(inf_ms, 3),
                                  "total": round(total_ms, 3)},
            warnings=warnings,
        )
        result.validate()
        return result

    def predict_batch(self, transactions: Iterable[dict] | pd.DataFrame,
                      transaction_ids: Sequence[str] | None = None,
                      explain: bool = False) -> list[FraudAssessment]:
        """Score many transactions in one vectorised pass.

        ``explain`` defaults to False here: per-row attribution is the
        expensive part and batch scoring is usually a throughput path.
        """
        t0 = time.perf_counter()
        df = self._to_frame(transactions)
        X = np.asarray(self.preprocessor.transform(df), dtype=float)
        probabilities = self._probabilities(X)
        total_ms = 1000 * (time.perf_counter() - t0)
        per_row_ms = total_ms / max(len(df), 1)

        results = []
        for i, p in enumerate(probabilities):
            assessment = self.risk.assess(float(p), self.threshold)
            tid = (str(transaction_ids[i]) if transaction_ids is not None
                   and i < len(transaction_ids) else f"batch-{i}")
            result = FraudAssessment(
                transaction_id=tid,
                fraud_probability=assessment["fraud_probability"],
                fraud_prediction=assessment["fraud_prediction"],
                risk_score=assessment["risk_score"],
                risk_level=assessment["risk_level"],
                model_name=self.model_name,
                model_version=self.model_version,
                threshold=self.threshold,
                top_contributing_features=(
                    self.explainer.explain(X[i]) if explain else []),
                latency_ms=round(per_row_ms, 4),
            )
            result.validate()
            results.append(result)
        return results

    def compare_models(self, transactions: Any,
                       transaction_id: str | None = None) -> FraudAssessment:
        """Score with the classical model and, if attached, the QML model.

        The QML branch reports a ``score`` rather than a ``fraud_probability``
        unless its metadata says the model genuinely produces one.
        """
        result = self.predict(transactions if isinstance(transactions, dict)
                              else transactions, transaction_id=transaction_id)
        classical = ModelOutput(
            model=self.model_name, prediction=result.fraud_prediction,
            fraud_probability=result.fraud_probability,
            score=result.fraud_probability, score_is_probability=True,
            threshold=self.threshold,
            latency_ms=result.latency_breakdown_ms.get("inference", 0.0),
        )
        result.classical = classical.__dict__

        if self._qml is None:
            result.warnings.append("no QML model attached; classical result only")
            return result

        try:
            df = self._to_frame(transactions)
            X = np.asarray(self.preprocessor.transform(df), dtype=float)
            reduction = self._qml["reduction"]
            reduced = reduction["reducer"].transform(X)
            if reduction.get("angle_scaler") is not None:
                reduced = np.clip(reduction["angle_scaler"].transform(reduced),
                                  0.0, float(np.pi))

            meta = self._qml["metadata"]
            t0 = time.perf_counter()
            estimator = self._qml["estimator"]
            if hasattr(estimator, "decision_function"):
                score = float(np.ravel(estimator.decision_function(reduced))[0])
            else:
                score = float(np.ravel(estimator.predict(reduced))[0])
            qml_ms = 1000 * (time.perf_counter() - t0)

            is_prob = bool(meta.get("score_is_probability", False))
            threshold = float(meta.get("threshold", 0.5))
            qml_out = ModelOutput(
                model=self._qml["name"],
                prediction=int(score >= threshold),
                fraud_probability=score if is_prob else None,
                score=round(score, 6),
                score_is_probability=is_prob,
                threshold=threshold,
                latency_ms=round(qml_ms, 3),
            )
            result.qml = qml_out.__dict__
            result.latency_breakdown_ms["qml"] = round(qml_ms, 3)
            if not is_prob:
                result.warnings.append(
                    f"{self._qml['name']} output is a decision score, not a "
                    "probability; it is not used for risk banding"
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("QML comparison failed: %r", exc)
            result.qml = ModelOutput(model=self._qml["name"], prediction=-1,
                                     error=repr(exc)).__dict__
        return result

    def describe(self) -> dict[str, Any]:
        """Everything the security layer might want to know about this engine."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "threshold": self.threshold,
            "score_is_probability": self.score_is_probability,
            "calibrated": self.calibrator is not None,
            "explainability_method": self.explainer.method,
            "n_raw_features": len(self.raw_features),
            "n_transformed_features": len(self.feature_names),
            "dataset": self.metadata.get("dataset"),
            "trained_at": self.metadata.get("trained_at"),
            "risk": self.risk.describe(),
            "test_metrics": self.metadata.get("test_metrics"),
            "qml_attached": self._qml["name"] if self._qml else None,
        }
