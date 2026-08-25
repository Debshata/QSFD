"""Key contributing factors for a fraud decision.

The SRS requires the engine to say *why* a transaction was flagged, not just
that it was. Three levels, in order of preference per model:

* **Logistic regression** -- signed contribution ``coef_j * x_j``. Exact, and
  the sign tells you which direction pushed the score.
* **Tree ensembles (RF / XGBoost)** -- SHAP values when SHAP is importable and
  fast enough, otherwise impurity/gain importance weighted by the transaction's
  own feature values.
* **Isolation Forest** -- no per-feature attribution is claimed; it returns an
  empty list rather than a fabricated one.

SHAP is optional on purpose: an explainability library that fails to import
must not take the fraud engine down with it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

_SHAP_STATE: dict[str, Any] = {"checked": False, "available": False, "module": None}


def shap_available() -> bool:
    """Import SHAP once, cache the outcome, never raise."""
    if not _SHAP_STATE["checked"]:
        _SHAP_STATE["checked"] = True
        try:
            import shap

            _SHAP_STATE.update(available=True, module=shap)
            LOGGER.info("SHAP available (%s)", getattr(shap, "__version__", "?"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.info("SHAP unavailable (%r); falling back to native importance", exc)
    return bool(_SHAP_STATE["available"])


class Explainer:
    """Per-prediction attribution for one fitted model."""

    def __init__(self, model, model_name: str, feature_names: list[str],
                 background: np.ndarray | None = None):
        self.model = model
        self.model_name = model_name
        self.feature_names = list(feature_names)
        self.method = "none"
        self._shap_explainer = None
        self._native_importance: np.ndarray | None = None

        cfg = load_config("risk")["explainability"]
        self.top_k = int(cfg.get("top_k", 5))
        want_shap = cfg.get("method", "shap") == "shap"

        if model_name == "isolation_forest":
            # An anomaly score has no per-feature decomposition we can honestly
            # report, so we do not invent one.
            self.method = "unsupported"
            return

        if model_name == "logistic_regression" and hasattr(model, "coef_"):
            self.method = "coefficient"
            self._coef = np.asarray(model.coef_).ravel()
            return

        if want_shap and model_name in ("random_forest", "xgboost") and shap_available():
            try:
                shap = _SHAP_STATE["module"]
                self._shap_explainer = shap.TreeExplainer(model)
                self.method = "shap"
                LOGGER.info("Explainer for %s: SHAP TreeExplainer", model_name)
                return
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("SHAP TreeExplainer failed for %s (%r); "
                               "using native importance", model_name, exc)

        if hasattr(model, "feature_importances_"):
            self.method = "native_importance"
            self._native_importance = np.asarray(model.feature_importances_, dtype=float)

    def explain(self, x_row: np.ndarray) -> list[dict[str, Any]]:
        """Top-k contributing features for a single transformed row."""
        x = np.asarray(x_row, dtype=float).reshape(1, -1)
        try:
            if self.method == "coefficient":
                contrib = self._coef * x[0]
            elif self.method == "shap":
                values = self._shap_explainer.shap_values(x, check_additivity=False)
                arr = np.asarray(values)
                if arr.ndim == 3:          # (n, features, classes) -> positive class
                    arr = arr[..., -1]
                contrib = arr.reshape(-1)
            elif self.method == "native_importance":
                # Importance alone is global; scaling by the row's own value
                # makes it at least transaction-specific. Signed by the value.
                contrib = self._native_importance * x[0]
            else:
                return []
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Explanation failed for %s: %r", self.model_name, exc)
            return []

        if contrib.shape[0] != len(self.feature_names):
            LOGGER.warning("Contribution/feature length mismatch (%d vs %d)",
                           contrib.shape[0], len(self.feature_names))
            return []

        order = np.argsort(np.abs(contrib))[::-1][: self.top_k]
        return [
            {
                "feature": self.feature_names[i],
                "contribution": round(float(contrib[i]), 6),
                "direction": "increases_fraud_risk" if contrib[i] > 0
                             else "decreases_fraud_risk",
                "value": round(float(x[0, i]), 6),
            }
            for i in order
        ]


def global_importance_report(model, model_name: str, feature_names: list[str],
                             splits, preprocessor) -> pd.DataFrame:
    """Dataset-level feature importance for the selected model."""
    rows: list[dict[str, Any]] = []
    if model_name == "logistic_regression" and hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        rows = [{"feature": f, "importance": abs(float(c)), "coefficient": float(c),
                 "method": "logistic_coefficient"}
                for f, c in zip(feature_names, coef)]
    elif hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        rows = [{"feature": f, "importance": float(v), "coefficient": float("nan"),
                 "method": f"{model_name}_native_importance"}
                for f, v in zip(feature_names, imp)]
    else:
        raise TypeError(f"No global importance available for {model_name!r}")

    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)
