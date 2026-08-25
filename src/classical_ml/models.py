"""Classical model factories.

Each factory returns a configured, unfitted estimator. Hyperparameters come
from ``config/classical.yaml`` and are deliberately fixed strong defaults --
this project does not run a large hyperparameter search (see
``docs/DECISIONS.md`` for the rationale).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.config import get_seed, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

SUPERVISED_MODELS = ("logistic_regression", "random_forest", "xgboost")
ANOMALY_MODELS = ("isolation_forest",)


def _params(name: str) -> dict[str, Any]:
    cfg = load_config("classical")["models"]
    return dict(cfg[name].get("params") or {})


def is_enabled(name: str) -> bool:
    cfg = load_config("classical")["models"]
    return bool(cfg.get(name, {}).get("enabled", False))


def build_logistic_regression():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(random_state=get_seed(), **_params("logistic_regression"))


def build_random_forest():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(random_state=get_seed(), **_params("random_forest"))


def build_xgboost(y_train: np.ndarray | None = None):
    from xgboost import XGBClassifier

    params = _params("xgboost")
    cfg = load_config("classical")["models"]["xgboost"]
    if cfg.get("auto_scale_pos_weight", True) and y_train is not None:
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        if n_pos:
            # The standard XGBoost imbalance handle: weight the positive class
            # by the negative:positive ratio of the TRAINING fold.
            params["scale_pos_weight"] = round(n_neg / n_pos, 4)
            LOGGER.info("xgboost scale_pos_weight=%.4f (neg=%d, pos=%d)",
                        params["scale_pos_weight"], n_neg, n_pos)
    return XGBClassifier(random_state=get_seed(), **params)


def build_isolation_forest():
    from sklearn.ensemble import IsolationForest

    return IsolationForest(random_state=get_seed(), **_params("isolation_forest"))


def build_model(name: str, y_train: np.ndarray | None = None):
    if name == "logistic_regression":
        return build_logistic_regression()
    if name == "random_forest":
        return build_random_forest()
    if name == "xgboost":
        return build_xgboost(y_train)
    if name == "isolation_forest":
        return build_isolation_forest()
    raise ValueError(f"Unknown model {name!r}")


def score_samples(model, X: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    """Return (score, score_is_probability) for any supported model.

    Isolation Forest has no probability. Its ``score_samples`` output is an
    anomaly score; we min-max it into [0, 1] purely so the same plotting and
    ranking code works, and flag it as *not* a probability so nothing
    downstream treats it as one.
    """
    if name in ANOMALY_MODELS:
        raw = -model.score_samples(X)          # higher = more anomalous
        lo, hi = float(raw.min()), float(raw.max())
        scaled = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
        return scaled, False
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1], True
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float), False
    raise TypeError(f"Model {name!r} exposes neither predict_proba nor decision_function")
