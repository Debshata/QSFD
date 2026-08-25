"""Classical-quantum hybrid fusion.

Fusion weights are learned on the **validation** fold only. Training them on
test labels would make every hybrid look good and mean nothing -- that is the
single most common way a hybrid-QML result gets inflated, so it is structurally
prevented here: ``fit_fusion`` never receives test labels.

Each hybrid is reported alongside its own ablation:

    classical only   |   quantum only   |   hybrid

so any gain can be attributed rather than assumed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.config import get_seed
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class FusionModel:
    """A fitted score-combination rule."""

    method: str
    model: Any
    component_names: list[str]
    weights: list[float] | None = None

    def combine(self, scores: np.ndarray) -> np.ndarray:
        """scores: (n_samples, n_components) -> fused score in [0, 1]."""
        if self.method == "weighted":
            w = np.asarray(self.weights, dtype=float)
            return np.clip(scores @ w, 0.0, 1.0)
        return self.model.predict_proba(scores)[:, 1]

    def info(self) -> dict[str, Any]:
        return {"fusion": self.method, "components": list(self.component_names),
                "weights": self.weights}


def _standardise(scores: np.ndarray) -> np.ndarray:
    """Map each component's scores into [0, 1] by rank-preserving min-max.

    Components live on different scales (probability, SVM margin, expectation
    value). Without this the meta-learner would be dominated by whichever
    component happens to have the largest numeric range.
    """
    out = np.asarray(scores, dtype=float).copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        lo, hi = float(np.min(col)), float(np.max(col))
        out[:, j] = (col - lo) / (hi - lo) if hi > lo else 0.5
    return out


def fit_fusion(val_scores: np.ndarray, y_val: np.ndarray,
               component_names: list[str], method: str = "logistic_meta"
               ) -> FusionModel:
    """Learn the combination rule from VALIDATION scores and labels only."""
    val_scores = np.asarray(val_scores, dtype=float)
    y_val = np.asarray(y_val).astype(int)

    if method == "weighted":
        # Weight each component by its validation ROC-AUC above chance, so a
        # component that is no better than random contributes nothing.
        from sklearn.metrics import roc_auc_score

        raw = []
        for j in range(val_scores.shape[1]):
            try:
                auc = float(roc_auc_score(y_val, val_scores[:, j]))
            except Exception:  # noqa: BLE001
                auc = 0.5
            raw.append(max(auc - 0.5, 0.0))
        total = sum(raw) or 1.0
        weights = [round(r / total, 6) for r in raw]
        LOGGER.info("Fusion (weighted): %s", dict(zip(component_names, weights)))
        return FusionModel(method="weighted", model=None,
                           component_names=component_names, weights=weights)

    from sklearn.linear_model import LogisticRegression

    meta = LogisticRegression(max_iter=1000, random_state=get_seed())
    meta.fit(val_scores, y_val)
    coefs = [round(float(c), 6) for c in meta.coef_.ravel()]
    LOGGER.info("Fusion (logistic_meta): coefficients %s",
                dict(zip(component_names, coefs)))
    return FusionModel(method="logistic_meta", model=meta,
                       component_names=component_names, weights=coefs)


def build_hybrid(components: dict[str, dict[str, np.ndarray]], y_val: np.ndarray,
                 method: str = "logistic_meta") -> dict[str, Any]:
    """Fuse component scores into a hybrid.

    ``components`` maps a component name to ``{"val": ..., "test": ...}``
    score arrays. Only the validation arrays are used for fitting.
    """
    names = list(components)
    val_raw = np.column_stack([components[n]["val"] for n in names])
    test_raw = np.column_stack([components[n]["test"] for n in names])

    # Scale val and test together only in the sense of applying the SAME
    # transform shape; the fusion model itself never sees test labels.
    val_scaled = _standardise(val_raw)
    test_scaled = _standardise(test_raw)

    t0 = time.perf_counter()
    fusion = fit_fusion(val_scaled, y_val, names, method=method)
    fit_seconds = time.perf_counter() - t0

    return {
        "fusion": fusion,
        "val_score": fusion.combine(val_scaled),
        "test_score": fusion.combine(test_scaled),
        "component_names": names,
        "fusion_seconds": round(fit_seconds, 4),
        "fusion_info": fusion.info(),
        # The fused output of a logistic meta-learner is a probability; a
        # weighted blend of heterogeneous scores is not.
        "score_is_probability": method == "logistic_meta",
    }


def train_classical_component(name: str, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, X_test: np.ndarray
                              ) -> dict[str, Any]:
    """Fit the classical half of a hybrid on the controlled subset."""
    seed = get_seed()
    t0 = time.perf_counter()

    if name == "xgboost":
        from xgboost import XGBClassifier

        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
            random_state=seed, n_jobs=-1,
            scale_pos_weight=(n_neg / n_pos) if n_pos else 1.0)
        model.fit(X_train, y_train)
    elif name == "chow_liu":
        from src.qml_experimental.chow_liu.classifier import ChowLiuClassifier

        model = ChowLiuClassifier(seed=seed).fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown classical component {name!r}")

    train_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    test_score = model.predict_proba(X_test)[:, 1]
    infer_seconds = time.perf_counter() - t0

    return {
        "model": model,
        "name": name,
        "val_score": model.predict_proba(X_val)[:, 1],
        "test_score": test_score,
        "score_is_probability": name == "xgboost",
        "train_seconds": round(train_seconds, 3),
        "inference_ms_per_transaction": round(
            1000 * infer_seconds / max(len(X_test), 1), 4),
    }
