"""Bridging the classical feature space into a quantum-sized one.

A 29-feature transaction cannot be fed into a 29-qubit circuit on a simulator.
The pipeline is:

    raw features -> classical preprocessing -> reduction to 4-8 dimensions
                 -> scaling into [0, pi] -> angle encoding

The reducer and both scalers are fitted on the **training fold only**; the QML
validation and test subsets are transformed, never fitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.config import get_seed, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class QMLSubset:
    """The reproducible QML experiment subset, in reduced feature space."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_features: int
    fraud_ratio_train: float
    reducer: Any
    angle_scaler: Any
    method: str
    explained_variance: float | None = None
    selected_feature_names: list[str] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "qml_training_samples": int(len(self.X_train)),
            "qml_validation_samples": int(len(self.X_val)),
            "qml_test_samples": int(len(self.X_test)),
            "qml_features": self.n_features,
            "reduction_method": self.method,
            "explained_variance_ratio": self.explained_variance,
            "selected_feature_names": self.selected_feature_names,
            "train_fraud": int(self.y_train.sum()),
            "val_fraud": int(self.y_val.sum()),
            "test_fraud": int(self.y_test.sum()),
            "train_fraud_ratio": round(float(self.y_train.mean()), 4),
            "test_fraud_ratio": round(float(self.y_test.mean()), 4),
        }


def stratified_subsample(X: np.ndarray, y: np.ndarray, n_samples: int,
                         fraud_ratio: float, seed: int) -> np.ndarray:
    """Indices of a seeded subsample with a target fraud proportion.

    Real base rates (~0.17%) would give under one fraud case at these sizes,
    so the subset is deliberately rebalanced. That makes QML metrics
    comparable to the *controlled* classical baselines on the same subset,
    and NOT to the full-scale classical benchmark.
    """
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    n_pos_wanted = int(round(n_samples * fraud_ratio))
    n_pos = min(n_pos_wanted, len(pos))
    n_neg = min(n_samples - n_pos, len(neg))
    if n_pos < n_pos_wanted:
        LOGGER.warning("Only %d fraud samples available (wanted %d); "
                       "subset fraud ratio will be %.3f",
                       n_pos, n_pos_wanted, n_pos / max(n_pos + n_neg, 1))

    idx = np.concatenate([
        rng.choice(pos, size=n_pos, replace=False),
        rng.choice(neg, size=n_neg, replace=False),
    ])
    rng.shuffle(idx)
    return idx


def build_reducer(method: str, n_components: int, seed: int):
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=n_components, random_state=seed)
    if method == "mutual_info":
        from sklearn.feature_selection import SelectKBest, mutual_info_classif

        return SelectKBest(
            score_func=lambda X, y: mutual_info_classif(X, y, random_state=seed),
            k=n_components,
        )
    if method == "model_based":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel

        return SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
            max_features=n_components, threshold=-np.inf,
        )
    raise ValueError(f"Unknown feature reduction method {method!r}")


def build_qml_subset(Xtr: np.ndarray, ytr: np.ndarray,
                     Xva: np.ndarray, yva: np.ndarray,
                     Xte: np.ndarray, yte: np.ndarray,
                     feature_names: list[str] | None = None) -> QMLSubset:
    """Subsample, reduce and angle-scale, fitting on the training fold only."""
    cfg = load_config("qml")
    sub_cfg, red_cfg = cfg["subset"], cfg["feature_reduction"]
    seed = get_seed()
    ratio = float(sub_cfg["fraud_ratio"])

    itr = stratified_subsample(Xtr, ytr, int(sub_cfg["train_samples"]), ratio, seed)
    iva = stratified_subsample(Xva, yva, int(sub_cfg["val_samples"]), ratio, seed + 1)
    ite = stratified_subsample(Xte, yte, int(sub_cfg["test_samples"]), ratio, seed + 2)

    Xtr_s, ytr_s = Xtr[itr], ytr[itr]
    Xva_s, yva_s = Xva[iva], yva[iva]
    Xte_s, yte_s = Xte[ite], yte[ite]

    method = red_cfg["method"]
    n_components = int(red_cfg["n_components"])
    reducer = build_reducer(method, n_components, seed)

    # Fitted on the QML TRAINING subset only.
    if method == "pca":
        reducer.fit(Xtr_s)
    else:
        reducer.fit(Xtr_s, ytr_s)

    Rtr, Rva, Rte = (reducer.transform(X) for X in (Xtr_s, Xva_s, Xte_s))

    explained = None
    selected_names = None
    if method == "pca":
        explained = round(float(np.sum(reducer.explained_variance_ratio_)), 4)
        LOGGER.info("PCA -> %d components, %.1f%% variance retained",
                    n_components, 100 * explained)
    elif feature_names is not None and hasattr(reducer, "get_support"):
        mask = reducer.get_support()
        selected_names = [n for n, keep in zip(feature_names, mask) if keep]
        LOGGER.info("%s selected features: %s", method, selected_names)

    angle_scaler = None
    if red_cfg.get("angle_scaling", True):
        from sklearn.preprocessing import MinMaxScaler

        # Angle encoding needs bounded inputs; [0, pi] keeps a full rotation
        # range without wrapping two distinct values onto the same angle.
        angle_scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)))
        angle_scaler.fit(Rtr)
        Rtr, Rva, Rte = (angle_scaler.transform(R) for R in (Rtr, Rva, Rte))
        # Val/test values can fall outside the training range; clip so the
        # encoding stays inside the intended angular window.
        Rva = np.clip(Rva, 0.0, np.pi)
        Rte = np.clip(Rte, 0.0, np.pi)

    subset = QMLSubset(
        X_train=Rtr, y_train=ytr_s, X_val=Rva, y_val=yva_s,
        X_test=Rte, y_test=yte_s, n_features=n_components,
        fraud_ratio_train=float(ytr_s.mean()), reducer=reducer,
        angle_scaler=angle_scaler, method=method,
        explained_variance=explained, selected_feature_names=selected_names,
    )
    LOGGER.info("QML subset: train=%d (%d fraud), val=%d (%d), test=%d (%d), dims=%d",
                len(Rtr), int(ytr_s.sum()), len(Rva), int(yva_s.sum()),
                len(Rte), int(yte_s.sum()), n_components)
    return subset
