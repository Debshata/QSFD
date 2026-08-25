"""The controlled experimental subset, shared by every experimental arm.

Every model in this research layer -- classical baselines, Chow-Liu, QNN, VQC,
QSVM, CLT-QNN, CLT-VQC, QBoost, hybrids -- is evaluated on **exactly the same
rows**. The only thing that varies between Pipeline A and Pipeline B is the
dimensionality-reduction step:

    Pipeline A (pca)       original -> preprocess -> PCA        -> n components
    Pipeline B (selected)  original -> preprocess -> MI select  -> n features

Row selection is seeded and identical across both, so a difference in results
is attributable to the features or the model, never to a luckier test set.

Nothing here is fitted on validation or test data.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from src.data.loaders import load_dataset
from src.data.preprocessing import fit_preprocessor, make_splits, transform
from src.data.registry import available_datasets
from src.qml.feature_reduction import build_reducer, stratified_subsample
from src.utils.config import get_seed, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ExperimentalSubset:
    """One (pipeline, n_features) view of the controlled subset."""

    pipeline: str
    n_features: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    interpretable_features: bool
    reducer: Any = None
    angle_scaler: Any = None
    explained_variance: float | None = None
    row_fingerprint: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names),
            "interpretable_features": self.interpretable_features,
            "explained_variance_ratio": self.explained_variance,
            "train_samples": int(len(self.X_train)),
            "val_samples": int(len(self.X_val)),
            "test_samples": int(len(self.X_test)),
            "train_fraud": int(self.y_train.sum()),
            "val_fraud": int(self.y_val.sum()),
            "test_fraud": int(self.y_test.sum()),
            "row_fingerprint": self.row_fingerprint,
        }


@lru_cache(maxsize=1)
def _base_folds() -> dict[str, Any]:
    """Preprocessed full-scale folds, rebuilt exactly as the baseline does.

    Cached: rebuilding costs ~20 s and every experiment needs it.
    """
    cfg = load_config("datasets")
    name = cfg["primary_dataset"]
    rec = available_datasets()[name]

    loaded = load_dataset(name, rec["entry"], rec["file"])
    splits = make_splits(loaded.X, loaded.y, loaded.spec)
    preprocessor, feature_names = fit_preprocessor(loaded.spec, splits.X_train)

    LOGGER.info("Base folds rebuilt for experiments: dataset=%s, %d features",
                name, len(feature_names))
    return {
        "dataset": name,
        "feature_names": feature_names,
        "Xtr": transform(preprocessor, splits.X_train),
        "ytr": splits.y_train.to_numpy(),
        "Xva": transform(preprocessor, splits.X_val),
        "yva": splits.y_val.to_numpy(),
        "Xte": transform(preprocessor, splits.X_test),
        "yte": splits.y_test.to_numpy(),
        "preprocessor": preprocessor,
    }


@lru_cache(maxsize=1)
def _controlled_indices() -> dict[str, np.ndarray]:
    """Row indices of the controlled subset -- computed once, reused by all.

    Uses the same seed offsets as the published baseline, so the experimental
    subset is the baseline subset.
    """
    base = _base_folds()
    exp = load_config("qml_experimental")["subset"]
    seed = get_seed()
    ratio = float(exp["fraud_ratio"])

    idx = {
        "train": stratified_subsample(base["Xtr"], base["ytr"],
                                      int(exp["train_samples"]), ratio, seed),
        "val": stratified_subsample(base["Xva"], base["yva"],
                                    int(exp["val_samples"]), ratio, seed + 1),
        "test": stratified_subsample(base["Xte"], base["yte"],
                                     int(exp["test_samples"]), ratio, seed + 2),
    }
    LOGGER.info("Controlled subset rows fixed: train=%d val=%d test=%d",
                len(idx["train"]), len(idx["val"]), len(idx["test"]))
    return idx


def row_fingerprint() -> str:
    """Hash of the selected row indices.

    Recorded on every experiment so a reviewer can verify that two results
    really were produced on the same rows.
    """
    idx = _controlled_indices()
    h = hashlib.sha256()
    for key in ("train", "val", "test"):
        h.update(np.asarray(idx[key], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def build_subset(pipeline: str, n_features: int) -> ExperimentalSubset:
    """Build one (pipeline, n_features) view of the controlled subset."""
    cfg = load_config("qml_experimental")
    if pipeline not in cfg["pipelines"]:
        raise ValueError(f"Unknown pipeline {pipeline!r}. "
                         f"Known: {sorted(cfg['pipelines'])}")
    pipe_cfg = cfg["pipelines"][pipeline]
    method = pipe_cfg["method"]
    seed = get_seed()

    base = _base_folds()
    idx = _controlled_indices()

    Xtr, ytr = base["Xtr"][idx["train"]], base["ytr"][idx["train"]]
    Xva, yva = base["Xva"][idx["val"]], base["yva"][idx["val"]]
    Xte, yte = base["Xte"][idx["test"]], base["yte"][idx["test"]]

    # Reducer is fitted on the experimental TRAINING rows only.
    reducer = build_reducer(method, n_features, seed)
    if method == "pca":
        reducer.fit(Xtr)
    else:
        reducer.fit(Xtr, ytr)

    Rtr, Rva, Rte = (reducer.transform(X) for X in (Xtr, Xva, Xte))

    explained = None
    if method == "pca":
        explained = round(float(np.sum(reducer.explained_variance_ratio_)), 4)
        names = [f"PC{i + 1}" for i in range(n_features)]
    elif hasattr(reducer, "get_support"):
        mask = reducer.get_support()
        names = [n for n, keep in zip(base["feature_names"], mask) if keep]
    else:
        names = [f"f{i}" for i in range(n_features)]

    angle_scaler = None
    if cfg.get("angle_scaling", True):
        from sklearn.preprocessing import MinMaxScaler

        angle_scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)))
        angle_scaler.fit(Rtr)
        Rtr = angle_scaler.transform(Rtr)
        # Val/test can fall outside the training range; clip so the angle
        # encoding stays inside the intended window rather than wrapping.
        Rva = np.clip(angle_scaler.transform(Rva), 0.0, np.pi)
        Rte = np.clip(angle_scaler.transform(Rte), 0.0, np.pi)

    subset = ExperimentalSubset(
        pipeline=pipeline, n_features=n_features,
        X_train=Rtr, y_train=ytr, X_val=Rva, y_val=yva, X_test=Rte, y_test=yte,
        feature_names=names,
        interpretable_features=bool(pipe_cfg.get("interpretable_features", False)),
        reducer=reducer, angle_scaler=angle_scaler,
        explained_variance=explained, row_fingerprint=row_fingerprint(),
    )
    LOGGER.info("Subset[%s, %dD]: %s%s", pipeline, n_features,
                names if len(names) <= 8 else f"{len(names)} features",
                f", {100 * explained:.1f}% variance" if explained else "")
    return subset


def unreduced_subset() -> dict[str, Any]:
    """The controlled subset in the FULL preprocessed feature space.

    Used by the controlled classical baselines (XGBoost etc.) and by QBoost,
    which are not restricted to a quantum-sized feature space.
    """
    base = _base_folds()
    idx = _controlled_indices()
    return {
        "X_train": base["Xtr"][idx["train"]], "y_train": base["ytr"][idx["train"]],
        "X_val": base["Xva"][idx["val"]], "y_val": base["yva"][idx["val"]],
        "X_test": base["Xte"][idx["test"]], "y_test": base["yte"][idx["test"]],
        "feature_names": base["feature_names"],
        "row_fingerprint": row_fingerprint(),
    }
