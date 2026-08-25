"""Splitting and preprocessing.

Leakage discipline, in one place:

* Duplicates are dropped **before** splitting so an identical transaction
  cannot appear in both train and test.
* The split happens **before** any fitting. Every transformer is fitted on the
  training fold only and merely *applied* to validation and test.
* Resampling (SMOTE / undersampling) touches the training fold only, never
  validation or test.

Everything is wrapped in a scikit-learn ``ColumnTransformer`` so this is
enforced structurally rather than by convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.registry import DatasetSpec
from src.data.validators import drop_cross_split_duplicates
from src.utils.config import get_seed, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class SplitData:
    """The three folds plus the metadata needed to reproduce them."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    strategy: str
    n_duplicates_dropped: int

    def summary(self) -> dict[str, Any]:
        def block(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
            return {
                "rows": int(len(X)),
                "fraud": int(y.sum()),
                "fraud_rate": round(float(y.mean()), 6),
            }

        return {
            "strategy": self.strategy,
            "duplicates_dropped": self.n_duplicates_dropped,
            "train": block(self.X_train, self.y_train),
            "val": block(self.X_val, self.y_val),
            "test": block(self.X_test, self.y_test),
        }


def make_splits(X: pd.DataFrame, y: pd.Series, spec: DatasetSpec,
                *, force_strategy: str | None = None) -> SplitData:
    """Produce train / validation / test folds.

    For a dataset whose rows are chronological, a temporal split is the
    honest one: in deployment you always predict forward in time. We use it
    when the source is ordered and a time column exists. Otherwise we fall
    back to a stratified random split.
    """
    base = load_config("base")
    seed = get_seed()
    test_size = float(base["split"]["test_size"])
    val_size = float(base["split"]["val_size"])

    X, y, n_dropped = drop_cross_split_duplicates(X, y)

    use_temporal = force_strategy == "temporal" or (
        force_strategy is None and spec.chronological and spec.time_column is not None
    )

    if use_temporal:
        # Rows are already in source order; slice by position.
        n = len(X)
        n_test = int(round(n * test_size))
        n_remaining = n - n_test
        n_val = int(round(n_remaining * val_size))
        n_train = n_remaining - n_val
        idx = np.arange(n)
        tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_remaining:]
        strategy = "temporal"
        # A temporal split can put zero fraud in a fold on a very sparse
        # dataset. That makes the fold useless, so fall back explicitly.
        if y.iloc[va].sum() == 0 or y.iloc[te].sum() == 0:
            LOGGER.warning(
                "Temporal split left a fold with no fraud cases; "
                "falling back to a stratified random split"
            )
            use_temporal = False
        else:
            splits = (X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va],
                      X.iloc[te], y.iloc[te])

    if not use_temporal:
        strategy = "stratified_random"
        stratify = y if base["split"].get("stratify", True) else None
        X_rem, X_test, y_rem, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_rem, y_rem, test_size=val_size, random_state=seed,
            stratify=y_rem if stratify is not None else None,
        )
        splits = (X_train, y_train, X_val, y_val, X_test, y_test)

    data = SplitData(
        X_train=splits[0].reset_index(drop=True), y_train=splits[1].reset_index(drop=True),
        X_val=splits[2].reset_index(drop=True), y_val=splits[3].reset_index(drop=True),
        X_test=splits[4].reset_index(drop=True), y_test=splits[5].reset_index(drop=True),
        strategy=strategy, n_duplicates_dropped=n_dropped,
    )
    LOGGER.info("Split (%s): train=%d/%d fraud, val=%d/%d, test=%d/%d",
                strategy, len(data.X_train), int(data.y_train.sum()),
                len(data.X_val), int(data.y_val.sum()),
                len(data.X_test), int(data.y_test.sum()))
    return data


def build_preprocessor(spec: DatasetSpec, X: pd.DataFrame) -> ColumnTransformer:
    """A ColumnTransformer that is fitted on training data only.

    Numeric: median impute -> standard scale.
    Categorical: most-frequent impute -> one-hot (unknown categories at
    inference time are encoded as all-zeros rather than raising).
    """
    numeric = [c for c in spec.numeric_features if c in X.columns]
    categorical = [c for c in spec.categorical_features if c in X.columns]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                 min_frequency=0.001)),
    ])

    transformers = []
    if numeric:
        transformers.append(("num", numeric_pipe, numeric))
    if categorical:
        transformers.append(("cat", categorical_pipe, categorical))
    if not transformers:
        raise ValueError("No usable numeric or categorical features for preprocessing")

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def fit_preprocessor(spec: DatasetSpec, X_train: pd.DataFrame
                     ) -> tuple[ColumnTransformer, list[str]]:
    """Fit on the training fold and return the transformer + output names."""
    pre = build_preprocessor(spec, X_train)
    pre.fit(X_train)
    try:
        names = [str(n) for n in pre.get_feature_names_out()]
    except Exception:  # pragma: no cover - defensive
        names = [f"f{i}" for i in range(pre.transform(X_train.head(1)).shape[1])]
    LOGGER.info("Preprocessor fitted on %d training rows -> %d output features",
                len(X_train), len(names))
    return pre, names


def transform(pre: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    out = np.asarray(pre.transform(X), dtype=np.float64)
    if not np.isfinite(out).all():
        # Should not happen after imputation, but a non-finite value silently
        # poisons every downstream model, so fail loudly rather than fit on NaN.
        bad = int((~np.isfinite(out)).sum())
        raise ValueError(f"Preprocessed matrix contains {bad} non-finite values")
    return out


def apply_imbalance(X: np.ndarray, y: np.ndarray, strategy: str | None = None
                    ) -> tuple[np.ndarray, np.ndarray, str]:
    """Resample the TRAINING fold only.

    Returns the (possibly unchanged) arrays and the strategy actually applied.
    Callers must never pass validation or test data here.
    """
    cfg = load_config("classical")["imbalance"]
    strategy = strategy or cfg.get("strategy", "class_weight")
    seed = get_seed()

    if strategy in ("none", "class_weight"):
        # class_weight is handled inside each estimator, not by resampling.
        return X, y, strategy

    try:
        if strategy == "smote":
            from imblearn.over_sampling import SMOTE

            ratio = float(cfg.get("smote_sampling_strategy", 0.1))
            current = y.mean() / (1 - y.mean()) if y.mean() < 1 else 1.0
            if current >= ratio:
                LOGGER.info("SMOTE skipped: minority ratio %.4f already >= %.4f",
                            current, ratio)
                return X, y, "none"
            sampler = SMOTE(sampling_strategy=ratio, random_state=seed, k_neighbors=5)
        elif strategy == "undersample":
            from imblearn.under_sampling import RandomUnderSampler

            sampler = RandomUnderSampler(
                sampling_strategy=float(cfg.get("undersample_ratio", 0.05)),
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown imbalance strategy {strategy!r}")

        X_res, y_res = sampler.fit_resample(X, y)
        LOGGER.info("Imbalance strategy %s: %d -> %d rows (%d -> %d fraud)",
                    strategy, len(y), len(y_res), int(y.sum()), int(y_res.sum()))
        return X_res, y_res, strategy
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Imbalance strategy %s failed (%r); continuing unresampled",
                     strategy, exc)
        return X, y, "none"
