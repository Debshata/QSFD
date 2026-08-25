"""Schema validation, profiling and leakage checks.

Two jobs:

1. **Profile** each dataset for the dataset report (rows, class balance,
   missing values, duplicates, column roles).
2. **Guard** against the failure modes that make fraud models look better than
   they are -- target-derived columns, identifiers used as features, and
   duplicate transactions straddling the train/test boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.registry import DatasetSpec
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

# A feature correlating with the target this strongly is almost never real
# signal in a fraud dataset -- it is usually a transformed copy of the label.
LEAKAGE_CORRELATION_THRESHOLD = 0.95


class ValidationError(Exception):
    """Raised when a dataset violates a hard contract."""


@dataclass
class ValidationReport:
    dataset: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False
        LOGGER.error("[%s] %s", self.dataset, msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        LOGGER.warning("[%s] %s", self.dataset, msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def profile_dataset(name: str, spec: DatasetSpec, X: pd.DataFrame,
                    y: pd.Series, n_rows_original: int) -> dict[str, Any]:
    """Row-level statistics used by the dataset report."""
    fraud = int(y.sum())
    total = int(len(y))
    missing = X.isna().sum()
    duplicates = int(X.duplicated().sum())
    return {
        "dataset": name,
        "source": spec.kaggle_id,
        "file": spec.file.replace("\\", "/").rsplit("/", 1)[-1],
        "rows": total,
        "rows_in_source_file": n_rows_original,
        "columns": int(X.shape[1] + 1),
        "target": spec.target,
        "fraud_count": fraud,
        "legitimate_count": total - fraud,
        "fraud_percentage": round(100.0 * fraud / total, 6) if total else 0.0,
        "imbalance_ratio": round((total - fraud) / fraud, 2) if fraud else float("inf"),
        "missing_values": int(missing.sum()),
        "columns_with_missing": int((missing > 0).sum()),
        "duplicate_rows": duplicates,
        "duplicate_percentage": round(100.0 * duplicates / total, 4) if total else 0.0,
        "n_numeric_features": len(spec.numeric_features),
        "n_categorical_features": len(spec.categorical_features),
        "numeric_features": "|".join(spec.numeric_features),
        "categorical_features": "|".join(spec.categorical_features),
        "identifier_features": "|".join(spec.identifier_features),
        "potential_leakage_features": "|".join(spec.potential_leakage_features),
        "excluded_features": "|".join(spec.unusable_features),
        "chronological": spec.chronological,
        "time_column": spec.time_column or "",
        "amount_column": spec.amount_column or "",
    }


def check_target_correlation(X: pd.DataFrame, y: pd.Series,
                             threshold: float = LEAKAGE_CORRELATION_THRESHOLD
                             ) -> dict[str, float]:
    """Numeric features whose |correlation| with the target is suspiciously high."""
    suspicious: dict[str, float] = {}
    numeric = X.select_dtypes(include=[np.number])
    if numeric.empty:
        return suspicious
    yv = y.astype(float).to_numpy()
    for col in numeric.columns:
        values = numeric[col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if np.std(values) == 0:
            continue
        corr = float(np.corrcoef(values, yv)[0, 1])
        if abs(corr) >= threshold:
            suspicious[col] = round(corr, 4)
    return suspicious


def validate_dataset(name: str, spec: DatasetSpec, X: pd.DataFrame,
                     y: pd.Series) -> ValidationReport:
    """Hard contract checks. Errors block training; warnings are recorded."""
    report = ValidationReport(dataset=name, passed=True)

    if len(X) != len(y):
        report.add_error(f"X/y length mismatch: {len(X)} vs {len(y)}")
    if len(X) == 0:
        report.add_error("dataset is empty")
        return report

    classes = set(pd.unique(y))
    if not classes.issubset({0, 1}):
        report.add_error(f"target is not binary 0/1 after normalisation: {classes}")
    if y.sum() == 0:
        report.add_error("no positive (fraud) samples present")
    if y.sum() == len(y):
        report.add_error("no negative (legitimate) samples present")

    # The target must never survive into the feature matrix.
    if spec.target in X.columns:
        report.add_error(f"target column {spec.target!r} is present in X")

    # Identifiers and known-leaky columns must have been removed.
    for col in spec.identifier_features + spec.potential_leakage_features:
        if col in X.columns:
            report.add_error(f"excluded column {col!r} leaked into the feature matrix")

    if X.shape[1] == 0:
        report.add_error("no usable features remain after column filtering")
        return report

    suspicious = check_target_correlation(X, y)
    for col, corr in suspicious.items():
        report.add_error(
            f"feature {col!r} correlates {corr} with the target "
            "-- almost certainly a transformed copy of the label"
        )

    n_missing = int(X.isna().sum().sum())
    if n_missing:
        report.add_warning(f"{n_missing} missing values present (imputed in-pipeline)")

    n_dupes = int(X.duplicated().sum())
    if n_dupes:
        report.add_warning(
            f"{n_dupes} duplicate feature rows ({100 * n_dupes / len(X):.2f}%) "
            "-- deduplicated before splitting to avoid train/test overlap"
        )

    fraud_rate = float(y.mean())
    if fraud_rate > 0.4:
        report.add_warning(
            f"fraud rate is {fraud_rate:.1%}: this dataset is balanced by "
            "construction and does not reflect a real base rate"
        )
    return report


def drop_cross_split_duplicates(X: pd.DataFrame, y: pd.Series
                                ) -> tuple[pd.DataFrame, pd.Series, int]:
    """Remove duplicate feature rows before splitting.

    Identical transactions appearing in both train and test inflate every
    metric: the model has literally memorised the test row. We keep the first
    occurrence only, and only when the label agrees -- a duplicate row with
    conflicting labels is genuinely ambiguous and both copies are kept so the
    model sees the ambiguity rather than a coin flip.
    """
    combined = X.copy()
    combined["__y__"] = y.to_numpy()
    dup_mask = combined.duplicated(keep="first")
    n_dropped = int(dup_mask.sum())
    if n_dropped:
        keep = ~dup_mask
        X = X.loc[keep].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True)
        LOGGER.info("Dropped %d exact duplicate rows (features + label) before splitting",
                    n_dropped)
    return X, y, n_dropped
