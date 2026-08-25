"""Dataset registry / adapter layer.

The four sources have four different schemas, so nothing downstream is allowed
to know a column name. A :class:`DatasetSpec` normalises each source into a
common contract:

    features (X)  +  binary target (y, 1 = fraud)  +  column roles

Column roles (numeric / categorical / identifier / leakage) are resolved by
combining the explicit config with name- and cardinality-based heuristics, so
an unseen dataset still gets a defensible answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class DatasetSpec:
    """Everything the pipeline needs to know about one dataset."""

    name: str
    kaggle_id: str
    file: str
    description: str
    target: str
    target_positive: list[Any]
    chronological: bool
    time_column: str | None
    amount_column: str | None
    drop_as_feature: list[str] = field(default_factory=list)
    identifier_features: list[str] = field(default_factory=list)
    potential_leakage_features: list[str] = field(default_factory=list)
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    unusable_features: list[str] = field(default_factory=list)

    @property
    def feature_columns(self) -> list[str]:
        return list(self.numeric_features) + list(self.categorical_features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kaggle_id": self.kaggle_id,
            "file": self.file,
            "description": self.description,
            "target": self.target,
            "chronological": self.chronological,
            "time_column": self.time_column,
            "amount_column": self.amount_column,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "identifier_features": self.identifier_features,
            "potential_leakage_features": self.potential_leakage_features,
            "unusable_features": self.unusable_features,
        }


def _matches_any(name: str, patterns: list[str]) -> bool:
    low = name.lower()
    return any(re.search(p, low) for p in patterns)


def detect_target(df: pd.DataFrame, configured: str | None,
                  candidates: list[str]) -> str:
    """Return the target column, preferring the configured one."""
    if configured and configured in df.columns:
        return configured
    for cand in candidates:
        if cand in df.columns:
            LOGGER.info("Auto-detected target column %r", cand)
            return cand
    # Last resort: a binary column whose name mentions fraud.
    for col in df.columns:
        if "fraud" in col.lower() and df[col].nunique(dropna=True) <= 2:
            LOGGER.warning("Falling back to target column %r", col)
            return col
    raise ValueError(f"Could not identify a target column among {list(df.columns)}")


def normalise_target(series: pd.Series, positive_values: list[Any]) -> pd.Series:
    """Map an arbitrary label encoding onto {0, 1} with 1 = fraud.

    Values not listed as positive are treated as legitimate. A column that
    matches no positive value at all is an error, not a silent all-zero target.
    """
    positives = {str(v).strip().lower() for v in positive_values}
    as_str = series.astype("string").str.strip().str.lower()
    y = as_str.isin(positives).astype("int8")
    if int(y.sum()) == 0:
        uniques = sorted(str(v) for v in series.dropna().unique()[:10])
        raise ValueError(
            "Target normalisation produced zero positives. "
            f"Observed values {uniques} match none of {sorted(positives)}."
        )
    return y


def classify_columns(df: pd.DataFrame, spec_target: str,
                     explicit_drop: list[str],
                     autodetect: dict) -> dict[str, list[str]]:
    """Assign every non-target column to a role."""
    id_patterns = autodetect["identifier_patterns"]
    leak_patterns = autodetect["leakage_patterns"]
    max_card = int(autodetect["max_categorical_cardinality"])

    identifiers: list[str] = []
    leakage: list[str] = []
    numeric: list[str] = []
    categorical: list[str] = []
    unusable: list[str] = []

    n_rows = len(df)
    for col in df.columns:
        if col == spec_target:
            continue
        if col in explicit_drop:
            # Explicitly excluded by config; still record which bucket it fell in.
            if _matches_any(col, id_patterns):
                identifiers.append(col)
            else:
                unusable.append(col)
            continue
        if _matches_any(col, leak_patterns):
            leakage.append(col)
            continue
        if _matches_any(col, id_patterns):
            identifiers.append(col)
            continue

        nunique = int(df[col].nunique(dropna=True))
        if nunique <= 1:
            unusable.append(col)          # constant column carries no signal
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            numeric.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # A numeric column unique on every row is an index, not a feature.
            if nunique == n_rows and n_rows > 1000:
                identifiers.append(col)
            else:
                numeric.append(col)
        else:
            if nunique > max_card:
                unusable.append(col)      # high-cardinality free text
            else:
                categorical.append(col)

    return {
        "numeric": numeric,
        "categorical": categorical,
        "identifier": identifiers,
        "leakage": leakage,
        "unusable": unusable,
    }


def build_spec(name: str, entry: dict[str, Any], file_path: str,
               sample: pd.DataFrame) -> DatasetSpec:
    """Build a :class:`DatasetSpec` from config plus an inspected sample."""
    autodetect = load_config("datasets")["autodetect"]

    target = detect_target(sample, entry.get("target"),
                           autodetect["target_candidates"])
    explicit_drop = list(entry.get("drop_columns") or []) + \
        list(entry.get("drop_as_feature") or [])
    roles = classify_columns(sample, target, explicit_drop, autodetect)

    spec = DatasetSpec(
        name=name,
        kaggle_id=entry["kaggle_id"],
        file=file_path,
        description=(entry.get("description") or "").strip(),
        target=target,
        target_positive=entry.get("target_positive") or [1],
        chronological=bool(entry.get("chronological", False)),
        time_column=entry.get("time_column"),
        amount_column=entry.get("amount_column"),
        drop_as_feature=explicit_drop,
        identifier_features=roles["identifier"],
        potential_leakage_features=roles["leakage"],
        numeric_features=roles["numeric"],
        categorical_features=roles["categorical"],
        unusable_features=roles["unusable"],
    )
    LOGGER.info(
        "%s: target=%s numeric=%d categorical=%d identifiers=%d leakage=%d",
        name, target, len(spec.numeric_features), len(spec.categorical_features),
        len(spec.identifier_features), len(spec.potential_leakage_features),
    )
    if spec.potential_leakage_features:
        LOGGER.warning("%s: excluding suspected leakage columns %s",
                       name, spec.potential_leakage_features)
    return spec


def available_datasets() -> dict[str, dict[str, Any]]:
    """Datasets that are both enabled in config and present on disk."""
    from src.data.download import load_manifest

    cfg = load_config("datasets")["datasets"]
    manifest = load_manifest()
    out: dict[str, dict[str, Any]] = {}
    for name, entry in cfg.items():
        if not entry.get("enabled", True):
            continue
        rec = manifest.get(name)
        if rec and rec.get("ok"):
            out[name] = {"entry": entry, "file": rec["file"]}
        else:
            LOGGER.warning("Dataset %s is enabled but not available locally", name)
    return out
