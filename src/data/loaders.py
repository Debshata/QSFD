"""Loading datasets into a normalised (X, y, spec) triple."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.data.registry import DatasetSpec, build_spec, normalise_target
from src.utils.config import get_seed, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

# Rows read when only the schema is needed. Large enough for cardinality
# heuristics to be meaningful, small enough to be instant on a 350 MB CSV.
SCHEMA_SAMPLE_ROWS = 50_000


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    X: pd.DataFrame
    y: pd.Series
    n_rows_original: int
    subsampled: bool

    @property
    def fraud_count(self) -> int:
        return int(self.y.sum())

    @property
    def fraud_rate(self) -> float:
        return float(self.y.mean())


def read_schema_sample(file_path: str, nrows: int = SCHEMA_SAMPLE_ROWS) -> pd.DataFrame:
    return pd.read_csv(file_path, nrows=nrows, low_memory=False)


def build_spec_for(name: str, entry: dict[str, Any], file_path: str) -> DatasetSpec:
    return build_spec(name, entry, file_path, read_schema_sample(file_path))


def load_dataset(name: str, entry: dict[str, Any], file_path: str,
                 *, max_rows: int | None = -1,
                 spec: DatasetSpec | None = None) -> LoadedDataset:
    """Load one dataset as features + binary target.

    ``max_rows`` of -1 means "use base.yaml"; ``None`` means no cap. Capping is
    a stratified, seeded subsample taken *before* any splitting, so the
    train/val/test split still sees a representative population.
    """
    if max_rows == -1:
        max_rows = load_config("base").get("max_rows_for_training")

    spec = spec or build_spec_for(name, entry, file_path)

    # Read only the columns we will actually use plus the target.
    usecols = spec.feature_columns + [spec.target]
    df = pd.read_csv(file_path, usecols=usecols, low_memory=False)
    n_original = len(df)

    y = normalise_target(df[spec.target], spec.target_positive)
    X = df.drop(columns=[spec.target])

    subsampled = False
    if max_rows and n_original > max_rows:
        # Stratified downsample: keep every fraud case we can, then fill with
        # legitimate rows. Throwing away positives would distort the base rate.
        frac = max_rows / n_original
        pos_idx = y[y == 1].index
        neg_idx = y[y == 0].index
        keep_pos = pos_idx if len(pos_idx) <= max_rows // 2 else \
            pd.Index(pd.Series(pos_idx).sample(int(len(pos_idx) * frac),
                                               random_state=get_seed()))
        n_neg = max(0, max_rows - len(keep_pos))
        keep_neg = pd.Index(pd.Series(neg_idx).sample(min(n_neg, len(neg_idx)),
                                                      random_state=get_seed()))
        keep = keep_pos.union(keep_neg).sort_values()
        X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
        subsampled = True
        LOGGER.info("%s: subsampled %d -> %d rows (kept %d/%d fraud cases)",
                    name, n_original, len(X), int(y.sum()), int(len(pos_idx)))

    LOGGER.info("%s: loaded %d rows, %d features, %d fraud (%.4f%%)",
                name, len(X), X.shape[1], int(y.sum()), 100 * float(y.mean()))
    return LoadedDataset(spec=spec, X=X, y=y.rename("target"),
                         n_rows_original=n_original, subsampled=subsampled)


def load_primary(**kwargs: Any) -> LoadedDataset:
    """Load the dataset configured as ``primary_dataset``."""
    from src.data.registry import available_datasets

    cfg = load_config("datasets")
    name = cfg["primary_dataset"]
    available = available_datasets()
    if name not in available:
        raise RuntimeError(
            f"Primary dataset {name!r} is not available. "
            f"Available: {sorted(available)}. Run: python -m src.data.download"
        )
    rec = available[name]
    return load_dataset(name, rec["entry"], rec["file"], **kwargs)
