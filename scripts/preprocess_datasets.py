"""Preprocess the four raw datasets into model-ready and quantum-ready files.

Reuses the project's existing data layer (src.data.registry / loaders /
preprocessing / validators) UNCHANGED for schema detection, leakage
screening, temporal-vs-stratified splitting and fit-on-train-only
encoding/scaling. This script only orchestrates those pieces and writes the
physical output files the rest of this task's tooling expects.

Sequence per dataset, exactly as required:

    raw file -> deterministic cleaning (existing loader/registry logic)
             -> train / validation / test split (existing make_splits)
             -> fit encoder+scaler on TRAIN ONLY (existing fit_preprocessor)
             -> transform validation, transform test
             -> quantum-ready reduced features (fit on TRAIN ONLY)

No resampling (SMOTE etc.) is applied to anything written to disk -- the
saved datasets are the original class distribution; resampling remains a
training-time, in-memory concern for the model pipelines.

    python scripts/preprocess_datasets.py
    python scripts/preprocess_datasets.py --dataset creditcardfraud_ulb
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.loaders import build_spec_for, load_dataset  # noqa: E402
from src.data.preprocessing import fit_preprocessor, make_splits, transform  # noqa: E402
from src.data.registry import DatasetSpec  # noqa: E402
from src.data.validators import (  # noqa: E402
    check_target_correlation,
    profile_dataset,
    validate_dataset,
)
from src.utils.config import get_path, get_seed, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.reproducibility import environment_info, set_global_seed  # noqa: E402
from scripts.download_datasets import FRIENDLY_NAMES  # noqa: E402

LOGGER = get_logger("preprocess_datasets")

STANDARD_TARGET = "is_fraud"           # standardised output target name
QUANTUM_FEATURE_COUNTS = (2, 4, 6, 8)

# Datasets that ship a genuine wall-clock timestamp (as opposed to
# creditcardfraud / creditcard_2023's "Time" = seconds-since-first-row,
# which carries no calendar information and cannot be decomposed into
# hour-of-day/day-of-week without an arbitrary assumed epoch).
TIMESTAMP_COLUMNS = {
    "fraud_detection_sim": "trans_date_trans_time",
    "synthetic_cc_fraud": "timestamp",
}


def friendly_name(internal_name: str) -> str:
    return FRIENDLY_NAMES.get(internal_name, internal_name)


def raw_file_for(internal_name: str, entry: dict[str, Any]) -> Path:
    """Prefer the local data/raw copy; fall back to the kagglehub cache path."""
    raw_dir = Path(get_path("data_raw")) / friendly_name(internal_name)
    if raw_dir.exists():
        matches = sorted(raw_dir.glob("*.csv"))
        if matches:
            return matches[0]

    from src.data.download import load_manifest

    manifest = load_manifest()
    record = manifest.get(internal_name)
    if record and record.get("ok"):
        return Path(record["file"])
    raise FileNotFoundError(
        f"No local raw file for {internal_name!r} and no usable download "
        "manifest entry. Run scripts/download_datasets.py first."
    )


# --------------------------------------------------------------------------
# temporal feature engineering
# --------------------------------------------------------------------------

def engineer_temporal_features(file_path: Path, time_col: str,
                               n_rows: int) -> pd.DataFrame | None:
    """Deterministic, row-wise hour/day/month/cyclic features from a raw
    wall-clock timestamp column. Reads only that one column, in the same row
    order as the rest of the file, so it can be concatenated positionally.

    This is NOT future information: each row's engineered features derive
    only from that row's own timestamp, never from dataset-wide statistics.
    """
    try:
        raw = pd.read_csv(file_path, usecols=[time_col], low_memory=False)
    except ValueError:
        LOGGER.warning("Timestamp column %r not found in %s; skipping "
                       "temporal feature engineering", time_col, file_path.name)
        return None

    ts = pd.to_datetime(raw[time_col], errors="coerce")
    if ts.isna().mean() > 0.5:
        LOGGER.warning("%s: over half of %r failed to parse as a timestamp; "
                       "skipping temporal feature engineering",
                       file_path.name, time_col)
        return None

    hour = ts.dt.hour.fillna(0).astype(int)
    dow = ts.dt.dayofweek.fillna(0).astype(int)          # 0=Monday
    out = pd.DataFrame({
        "txn_hour": hour,
        "txn_day": ts.dt.day.fillna(1).astype(int),
        "txn_day_of_week": dow,
        "txn_month": ts.dt.month.fillna(1).astype(int),
        "txn_is_weekend": (dow >= 5).astype(int),
        "txn_hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "txn_hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "txn_dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "txn_dow_cos": np.cos(2 * np.pi * dow / 7.0),
    })
    return out.iloc[:n_rows].reset_index(drop=True)


# --------------------------------------------------------------------------
# quantum-ready feature reduction
# --------------------------------------------------------------------------

def compute_feature_scores(Xtr_scaled: np.ndarray, ytr: np.ndarray,
                           method: str, seed: int,
                           max_score_rows: int = 100_000) -> np.ndarray:
    """Rank every feature ONCE on TRAINING data only.

    Called once per dataset regardless of how many k values are requested --
    mutual_info_classif and RandomForest importance are each O(n) or worse
    per feature, so recomputing per-k would redo the same expensive scan four
    times for no reason. A large training fold is subsampled (seeded,
    stratified) purely for the SCORING step; the features themselves are
    still sliced from the FULL training matrix afterwards.
    """
    Xs, ys = Xtr_scaled, ytr
    if len(ys) > max_score_rows:
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(ys == 1)
        neg = np.flatnonzero(ys == 0)
        n_pos = min(len(pos), max_score_rows // 2)
        n_neg = min(len(neg), max_score_rows - n_pos)
        idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                              rng.choice(neg, n_neg, replace=False)])
        Xs, ys = Xtr_scaled[idx], ytr[idx]
        LOGGER.info("Feature scoring (%s) subsampled to %d rows for speed "
                    "(scoring only; feature values themselves use the full "
                    "training set)", method, len(ys))

    if method == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif

        return mutual_info_classif(Xs, ys, random_state=seed)
    if method == "anova_f":
        from sklearn.feature_selection import f_classif

        scores, _ = f_classif(Xs, ys)
        return np.nan_to_num(scores)
    if method == "random_forest_importance":
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        rf.fit(Xs, ys)
        return rf.feature_importances_
    raise ValueError(f"Unknown quantum feature-selection method {method!r}")


def select_quantum_features(Xtr_scaled: np.ndarray, ytr: np.ndarray,
                            feature_names: list[str], k: int, method: str,
                            seed: int, *, scores: np.ndarray | None = None
                            ) -> dict[str, Any]:
    """Return the top-k features by a precomputed (or PCA) ranking.

    Methods are exploratory, not a claim of optimality -- the choice actually
    used is recorded in metadata so it can be audited or changed later.
    """
    k = min(k, Xtr_scaled.shape[1])
    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=k, random_state=seed)
        pca.fit(Xtr_scaled)
        return {
            "method": "pca", "k": k, "selected_indices": None,
            "selected_feature_names": [f"PC{i + 1}" for i in range(k)],
            "explained_variance_ratio": round(float(pca.explained_variance_ratio_.sum()), 4),
            "transformer": pca,
        }

    if scores is None:
        scores = compute_feature_scores(Xtr_scaled, ytr, method, seed)

    order = np.argsort(scores)[::-1][:k]
    return {
        "method": method, "k": k, "selected_indices": order.tolist(),
        "selected_feature_names": [feature_names[i] for i in order],
        "scores": {feature_names[i]: round(float(scores[i]), 6) for i in order},
        "transformer": None,
    }


# --------------------------------------------------------------------------
# per-dataset pipeline
# --------------------------------------------------------------------------

def process_one(internal_name: str, *, max_rows: int | None = None,
                quantum_method: str = "mutual_info") -> dict[str, Any]:
    t_start = time.perf_counter()
    friendly = friendly_name(internal_name)
    LOGGER.info("=" * 70)
    LOGGER.info("Processing %s (%s)", internal_name, friendly)

    ds_cfg = load_config("datasets")["datasets"][internal_name]
    raw_file = raw_file_for(internal_name, ds_cfg)
    spec = build_spec_for(internal_name, ds_cfg, str(raw_file))

    loaded = load_dataset(internal_name, ds_cfg, str(raw_file),
                          max_rows=max_rows, spec=spec)
    X, y = loaded.X, loaded.y

    # -- leakage screen (feeds data/metadata/leakage_audit.md) -------------
    validation = validate_dataset(internal_name, spec, X, y)
    correlated = check_target_correlation(X, y)
    profile = profile_dataset(internal_name, spec, X, y, loaded.n_rows_original)

    # -- split (existing convention: temporal where chronological, else
    #    stratified random; see docs/DECISIONS.md D5) ----------------------
    splits = make_splits(X, y, spec)

    # -- optional temporal feature engineering, row-aligned to the ORIGINAL
    #    (pre-split, pre-dedup) file, then re-sliced onto each split's index
    #    via the row position each split already carries ------------------
    temporal_cols: list[str] = []
    if internal_name in TIMESTAMP_COLUMNS:
        # make_splits drops cross-split duplicates and reindexes 0..n-1, so we
        # cannot positionally re-align after that point. Recompute engineered
        # features against the ORIGINAL (pre-dedup) row count instead, then
        # reindex using each split's retained original positions is not
        # available post-hoc -- instead we engineer BEFORE dedup, on the full
        # loaded frame, and re-run the identical split so indices line up.
        temporal_full = engineer_temporal_features(
            raw_file, TIMESTAMP_COLUMNS[internal_name], len(loaded.X))
        if temporal_full is not None:
            X_with_time = X.copy()
            for col in temporal_full.columns:
                X_with_time[col] = temporal_full[col].to_numpy()
            splits = make_splits(X_with_time, y, spec)
            temporal_cols = list(temporal_full.columns)
            LOGGER.info("%s: added %d temporal features from %r",
                        internal_name, len(temporal_cols), TIMESTAMP_COLUMNS[internal_name])
    else:
        LOGGER.info("%s: no calendar timestamp available (Time is a row-order "
                    "index, not a wall-clock field); temporal feature "
                    "engineering skipped by design", internal_name)

    # -- fit preprocessor on TRAIN ONLY -------------------------------------
    preprocessor, feature_names = fit_preprocessor(spec, splits.X_train)
    # Temporal columns are numeric and already dropped in classify_columns's
    # scan? No -- they were injected after spec was built, so they are not
    # part of spec.feature_columns / the ColumnTransformer. Scale them
    # separately with the same discipline (fit on train only).
    temporal_scaler = None
    if temporal_cols:
        from sklearn.preprocessing import StandardScaler

        temporal_scaler = StandardScaler().fit(splits.X_train[temporal_cols])

    def build_clean_frame(Xs: pd.DataFrame, ys: pd.Series) -> pd.DataFrame:
        frame = Xs.copy()
        frame[STANDARD_TARGET] = ys.to_numpy()
        return frame

    def build_scaled_frame(Xs: pd.DataFrame, ys: pd.Series) -> pd.DataFrame:
        base_cols = [c for c in Xs.columns if c not in temporal_cols]
        mat = transform(preprocessor, Xs[base_cols])
        frame = pd.DataFrame(mat, columns=feature_names)
        if temporal_cols:
            temporal_scaled = temporal_scaler.transform(Xs[temporal_cols])
            for i, col in enumerate(temporal_cols):
                frame[col] = temporal_scaled[:, i]
        frame[STANDARD_TARGET] = ys.to_numpy()
        return frame

    clean = {k: build_clean_frame(getattr(splits, f"X_{k}"), getattr(splits, f"y_{k}"))
             for k in ("train", "val", "test")}
    scaled = {k: build_scaled_frame(getattr(splits, f"X_{k}"), getattr(splits, f"y_{k}"))
             for k in ("train", "val", "test")}

    # -- quantum-ready reduced feature sets, fit on TRAIN ONLY --------------
    all_feature_names = list(feature_names) + list(temporal_cols)
    Xtr_full = scaled["train"][all_feature_names].to_numpy()
    ytr = splits.y_train.to_numpy()

    precomputed_scores = (
        compute_feature_scores(Xtr_full, ytr, quantum_method, get_seed())
        if quantum_method != "pca" else None
    )
    quantum_sets: dict[int, dict[str, Any]] = {}
    for k in QUANTUM_FEATURE_COUNTS:
        if k > Xtr_full.shape[1]:
            continue
        selection = select_quantum_features(Xtr_full, ytr, all_feature_names, k,
                                            quantum_method, get_seed(),
                                            scores=precomputed_scores)
        if selection["transformer"] is not None:            # pca branch
            reducer = selection["transformer"]
            reduced = {
                split: reducer.transform(scaled[split][all_feature_names].to_numpy())
                for split in ("train", "val", "test")
            }
            cols = selection["selected_feature_names"]
        else:
            idx = selection["selected_indices"]
            cols = selection["selected_feature_names"]
            reduced = {
                split: scaled[split][all_feature_names].to_numpy()[:, idx]
                for split in ("train", "val", "test")
            }
        frame = pd.concat([
            pd.DataFrame(reduced[s], columns=cols).assign(
                split=s, **{STANDARD_TARGET: getattr(splits, f"y_{s}").to_numpy()})
            for s in ("train", "val", "test")
        ], ignore_index=True)
        quantum_sets[k] = {"frame": frame, "selection": selection}
        LOGGER.info("%s: quantum_%d_features -> %s (method=%s)",
                    internal_name, k, cols, quantum_method)

    # -- write everything ----------------------------------------------------
    out_dir = Path(get_path("data_processed")) / friendly
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean, interpretable splits stay CSV (small: original column count,
    # no one-hot expansion). Scaled/model-ready and quantum-reduced matrices
    # use Parquet -- one-hot expansion made the equivalent CSV 260MB+ on the
    # synthetic dataset alone, and the full-size kartik2112 dataset (2.6x the
    # rows) would be worse. Parquet keeps this efficient without losing any
    # information loaders need; the task's own instructions permit this
    # substitution explicitly for "unnecessarily huge" CSV output.
    written: list[str] = []
    for split_name, frame in clean.items():
        name = split_name if split_name != "val" else "validation"
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written.append(path.name)
    for split_name, frame in scaled.items():
        name = split_name if split_name != "val" else "validation"
        path = out_dir / f"{name}_scaled.parquet"
        frame.to_parquet(path, index=False)
        written.append(path.name)
    for k, payload in quantum_sets.items():
        path = out_dir / f"quantum_{k}_features.parquet"
        payload["frame"].to_parquet(path, index=False)
        written.append(path.name)

    removed_columns = sorted(set(
        spec.identifier_features + spec.potential_leakage_features
        + spec.unusable_features + spec.drop_as_feature
    ))

    metadata = {
        "dataset": internal_name,
        "friendly_name": friendly,
        "kaggle_id": spec.kaggle_id,
        "raw_file": str(raw_file),
        "standardised_target": STANDARD_TARGET,
        "original_target_column": spec.target,
        "original_target_positive_values": [str(v) for v in spec.target_positive],
        "split_strategy": splits.strategy,
        "duplicates_dropped_before_split": splits.n_duplicates_dropped,
        "split_summary": splits.summary(),
        "removed_columns": removed_columns,
        "identifier_columns": spec.identifier_features,
        "potential_leakage_columns": spec.potential_leakage_features,
        "unusable_columns": spec.unusable_features,
        "numeric_features": spec.numeric_features,
        "categorical_features": spec.categorical_features,
        "temporal_features_engineered": temporal_cols,
        "preprocessing": {
            "numeric": "median impute -> StandardScaler, fit on train only",
            "categorical": "most-frequent impute -> OneHotEncoder(handle_unknown='ignore'), fit on train only",
            "temporal": ("StandardScaler on engineered hour/day/month/cyclic "
                        "features, fit on train only" if temporal_cols else "n/a"),
            "resampling_applied_to_saved_files": "none (class_weight/SMOTE remain training-time-only concerns)",
            "file_formats": "train/validation/test.csv are the clean, "
                            "un-scaled, human-readable features (CSV). "
                            "*_scaled and quantum_*_features are the "
                            "one-hot-expanded, scaled, model-ready matrices "
                            "(Parquet, to avoid multi-hundred-MB CSVs).",
        },
        "output_files": written,
        "transformed_feature_names": feature_names,
        "all_model_ready_feature_names": all_feature_names,
        "quantum_feature_sets": {
            str(k): {
                "method": payload["selection"]["method"],
                "k": payload["selection"]["k"],
                "selected_features": payload["selection"]["selected_feature_names"],
                "explained_variance_ratio": payload["selection"].get("explained_variance_ratio"),
                "note": "Selected on TRAINING data only; not claimed optimal "
                        "without experimental validation.",
            }
            for k, payload in quantum_sets.items()
        },
        "validation": validation.to_dict(),
        "suspicious_target_correlations": correlated,
        "seed": get_seed(),
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": environment_info(),
        "processing_seconds": round(time.perf_counter() - t_start, 2),
    }
    (out_dir / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    LOGGER.info("%s: wrote processed files -> %s (%.1fs)",
                internal_name, out_dir, time.perf_counter() - t_start)
    return {"internal_name": internal_name, "friendly_name": friendly,
            "spec": spec, "profile": profile, "validation": validation,
            "splits": splits, "metadata": metadata, "out_dir": str(out_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess raw datasets")
    parser.add_argument("--dataset", default=None,
                        help="friendly name (e.g. creditcardfraud_ulb) or "
                             "internal name (e.g. creditcardfraud); default: all")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="cap rows per dataset (default: full file)")
    args = parser.parse_args()

    set_global_seed(get_seed())
    ds_cfg = load_config("datasets")["datasets"]
    reverse = {v: k for k, v in FRIENDLY_NAMES.items()}

    if args.dataset:
        internal = reverse.get(args.dataset, args.dataset)
        if internal not in ds_cfg:
            LOGGER.error("Unknown dataset %r", args.dataset)
            return 1
        targets = [internal]
    else:
        targets = [n for n, e in ds_cfg.items() if e.get("enabled", True)]

    results = []
    failures = []
    for name in targets:
        try:
            results.append(process_one(name, max_rows=args.max_rows))
        except Exception as exc:  # noqa: BLE001 - one dataset must not sink the rest
            LOGGER.exception("Preprocessing failed for %s: %r", name, exc)
            failures.append({"dataset": name, "error": repr(exc)})

    if failures:
        LOGGER.warning("Failed datasets: %s", [f["dataset"] for f in failures])

    LOGGER.info("Preprocessed %d/%d datasets successfully", len(results), len(targets))
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
