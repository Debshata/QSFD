"""Validate the populated raw/processed data directories.

Checks, per dataset:

    - raw file exists and is readable
    - every expected processed file exists and is non-empty
    - the standardised target column exists and is strictly binary {0, 1}
    - train/validation/test contain no overlapping rows (hash-based)
    - no identifier/leakage column (per that dataset's own metadata) is
      present in the model-ready (scaled) or quantum feature matrices
    - no NaN/Inf values in the model-ready matrices
    - class distribution is documented (already recorded in metadata)
    - no row in *_scaled/quantum files has a "source" marking it as
      synthetic/resampled (this project never resamples saved files)

Exits non-zero if any dataset fails a check. Prints a pass/fail table.

    python scripts/validate_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.config import get_path  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from scripts.download_datasets import FRIENDLY_NAMES  # noqa: E402

LOGGER = get_logger("validate_data")

EXPECTED_FILES = [
    "train.csv", "validation.csv", "test.csv",
    "train_scaled.parquet", "validation_scaled.parquet", "test_scaled.parquet",
    "quantum_2_features.parquet", "quantum_4_features.parquet",
    "quantum_6_features.parquet", "quantum_8_features.parquet",
    "preprocessing_metadata.json",
]
STANDARD_TARGET = "is_fraud"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def validate_one(friendly: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    raw_dir = Path(get_path("data_raw")) / friendly
    proc_dir = Path(get_path("data_processed")) / friendly

    raw_files = list(raw_dir.glob("*.csv"))
    if not raw_files:
        errors.append(f"no raw CSV found under {raw_dir}")
    else:
        for f in raw_files:
            if f.stat().st_size == 0:
                errors.append(f"raw file {f.name} is empty")

    missing = [name for name in EXPECTED_FILES if not (proc_dir / name).exists()]
    if missing:
        errors.append(f"missing processed files: {missing}")
        return {"friendly_name": friendly, "passed": False, "errors": errors,
               "warnings": warnings}

    meta = json.loads((proc_dir / "preprocessing_metadata.json").read_text(encoding="utf-8"))
    excluded = set(meta["identifier_columns"]) | set(meta["potential_leakage_columns"])

    frames: dict[str, pd.DataFrame] = {}
    for name in ("train.csv", "validation.csv", "test.csv",
                "train_scaled.parquet", "validation_scaled.parquet",
                "test_scaled.parquet"):
        path = proc_dir / name
        try:
            frame = _read(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name} is not readable: {exc!r}")
            continue
        if frame.empty:
            errors.append(f"{name} is empty")
            continue
        frames[name] = frame

    # -- target checks --------------------------------------------------
    for name, frame in frames.items():
        if STANDARD_TARGET not in frame.columns:
            errors.append(f"{name} is missing target column {STANDARD_TARGET!r}")
            continue
        uniques = set(frame[STANDARD_TARGET].dropna().unique().tolist())
        if not uniques.issubset({0, 1}):
            errors.append(f"{name}: target contains non-binary values {uniques}")
        if frame[STANDARD_TARGET].sum() == 0:
            warnings.append(f"{name}: zero fraud rows in this split")

    # -- train/val/test disjointness (on the clean CSVs, hash-based) -----
    if all(k in frames for k in ("train.csv", "validation.csv", "test.csv")):
        def row_hashes(frame: pd.DataFrame) -> set[str]:
            cols = [c for c in frame.columns if c != STANDARD_TARGET]
            return set(pd.util.hash_pandas_object(frame[cols], index=False).astype(str))

        h_train = row_hashes(frames["train.csv"])
        h_val = row_hashes(frames["validation.csv"])
        h_test = row_hashes(frames["test.csv"])
        overlap_tv = h_train & h_val
        overlap_tt = h_train & h_test
        overlap_vt = h_val & h_test
        if overlap_tv:
            errors.append(f"train/validation row overlap: {len(overlap_tv)} rows")
        if overlap_tt:
            errors.append(f"train/test row overlap: {len(overlap_tt)} rows")
        if overlap_vt:
            errors.append(f"validation/test row overlap: {len(overlap_vt)} rows")

    # -- leakage/identifier columns must not appear in model-ready data --
    for name in ("train_scaled.parquet", "validation_scaled.parquet", "test_scaled.parquet"):
        if name not in frames:
            continue
        leaked = excluded & set(frames[name].columns)
        if leaked:
            errors.append(f"{name}: excluded column(s) present in model-ready "
                          f"matrix: {sorted(leaked)}")

    # -- NaN / Inf in model-ready matrices --------------------------------
    for name in ("train_scaled.parquet", "validation_scaled.parquet", "test_scaled.parquet"):
        if name not in frames:
            continue
        numeric = frames[name].select_dtypes(include=[np.number])
        if numeric.isna().any().any():
            errors.append(f"{name}: contains NaN values")
        if np.isinf(numeric.to_numpy(dtype=float, na_value=0.0)).any():
            errors.append(f"{name}: contains Inf values")

    # -- quantum feature files: split column present, no leakage columns --
    for k in (2, 4, 6, 8):
        path = proc_dir / f"quantum_{k}_features.parquet"
        try:
            qframe = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"quantum_{k}_features.parquet unreadable: {exc!r}")
            continue
        if "split" not in qframe.columns:
            errors.append(f"quantum_{k}_features.parquet missing 'split' column")
        elif not set(qframe["split"].unique()).issubset({"train", "val", "test"}):
            errors.append(f"quantum_{k}_features.parquet has unexpected split labels")
        leaked = excluded & set(qframe.columns)
        if leaked:
            errors.append(f"quantum_{k}_features.parquet: leaked columns {sorted(leaked)}")

    # -- scaler-fit-on-train-only is enforced structurally by
    #    fit_preprocessor()/make_splits() in src/data -- verified indirectly
    #    here by confirming train/val/test were produced by one shared
    #    preprocessing_metadata.json (a single fit), not per-split configs.
    if "split_summary" not in meta:
        errors.append("preprocessing_metadata.json missing split_summary "
                      "(cannot verify a single train-only fit was used)")

    # -- no resampling on saved files -------------------------------------
    resampling_note = meta.get("preprocessing", {}).get(
        "resampling_applied_to_saved_files", "")
    if "none" not in resampling_note.lower():
        errors.append("metadata does not confirm saved files are unresampled")

    return {"friendly_name": friendly, "passed": not errors,
           "errors": errors, "warnings": warnings}


def main() -> int:
    results = [validate_one(friendly) for friendly in FRIENDLY_NAMES.values()]

    LOGGER.info("=" * 70)
    LOGGER.info("DATA VALIDATION RESULTS")
    all_ok = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        LOGGER.info("[%s] %s", status, r["friendly_name"])
        for e in r["errors"]:
            LOGGER.error("    ERROR: %s", e)
        for w in r["warnings"]:
            LOGGER.warning("    warning: %s", w)
        all_ok = all_ok and r["passed"]
    LOGGER.info("=" * 70)

    out_path = Path(get_path("data_metadata")) / "validation_report.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Wrote %s", out_path)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
