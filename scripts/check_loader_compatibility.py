"""Verify the existing Classical ML / QML / QBoost / CLT loaders still work.

This project's model pipelines (src/classical_ml/train.py, src/qml/train.py,
src/qml_experimental/data.py) load directly from the raw file resolved by
src.data.download's manifest and do their OWN in-memory preprocessing on
every run -- they do not read data/processed/*.csv|parquet. That design is
unchanged by this data-population task. This script verifies:

    1. The existing loader path (src.data.registry/loaders, unmodified)
       still resolves and loads every dataset successfully -- i.e. adding
       data/raw/<friendly>/ copies has not broken anything.
    2. Every dataset's file the manifest points to is now ALSO present as a
       physical local copy in data/raw/, so a loader could be pointed at the
       local copy instead of the kagglehub cache without any code change.
    3. The newly written data/processed/ files are structurally consumable
       (readable, correct target, right shape) by any future loader, without
       requiring changes to the model pipelines.

No model code is imported for training here -- only the shared data-layer
functions the pipelines already depend on.

    python scripts/check_loader_compatibility.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.download import load_manifest  # noqa: E402
from src.data.loaders import load_dataset  # noqa: E402
from src.data.registry import available_datasets  # noqa: E402
from src.utils.config import get_path, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from scripts.download_datasets import FRIENDLY_NAMES  # noqa: E402

LOGGER = get_logger("check_loader_compatibility")


def check_existing_loader_path() -> dict[str, bool]:
    """Classical ML / QML / QBoost / CLT all funnel through this exact call."""
    results = {}
    available = available_datasets()
    for name, rec in available.items():
        try:
            loaded = load_dataset(name, rec["entry"], rec["file"], max_rows=1000)
            ok = len(loaded.X) > 0 and loaded.y.isin([0, 1]).all()
            results[name] = ok
            LOGGER.info("[%s] existing loader path: %s (%d rows, target ok=%s)",
                        "OK" if ok else "FAIL", name, len(loaded.X), ok)
        except Exception as exc:  # noqa: BLE001
            results[name] = False
            LOGGER.error("[FAIL] existing loader path: %s -> %r", name, exc)
    return results


def check_local_raw_copies_available() -> dict[str, bool]:
    manifest = load_manifest()
    results = {}
    for name, friendly in FRIENDLY_NAMES.items():
        cache_file = manifest.get(name, {}).get("file")
        local_dir = Path(get_path("data_raw")) / friendly
        local_files = list(local_dir.glob("*.csv"))
        ok = bool(local_files) and cache_file is not None
        results[name] = ok
        LOGGER.info("[%s] local raw copy present for %s: %s",
                    "OK" if ok else "FAIL", name,
                    local_files[0].name if local_files else "MISSING")
    return results


def check_processed_files_consumable() -> dict[str, bool]:
    results = {}
    for name, friendly in FRIENDLY_NAMES.items():
        proc_dir = Path(get_path("data_processed")) / friendly
        train_path = proc_dir / "train.csv"
        scaled_path = proc_dir / "train_scaled.parquet"
        try:
            train = pd.read_csv(train_path, nrows=5)
            scaled = pd.read_parquet(scaled_path)
            ok = ("is_fraud" in train.columns and "is_fraud" in scaled.columns
                  and scaled.shape[1] > 1)
            results[name] = ok
            LOGGER.info("[%s] processed files consumable for %s "
                        "(train cols=%d, scaled shape=%s)",
                        "OK" if ok else "FAIL", name, len(train.columns),
                        scaled.shape)
        except Exception as exc:  # noqa: BLE001
            results[name] = False
            LOGGER.error("[FAIL] processed files for %s -> %r", name, exc)
    return results


def main() -> int:
    LOGGER.info("=" * 70)
    LOGGER.info("1. Existing loader path (Classical ML / QML / QBoost / CLT "
                "all use this) -- unmodified, must still work")
    existing = check_existing_loader_path()

    LOGGER.info("=" * 70)
    LOGGER.info("2. Local data/raw/ physical copies available as an alternative "
                "source")
    local = check_local_raw_copies_available()

    LOGGER.info("=" * 70)
    LOGGER.info("3. New data/processed/ files are structurally consumable")
    processed = check_processed_files_consumable()

    report = {"existing_loader_path": existing, "local_raw_copies": local,
              "processed_files_consumable": processed}
    out_path = Path(get_path("data_metadata")) / "loader_compatibility_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=bool), encoding="utf-8")
    LOGGER.info("Wrote %s", out_path)

    all_ok = all(existing.values()) and all(local.values()) and all(processed.values())
    LOGGER.info("=" * 70)
    LOGGER.info("Overall compatibility: %s", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
