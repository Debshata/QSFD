"""Download the four Kaggle fraud datasets and populate data/raw/.

Reuses ``src.data.download`` (unmodified) to resolve/cache each dataset via
kagglehub, then physically copies the resolved source file into this
project's own ``data/raw/<friendly_name>/`` directory. The kagglehub cache
copy is left untouched -- this is an additional, explicit local copy, not a
move.

    python scripts/download_datasets.py

Exits non-zero only if every dataset failed to resolve; a partial failure
(some datasets available, others not) is reported but does not abort.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import download_all  # noqa: E402
from src.utils.config import get_path, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger("download_datasets")

# Internal registry name (config/datasets.yaml) -> requested local folder name.
# This mapping is local to these data-preparation scripts only; it does not
# touch config/datasets.yaml or any code the training pipelines depend on.
FRIENDLY_NAMES = {
    "creditcardfraud": "creditcardfraud_ulb",
    "fraud_detection_sim": "fraud_detection_kartik",
    "creditcard_2023": "credit_card_fraud_2023",
    "synthetic_cc_fraud": "synthetic_transactions",
}


def populate_raw_folder(name: str, record: dict) -> dict:
    """Copy one resolved dataset file into data/raw/<friendly_name>/."""
    friendly = FRIENDLY_NAMES.get(name, name)
    dest_dir = Path(get_path("data_raw")) / friendly
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not record.get("ok"):
        return {"name": name, "friendly_name": friendly, "populated": False,
                "reason": record.get("error", "download failed")}

    src_file = Path(record["file"])
    dest_file = dest_dir / src_file.name

    if dest_file.exists() and dest_file.stat().st_size == src_file.stat().st_size:
        LOGGER.info("%s: %s already present and correctly sized, skipping copy",
                    name, dest_file.name)
    else:
        LOGGER.info("%s: copying %s -> %s (%.1f MB)", name, src_file.name,
                    dest_dir, src_file.stat().st_size / 1e6)
        t0 = time.perf_counter()
        shutil.copy2(src_file, dest_file)
        LOGGER.info("%s: copy finished in %.1fs", name, time.perf_counter() - t0)

    return {
        "name": name, "friendly_name": friendly, "populated": True,
        "raw_path": str(dest_dir), "raw_file": str(dest_file),
        "size_mb": round(dest_file.stat().st_size / 1e6, 2),
        "source_cache_path": str(src_file),
        "kaggle_id": record.get("kaggle_id"),
    }


def main() -> int:
    LOGGER.info("Resolving all four Kaggle datasets via kagglehub...")
    manifest = download_all()

    results = []
    for name, record in manifest.items():
        results.append(populate_raw_folder(name, record))

    LOGGER.info("=" * 60)
    LOGGER.info("Download summary:")
    ok = 0
    for r in results:
        if r["populated"]:
            ok += 1
            LOGGER.info("  [OK]   %-22s -> %s (%.1f MB)",
                        r["name"], r["raw_path"], r["size_mb"])
        else:
            LOGGER.error("  [FAIL] %-22s -> %s", r["name"], r["reason"])
    LOGGER.info("%d/%d datasets populated in data/raw/", ok, len(results))

    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
