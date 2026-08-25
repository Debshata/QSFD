"""Master data-preparation pipeline.

    Check dependencies
          |
    Download datasets (-> data/raw/)
          |
    Preprocess + split + leakage screen (-> data/processed/)
          |
    Generate metadata (leakage_audit.md, dataset_manifest.json,
                       dataset_summary.csv)
          |
    Validate outputs
          |
    Check compatibility with existing Classical ML / QML / QBoost / CLT
    loaders (read-only checks; no model code is modified or trained)

Does not touch any model/report code. Running this script reproduces the
complete local data environment from scratch.

    python scripts/prepare_data.py
    python scripts/prepare_data.py --skip-download   # data/raw/ already populated
    python scripts/prepare_data.py --dataset creditcardfraud_ulb
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger("prepare_data")

REQUIRED_PACKAGES = ["kagglehub", "pandas", "numpy", "sklearn", "pyarrow"]


def check_dependencies() -> bool:
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        LOGGER.error("Missing required packages: %s. Install with: "
                     "pip install %s", missing, " ".join(missing))
        return False
    LOGGER.info("All required packages present: %s", REQUIRED_PACKAGES)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the local data environment")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dataset", default=None,
                        help="process only this dataset (friendly or internal name)")
    args = parser.parse_args()

    t_start = time.perf_counter()
    stage_results: dict[str, str] = {}

    LOGGER.info("#" * 70)
    LOGGER.info("DATA PREPARATION PIPELINE")
    LOGGER.info("#" * 70)

    if not check_dependencies():
        return 1
    stage_results["dependencies"] = "ok"

    if not args.skip_download:
        LOGGER.info("--- Stage: download ---")
        from scripts.download_datasets import main as download_main

        stage_results["download"] = "ok" if download_main() == 0 else "failed"
        if stage_results["download"] == "failed":
            LOGGER.error("Download stage failed for every dataset; aborting")
            return 1
    else:
        LOGGER.info("--- Stage: download SKIPPED (--skip-download) ---")
        stage_results["download"] = "skipped"

    LOGGER.info("--- Stage: preprocess ---")
    from scripts.preprocess_datasets import main as preprocess_main

    argv_backup = sys.argv
    sys.argv = ["preprocess_datasets.py"]
    if args.dataset:
        sys.argv += ["--dataset", args.dataset]
    try:
        stage_results["preprocess"] = "ok" if preprocess_main() == 0 else "failed"
    finally:
        sys.argv = argv_backup

    LOGGER.info("--- Stage: metadata ---")
    from scripts.generate_data_metadata import main as metadata_main

    stage_results["metadata"] = "ok" if metadata_main() == 0 else "failed"

    LOGGER.info("--- Stage: validate ---")
    from scripts.validate_data import main as validate_main

    stage_results["validate"] = "ok" if validate_main() == 0 else "failed"

    LOGGER.info("--- Stage: loader compatibility ---")
    from scripts.check_loader_compatibility import main as compat_main

    stage_results["loader_compatibility"] = "ok" if compat_main() == 0 else "failed"

    total = time.perf_counter() - t_start
    LOGGER.info("#" * 70)
    LOGGER.info("DATA PREPARATION SUMMARY (%.1fs total)", total)
    for stage, status in stage_results.items():
        LOGGER.info("  %-22s %s", stage, status.upper())
    LOGGER.info("#" * 70)

    required = ("dependencies", "preprocess", "metadata", "validate")
    failed = [s for s in required if stage_results.get(s) == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
