"""Aggregate per-dataset preprocessing_metadata.json into project-wide
data/metadata/ artifacts:

    data/metadata/leakage_audit.md      -- one row per suspicious column
    data/metadata/dataset_manifest.json -- everything, per dataset
    data/metadata/dataset_summary.csv   -- one row per dataset

Reads ONLY what scripts/preprocess_datasets.py already wrote to disk; never
invents a row count, fraud percentage or column name.

    python scripts/generate_data_metadata.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.utils.config import get_path, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from scripts.download_datasets import FRIENDLY_NAMES  # noqa: E402

LOGGER = get_logger("generate_data_metadata")


def load_all_metadata() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    processed_root = Path(get_path("data_processed"))
    for internal_name, friendly in FRIENDLY_NAMES.items():
        path = processed_root / friendly / "preprocessing_metadata.json"
        if not path.exists():
            LOGGER.warning("No preprocessing_metadata.json for %s (%s); "
                           "run scripts/preprocess_datasets.py first",
                           internal_name, friendly)
            continue
        out[internal_name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def build_leakage_audit(all_meta: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Leakage Audit",
        "",
        "Generated from the actual column classification and target-correlation "
        "screen each dataset went through in `scripts/preprocess_datasets.py` "
        "(via the existing, unmodified `src/data/registry.py` and "
        "`src/data/validators.py`). Every row below reflects a column that "
        "was genuinely excluded or flagged -- none are hypothetical.",
        "",
        "| Dataset | Column | Risk | Decision | Reason |",
        "|---|---|---|---|---|",
    ]

    for internal_name, meta in all_meta.items():
        friendly = meta["friendly_name"]

        for col in meta.get("identifier_columns", []):
            lines.append(f"| {friendly} | `{col}` | Identifier | Excluded from "
                         "features | Uniquely or near-uniquely identifies a "
                         "row, card or customer; a model could memorise "
                         "individuals instead of learning fraud patterns. |")

        for col in meta.get("potential_leakage_columns", []):
            lines.append(f"| {friendly} | `{col}` | Leakage (name pattern) | "
                         "Excluded from features | Column name matches a "
                         "post-outcome pattern (e.g. chargeback/dispute/"
                         "investigation/resolution); would only be known "
                         "after a fraud determination, not at scoring time. |")

        for col, corr in meta.get("suspicious_target_correlations", {}).items():
            lines.append(f"| {friendly} | `{col}` | Leakage (correlation "
                         f"{corr}) | Hard validation error -- excluded | "
                         "Correlates >=0.95 with the target; almost certainly "
                         "a transformed copy of the label rather than a "
                         "genuine predictive feature. |")

        for col in meta.get("unusable_columns", []):
            lines.append(f"| {friendly} | `{col}` | Unusable (constant / "
                         "high-cardinality free text) | Excluded from "
                         "features | Either carries zero variance or exceeds "
                         "the categorical-cardinality cap (free-text name/"
                         "city/job fields); neither generalises. |")

        errors = meta.get("validation", {}).get("errors", [])
        for err in errors:
            lines.append(f"| {friendly} | (dataset-level) | Validation error | "
                         f"Blocked | {err} |")
        warnings = meta.get("validation", {}).get("warnings", [])
        for warn in warnings:
            lines.append(f"| {friendly} | (dataset-level) | Validation warning | "
                         f"Recorded, not blocking | {warn} |")

    lines += ["", "## Duplicate-row leakage (train/test overlap)", "",
             "| Dataset | Duplicates removed before split |", "|---|---|"]
    for internal_name, meta in all_meta.items():
        lines.append(f"| {meta['friendly_name']} | "
                     f"{meta['duplicates_dropped_before_split']:,} |")

    return "\n".join(lines) + "\n"


def build_manifest(all_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ds_cfg = load_config("datasets")["datasets"]
    manifest: dict[str, Any] = {}
    for internal_name, meta in all_meta.items():
        split = meta["split_summary"]
        manifest[internal_name] = {
            "friendly_name": meta["friendly_name"],
            "kaggle_id": meta["kaggle_id"],
            "original_files": [Path(meta["raw_file"]).name],
            "local_raw_path": str(Path(get_path("data_raw")) / meta["friendly_name"]),
            "local_processed_path": str(Path(get_path("data_processed")) / meta["friendly_name"]),
            "row_count": (split["train"]["rows"] + split["val"]["rows"]
                         + split["test"]["rows"]),
            "feature_count": len(meta["all_model_ready_feature_names"]),
            "original_target_column": meta["original_target_column"],
            "standardised_target_column": meta["standardised_target"],
            "fraud_count": (split["train"]["fraud"] + split["val"]["fraud"]
                           + split["test"]["fraud"]),
            "legitimate_count": ((split["train"]["rows"] - split["train"]["fraud"])
                                 + (split["val"]["rows"] - split["val"]["fraud"])
                                 + (split["test"]["rows"] - split["test"]["fraud"])),
            "train_size": split["train"]["rows"],
            "validation_size": split["val"]["rows"],
            "test_size": split["test"]["rows"],
            "split_strategy": meta["split_strategy"],
            "categorical_feature_count": len(meta["categorical_features"]),
            "numerical_feature_count": len(meta["numeric_features"]),
            "temporal_features_engineered": meta["temporal_features_engineered"],
            "removed_columns": meta["removed_columns"],
            "quantum_feature_sets": meta["quantum_feature_sets"],
            "preprocessing_strategy": meta["preprocessing"],
            "output_files": meta.get("output_files", []),
            "validation_passed": meta["validation"]["passed"],
            "seed": meta["seed"],
            "processed_at": meta["processed_at"],
        }
    return manifest


def build_summary_csv(manifest: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for internal_name, m in manifest.items():
        rows.append({
            "dataset": internal_name,
            "friendly_name": m["friendly_name"],
            "kaggle_id": m["kaggle_id"],
            "row_count": m["row_count"],
            "feature_count": m["feature_count"],
            "target_column": m["standardised_target_column"],
            "fraud_count": m["fraud_count"],
            "legitimate_count": m["legitimate_count"],
            "fraud_percentage": round(
                100.0 * m["fraud_count"] / m["row_count"], 6) if m["row_count"] else 0.0,
            "train_size": m["train_size"],
            "validation_size": m["validation_size"],
            "test_size": m["test_size"],
            "split_strategy": m["split_strategy"],
            "categorical_features": m["categorical_feature_count"],
            "numerical_features": m["numerical_feature_count"],
            "n_removed_columns": len(m["removed_columns"]),
            "quantum_feature_counts": "|".join(m["quantum_feature_sets"].keys()),
            "validation_passed": m["validation_passed"],
        })
    return pd.DataFrame(rows)


def main() -> int:
    all_meta = load_all_metadata()
    if not all_meta:
        LOGGER.error("No processed datasets found; nothing to summarise")
        return 1

    metadata_dir = Path(get_path("data_metadata"))
    metadata_dir.mkdir(parents=True, exist_ok=True)

    (metadata_dir / "leakage_audit.md").write_text(
        build_leakage_audit(all_meta), encoding="utf-8")
    LOGGER.info("Wrote %s", metadata_dir / "leakage_audit.md")

    manifest = build_manifest(all_meta)
    (metadata_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Wrote %s", metadata_dir / "dataset_manifest.json")

    summary = build_summary_csv(manifest)
    summary.to_csv(metadata_dir / "dataset_summary.csv", index=False)
    LOGGER.info("Wrote %s (%d datasets)", metadata_dir / "dataset_summary.csv",
                len(summary))

    return 0 if len(all_meta) == len(FRIENDLY_NAMES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
